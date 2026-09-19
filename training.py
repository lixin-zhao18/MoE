"""Training utilities for the streamflow models."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress display
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


TARGET_STD_EPSILON = 1e-6

__all__ = [
    "EarlyStoppingUpdate",
    "normalized_masked_mse",
    "moe_loss",
    "compute_moe_temperature",
    "create_adam_optimizer",
    "create_multistep_scheduler",
    "create_grad_scaler",
    "autocast_context",
    "train_global_lstm_epoch",
    "evaluate_global_lstm_loss",
    "train_moe_epoch",
    "evaluate_moe_loss",
    "train_fixed_routing_moe_epoch",
    "evaluate_fixed_routing_moe_loss",
    "update_early_stopping",
    "save_model_checkpoint",
]


def normalized_masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_stds: torch.Tensor,
    target_mask: Optional[torch.Tensor] = None,
    epsilon: float = TARGET_STD_EPSILON,
) -> torch.Tensor:
    """Compute the basin-normalized masked mean squared error.

    For basin ``b`` and valid forecast element ``t``, the contribution is
    ``(prediction[b,t] - target[b,t])**2 / (target_std[b] + epsilon)**2``.
    Contributions are summed and divided by the total number of valid elements.
    """

    target_stds_broadcast = target_stds.unsqueeze(-1)
    if target_mask is None:
        valid = torch.isfinite(targets)
    else:
        valid = target_mask.to(dtype=torch.bool) & torch.isfinite(targets)
    if not torch.any(valid):
        return predictions.sum() * 0.0

    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    safe_predictions = torch.where(valid, predictions, torch.zeros_like(predictions))
    squared_errors = (safe_predictions - safe_targets) ** 2
    target_variances = (target_stds_broadcast + epsilon) ** 2
    normalized_squared_errors = squared_errors / target_variances
    valid_float = valid.to(dtype=normalized_squared_errors.dtype)
    return (
        (normalized_squared_errors * valid_float).sum()
        / valid_float.sum().clamp_min(1.0)
    )


def moe_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    target_stds: torch.Tensor,
    load_balance: torch.Tensor,
    load_balance_weight: float,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine normalized prediction loss and load-balancing regularization."""

    prediction_loss = normalized_masked_mse(
        predictions,
        targets,
        target_stds,
        target_mask=target_mask,
    )
    return prediction_loss + load_balance_weight * load_balance


def compute_moe_temperature(
    epoch: int,
    start_temperature: float,
    end_temperature: float,
    warmup_epochs: int,
    decay_end_epoch: int = 50,
) -> float:
    """Return the epoch-indexed linear gate-temperature schedule."""

    if epoch <= warmup_epochs:
        return start_temperature
    decay_start_epoch = warmup_epochs + 1
    if decay_end_epoch <= decay_start_epoch:
        return end_temperature if epoch >= decay_end_epoch else start_temperature
    if epoch >= decay_end_epoch:
        return end_temperature
    progress = (epoch - decay_start_epoch) / (decay_end_epoch - decay_start_epoch)
    return start_temperature + (end_temperature - start_temperature) * progress


def create_adam_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.Adam:
    """Create the Adam optimizer used for model training."""

    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def create_multistep_scheduler(
    optimizer: torch.optim.Optimizer,
    milestones: Sequence[int],
    gamma: float,
) -> Optional[torch.optim.lr_scheduler.MultiStepLR]:
    """Create a MultiStepLR, or ``None`` when no milestones are configured."""

    if not milestones:
        return None
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(milestones),
        gamma=gamma,
    )


def autocast_context(device: torch.device, amp: bool):
    if not (amp and device.type == "cuda"):
        return nullcontext
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "autocast"):
        autocast_function = amp_module.autocast
        try:
            return partial(autocast_function, device_type="cuda")
        except TypeError:
            return autocast_function
    cuda_amp = getattr(torch.cuda, "amp", None)
    if cuda_amp is not None and hasattr(cuda_amp, "autocast"):
        return cuda_amp.autocast
    return nullcontext


def create_grad_scaler(device: torch.device, amp: bool):
    """Create a CUDA GradScaler; return ``None`` on CPU or without AMP."""

    if not (amp and device.type == "cuda"):
        return None
    grad_scaler_class = None
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        grad_scaler_class = amp_module.GradScaler
    elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        grad_scaler_class = torch.cuda.amp.GradScaler
    if grad_scaler_class is None:
        return None
    try:
        return grad_scaler_class(device_type="cuda")
    except TypeError:
        try:
            return grad_scaler_class()
        except TypeError:
            signature = inspect.signature(grad_scaler_class)
            kwargs: Dict[str, object] = {}
            if "enabled" in signature.parameters:
                kwargs["enabled"] = True
            if "init_scale" in signature.parameters:
                kwargs.setdefault("init_scale", 65536.0)
            return grad_scaler_class(**kwargs)


def train_global_lstm_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: Optional[float] = None,
    amp: bool = False,
    scaler=None,
    target_log1p: bool = False,
) -> float:
    """Run one Global LSTM training epoch."""

    model.train()
    autocast_context_manager = autocast_context(device, amp)
    if scaler is None:
        scaler = create_grad_scaler(device, amp)
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(loader, desc="Train", dynamic_ncols=True, leave=False):
        non_blocking = device.type == "cuda"
        encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
        decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        target_mask = batch.get("target_mask")
        if target_mask is not None:
            target_mask = target_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=non_blocking,
            )
        else:
            target_mask = torch.isfinite(targets)
        target_std = batch["target_std"].to(device, non_blocking=non_blocking)
        safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))

        with autocast_context_manager():
            predictions = model(encoder, decoder)
            targets_for_loss = (
                torch.log1p(torch.clamp_min(safe_targets, 0.0))
                if target_log1p
                else safe_targets
            )
            loss = normalized_masked_mse(
                predictions,
                targets_for_loss,
                target_std,
                target_mask=target_mask,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if gradient_clip and gradient_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        total_loss += loss.detach().float().item() * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


def evaluate_global_lstm_loss(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    amp: bool = False,
    target_log1p: bool = False,
) -> float:
    """Evaluate normalized prediction loss for the Global LSTM."""

    model.eval()
    autocast_context_manager = autocast_context(device, amp)
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validate", dynamic_ncols=True, leave=False):
            non_blocking = device.type == "cuda"
            encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
            decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)
            target_mask = batch.get("target_mask")
            if target_mask is not None:
                target_mask = target_mask.to(
                    device=device,
                    dtype=torch.bool,
                    non_blocking=non_blocking,
                )
            else:
                target_mask = torch.isfinite(targets)
            target_std = batch["target_std"].to(device, non_blocking=non_blocking)
            safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))
            with autocast_context_manager():
                predictions = model(encoder, decoder)
                targets_for_loss = (
                    torch.log1p(torch.clamp_min(safe_targets, 0.0))
                    if target_log1p
                    else safe_targets
                )
                loss = normalized_masked_mse(
                    predictions,
                    targets_for_loss,
                    target_std,
                    target_mask=target_mask,
                )
            batch_size = targets.size(0)
            total_loss += loss.detach().float().item() * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)


def train_moe_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: float,
    amp: bool,
    scaler,
    target_log1p: bool,
    load_balance_weight: float,
) -> float:
    """Run one MoE training epoch.

    The objective combines normalized runoff prediction loss and the
    load-balancing regularization.
    """

    model.train()
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_context_manager = autocast_context(device, amp)

    for batch in tqdm(loader, desc="Train", dynamic_ncols=True, leave=False):
        non_blocking = device.type == "cuda"
        encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
        decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        target_mask = batch.get("target_mask")
        if target_mask is not None:
            target_mask = target_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=non_blocking,
            )
        else:
            target_mask = torch.isfinite(targets)
        static_features = batch["static_features"].to(
            device,
            non_blocking=non_blocking,
        )
        target_std = batch["target_std"].to(device, non_blocking=non_blocking)
        safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))

        with autocast_context_manager():
            predictions, balance_loss, _routing = model(
                encoder,
                decoder,
                static_features,
            )
            targets_for_loss = (
                torch.log1p(torch.clamp_min(safe_targets, 0.0))
                if target_log1p
                else safe_targets
            )
            loss = moe_loss(
                predictions,
                targets_for_loss,
                target_std,
                balance_loss,
                load_balance_weight=load_balance_weight,
                target_mask=target_mask,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        total_loss += loss.detach().float().item() * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


def evaluate_moe_loss(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    amp: bool,
    target_log1p: bool,
    load_balance_weight: float,
) -> float:
    """Evaluate the MoE objective, including load balancing."""

    model.eval()
    total_loss = 0.0
    total_count = 0
    autocast_context_manager = autocast_context(device, amp)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            non_blocking = device.type == "cuda"
            encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
            decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)
            target_mask = batch.get("target_mask")
            if target_mask is not None:
                target_mask = target_mask.to(
                    device=device,
                    dtype=torch.bool,
                    non_blocking=non_blocking,
                )
            else:
                target_mask = torch.isfinite(targets)
            static_features = batch["static_features"].to(
                device,
                non_blocking=non_blocking,
            )
            target_std = batch["target_std"].to(device, non_blocking=non_blocking)
            safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))
            with autocast_context_manager():
                predictions, balance_loss, _routing = model(
                    encoder,
                    decoder,
                    static_features,
                )
                targets_for_loss = (
                    torch.log1p(torch.clamp_min(safe_targets, 0.0))
                    if target_log1p
                    else safe_targets
                )
                loss = moe_loss(
                    predictions,
                    targets_for_loss,
                    target_std,
                    balance_loss,
                    load_balance_weight=load_balance_weight,
                    target_mask=target_mask,
                )
            batch_size = targets.size(0)
            total_loss += loss.detach().float().item() * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)


def train_fixed_routing_moe_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip: float,
    amp: bool,
    scaler,
    target_log1p: bool,
) -> float:
    """Train the fixed-routing MoE for one epoch.

    The loss is the same normalized runoff prediction loss used by the other
    models. No gate or load-balancing objective is used because routing labels
    are fixed before training.
    """

    model.train()
    total_loss = 0.0
    total_count = 0
    optimizer.zero_grad(set_to_none=True)
    autocast_context_manager = autocast_context(device, amp)

    for batch in tqdm(loader, desc="Train", dynamic_ncols=True, leave=False):
        non_blocking = device.type == "cuda"
        encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
        decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
        targets = batch["targets"].to(device, non_blocking=non_blocking)
        target_mask = batch.get("target_mask")
        if target_mask is not None:
            target_mask = target_mask.to(
                device=device, dtype=torch.bool, non_blocking=non_blocking
            )
        else:
            target_mask = torch.isfinite(targets)
        static_features = batch["static_features"].to(
            device, non_blocking=non_blocking
        )
        routing_labels = batch["routing_label"].to(
            device, non_blocking=non_blocking
        )
        target_std = batch["target_std"].to(device, non_blocking=non_blocking)
        safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))

        with autocast_context_manager():
            predictions, _balance_loss, _routing = model(
                encoder, decoder, static_features, routing_labels
            )
            targets_for_loss = (
                torch.log1p(torch.clamp_min(safe_targets, 0.0))
                if target_log1p
                else safe_targets
            )
            loss = normalized_masked_mse(
                predictions,
                targets_for_loss,
                target_std,
                target_mask=target_mask,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_size = targets.size(0)
        total_loss += loss.detach().float().item() * batch_size
        total_count += batch_size
    return total_loss / max(1, total_count)


def evaluate_fixed_routing_moe_loss(
    model: nn.Module,
    loader: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    amp: bool,
    target_log1p: bool,
) -> float:
    """Evaluate fixed-routing MoE prediction loss."""

    model.eval()
    total_loss = 0.0
    total_count = 0
    autocast_context_manager = autocast_context(device, amp)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", dynamic_ncols=True, leave=False):
            non_blocking = device.type == "cuda"
            encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
            decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].to(device, non_blocking=non_blocking)
            target_mask = batch.get("target_mask")
            if target_mask is not None:
                target_mask = target_mask.to(
                    device=device, dtype=torch.bool, non_blocking=non_blocking
                )
            else:
                target_mask = torch.isfinite(targets)
            static_features = batch["static_features"].to(
                device, non_blocking=non_blocking
            )
            routing_labels = batch["routing_label"].to(
                device, non_blocking=non_blocking
            )
            target_std = batch["target_std"].to(device, non_blocking=non_blocking)
            safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))
            with autocast_context_manager():
                predictions, _balance_loss, _routing = model(
                    encoder, decoder, static_features, routing_labels
                )
                targets_for_loss = (
                    torch.log1p(torch.clamp_min(safe_targets, 0.0))
                    if target_log1p
                    else safe_targets
                )
                loss = normalized_masked_mse(
                    predictions,
                    targets_for_loss,
                    target_std,
                    target_mask=target_mask,
                )
            batch_size = targets.size(0)
            total_loss += loss.detach().float().item() * batch_size
            total_count += batch_size
    return total_loss / max(1, total_count)


@dataclass(frozen=True)
class EarlyStoppingUpdate:
    """State transition produced by the strict-improvement rule."""

    best_validation_loss: float
    epochs_without_improvement: int
    improved: bool
    should_stop: bool


def update_early_stopping(
    validation_loss: float,
    best_validation_loss: float,
    epochs_without_improvement: int,
    patience: int,
) -> EarlyStoppingUpdate:
    """Apply strict ``<`` improvement and the configured patience threshold."""

    if validation_loss < best_validation_loss:
        return EarlyStoppingUpdate(validation_loss, 0, True, False)
    counter = epochs_without_improvement + 1
    should_stop = bool(patience and counter >= patience)
    return EarlyStoppingUpdate(best_validation_loss, counter, False, should_stop)


def save_model_checkpoint(model: nn.Module, epoch: int, path: Path) -> None:
    """Save the model state and its epoch."""

    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, path)
