"""Training entry point for the streamflow models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.lstm import GlobalLSTM
from models.fixed_routing_moe import FixedRoutingMoE
from models.moe import MoE
from training import (
    compute_moe_temperature,
    create_adam_optimizer,
    create_grad_scaler,
    create_multistep_scheduler,
    evaluate_moe_loss,
    evaluate_fixed_routing_moe_loss,
    evaluate_global_lstm_loss,
    save_model_checkpoint,
    train_moe_epoch,
    train_fixed_routing_moe_epoch,
    train_global_lstm_epoch,
    update_early_stopping,
)


SUPPORTED_MODELS = {
    "global_lstm",
    "moe",
    "cluster_specific_lstms",
    "fixed_routing_moe",
}


@dataclass
class PreparedData:
    """Objects needed by the train and evaluation entry points."""

    feature_names: list[str]
    static_feature_names: list[str]
    feature_scaler: Any
    target_standard_deviations: Dict[str, float]
    model_input_arrays: Dict[str, Any]
    static_features: Dict[str, np.ndarray]
    datasets: tuple[Any, Any, Any]


def load_config(path: Path) -> Dict[str, Any]:
    """Read a JSON run configuration without supplying scientific defaults."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("The configuration root must be a JSON object.")
    return payload


def resolve_model_type(config: Dict[str, Any]) -> str:
    """Resolve the configured model type."""

    model_type = config.get("model")
    if model_type is None:
        raise KeyError("Config must define 'model'.")
    model_type = str(model_type)
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model '{model_type}'. Expected one of {sorted(SUPPORTED_MODELS)}."
        )
    return model_type


def resolve_static_attribute_config(
    config: Dict[str, Any],
) -> tuple[Path, str, list[str], Optional[str]]:
    """Resolve the static-attribute configuration."""

    return (
        Path(config["attributes_file"]),
        str(config["basin_id_column"]),
        list(config["static_attribute_columns"]),
        config.get("attributes_sheet"),
    )


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(
    config: Dict[str, Any],
    input_size: int,
    static_input_dim: int,
    device: torch.device,
) -> nn.Module:
    """Construct the configured model."""

    model_type = resolve_model_type(config)
    if model_type == "moe":
        model = MoE(
            input_size=input_size,
            hidden_size=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
            static_input_dim=static_input_dim,
            static_embed_dim=int(config["static_embed_dim"]),
            gating_hidden_dim=int(config["gating_hidden_dim"]),
            num_experts=int(config["num_experts"]),
            top_k=int(config["top_k"]),
            temperature=float(config["initial_temperature"]),
        )
    elif model_type == "fixed_routing_moe":
        model = FixedRoutingMoE(
            input_size=input_size,
            hidden_size=int(config["hidden_size"]),
            dropout=float(config["dropout"]),
            static_input_dim=static_input_dim,
            static_embed_dim=int(config["static_embed_dim"]),
            num_experts=int(config["num_experts"]),
        )
    else:
        model = GlobalLSTM(
            input_size=input_size,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config.get("num_layers", 1)),
        )
    return model.to(device)


def make_data_loader(
    dataset: Any,
    config: Dict[str, Any],
    device: torch.device,
    *,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader with configured batching and device memory settings."""

    num_workers = int(config["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def prepare_data(
    config: Dict[str, Any],
    *,
    feature_scaler: Any = None,
    use_npy_cache: bool = True,
    write_npy_cache: bool = True,
    basin_ids: Optional[set[str]] = None,
) -> PreparedData:
    """Prepare data for training and evaluation."""

    from data import (
        FeatureStandardizer,
        build_explicit_date_splits,
        build_model_input_arrays,
        build_needed_columns,
        build_static_feature_map,
        build_streamflow_datasets,
        compute_feature_standardizer,
        compute_target_standard_deviations,
        ensure_parquet_cache,
        is_npy_cache_compatible,
        load_npy_cache,
        load_static_attribute_table,
        load_basin_cluster_assignments,
        load_streamflow_timeseries,
        prepare_basin_arrays,
        save_npy_cache,
    )

    dynamic_features = list(config["dynamic_features"])
    static_path, static_id_column, static_columns, static_sheet = (
        resolve_static_attribute_config(config)
    )
    static_table = load_static_attribute_table(
        static_path,
        static_id_column,
        static_columns,
        sheet_name=static_sheet,
    )
    expected_feature_names = dynamic_features + [f"attr_{name}" for name in static_columns]
    splits = build_explicit_date_splits(
        config["train_start_date"],
        config["train_end_date"],
        config["val_start_date"],
        config["val_end_date"],
        config["test_start_date"],
        config["test_end_date"],
    )

    cache_dir_value = config.get("npy_cache_dir")
    cache_dir = Path(cache_dir_value) if cache_dir_value else None
    cache_loaded = False
    cache_metadata: Dict[str, Any] = {}
    model_inputs: Dict[str, Any]
    target_stds: Dict[str, float]
    scaler = feature_scaler

    if feature_scaler is None and use_npy_cache and cache_dir is not None and cache_dir.exists():
        cached_inputs, cache_metadata = load_npy_cache(cache_dir)
        if is_npy_cache_compatible(
            cache_metadata,
            expected_feature_names=expected_feature_names,
            target_column=str(config["target_column"]),
            dynamic_feature_names=dynamic_features,
            log_feature_names=list(config.get("log_features") or []),
            history_length=int(config["history_length"]),
            forecast_length=int(config["forecast_length"]),
        ):
            raw_target_stds = cache_metadata.get("target_scalers")
            if not isinstance(raw_target_stds, dict):
                raise ValueError("Compatible NPY cache is missing target_scalers.")
            target_stds = {str(key): float(value) for key, value in raw_target_stds.items()}
            model_inputs = cached_inputs
            scaler_mean = cache_metadata.get("scaler_mean")
            scaler_std = cache_metadata.get("scaler_std")
            if isinstance(scaler_mean, list) and isinstance(scaler_std, list):
                scaler = FeatureStandardizer(
                    np.asarray(scaler_mean, dtype=np.float32),
                    np.asarray(scaler_std, dtype=np.float32),
                )
            cache_loaded = True
        else:
            print("NPY cache metadata mismatch; rebuilding explicitly from source data.")

    if cache_loaded:
        working_arrays = model_inputs
        feature_names = list(cache_metadata["feature_names"])
    else:
        needed_columns = build_needed_columns(dynamic_features, str(config["target_column"]))
        data_dir = Path(config["data_dir"])
        ensure_parquet_cache(
            data_dir,
            needed_columns=needed_columns,
        )
        basin_frames = load_streamflow_timeseries(
            data_dir,
            needed_columns=needed_columns,
        )
        if basin_ids is not None:
            basin_frames = {
                basin_id: frame
                for basin_id, frame in basin_frames.items()
                if basin_id in basin_ids
            }
        basin_arrays, feature_names = prepare_basin_arrays(
            basin_frames,
            dynamic_features,
            str(config["target_column"]),
            not bool(config["no_interpolate"]),
            list(config.get("log_features") or []),
            static_table,
            interpolate_target=bool(config["interpolate_target"]),
        )
        working_arrays = basin_arrays

    static_features, static_feature_names = build_static_feature_map(
        working_arrays,
        static_table,
        static_columns,
    )
    working_arrays = {
        basin_id: working_arrays[basin_id]
        for basin_id in working_arrays
        if basin_id in static_features
    }
    static_features = {basin_id: static_features[basin_id] for basin_id in working_arrays}
    if not working_arrays:
        raise RuntimeError("No basins remain after static-attribute alignment.")

    if cache_loaded:
        model_inputs = {basin_id: model_inputs[basin_id] for basin_id in working_arrays}
    else:
        if scaler is None:
            scaler = compute_feature_standardizer(working_arrays, splits[0])
        target_stds = compute_target_standard_deviations(
            working_arrays,
            splits[0],
            bool(config["target_log1p"]),
        )
        model_inputs = build_model_input_arrays(working_arrays, scaler)
        if write_npy_cache and cache_dir is not None:
            save_npy_cache(
                model_inputs,
                target_stds,
                feature_names,
                cache_dir,
                scaler=scaler,
                target_column=str(config["target_column"]),
                dynamic_feature_names=dynamic_features,
                log_feature_names=list(config.get("log_features") or []),
                history_length=int(config["history_length"]),
                forecast_length=int(config["forecast_length"]),
            )

    routing_labels = None
    if resolve_model_type(config) == "fixed_routing_moe":
        _basin_clusters, all_routing_labels = load_basin_cluster_assignments(
            Path(config["cluster_assignments_file"]),
            str(config.get("cluster_id_column", "gauge_id")),
            str(config.get("cluster_label_column", "cluster_category")),
        )
        routing_labels = {
            basin_id: all_routing_labels[basin_id]
            for basin_id in model_inputs
            if basin_id in all_routing_labels
        }
        missing_labels = sorted(set(model_inputs) - set(routing_labels))
        if missing_labels:
            raise ValueError(
                f"Missing fixed-routing labels for {len(missing_labels)} basins."
            )

    datasets = build_streamflow_datasets(
        model_inputs,
        target_stds,
        splits,
        static_features,
        history_length=int(config["history_length"]),
        forecast_length=int(config["forecast_length"]),
        allow_missing_targets=not bool(config["strict_target_window"]),
        routing_labels=routing_labels,
    )
    return PreparedData(
        feature_names=feature_names,
        static_feature_names=static_feature_names,
        feature_scaler=scaler,
        target_standard_deviations=target_stds,
        model_input_arrays=model_inputs,
        static_features=static_features,
        datasets=datasets,
    )


class _BalanceRecorder:
    def __init__(self, model: nn.Module) -> None:
        self.weighted_sum = 0.0
        self.count = 0
        self.handle = model.register_forward_hook(self._record)

    def _record(self, _module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) < 2:
            return
        predictions, balance = output[0], output[1]
        batch_size = int(predictions.size(0))
        self.weighted_sum += float(balance.detach().float().item()) * batch_size
        self.count += batch_size

    def close(self) -> float:
        self.handle.remove()
        return self.weighted_sum / max(1, self.count)


def _write_history(path: Path, history: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "epoch",
        "temperature",
        "learning_rate",
        "train_prediction_loss",
        "train_load_balance_loss",
        "train_total_loss",
        "validation_prediction_loss",
        "validation_load_balance_loss",
        "validation_total_loss",
        "best_validation_loss",
        "early_stopping_counter",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_epoch_loop(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_scaler: Any,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> list[Dict[str, Any]]:
    """Run training epochs and return training history."""

    model_type = resolve_model_type(config)
    best_validation_loss = float("inf")
    early_counter = 0
    history: list[Dict[str, Any]] = []
    best_path = output_dir / "model_best.pt"

    for epoch in range(1, int(config["epochs"]) + 1):
        temperature: Optional[float] = None
        train_balance = 0.0
        validation_balance = 0.0
        if model_type == "moe":
            temperature = compute_moe_temperature(
                epoch,
                float(config["initial_temperature"]),
                float(config["final_temperature"]),
                int(config["temperature_warmup_epochs"]),
                int(config["temperature_decay_end_epoch"]),
            )
            model.temperature = temperature
            recorder = _BalanceRecorder(model)
            train_total = train_moe_epoch(
                model,
                train_loader,
                optimizer,
                device,
                gradient_clip=float(config["gradient_clip"]),
                amp=bool(config["amp"]),
                scaler=grad_scaler,
                target_log1p=bool(config["target_log1p"]),
                load_balance_weight=float(config["load_balance_weight"]),
            )
            train_balance = recorder.close()
            recorder = _BalanceRecorder(model)
            validation_total = evaluate_moe_loss(
                model,
                validation_loader,
                device,
                amp=bool(config["amp"]),
                target_log1p=bool(config["target_log1p"]),
                load_balance_weight=float(config["load_balance_weight"]),
            )
            validation_balance = recorder.close()
            balance_weight = float(config["load_balance_weight"])
            train_prediction = train_total - balance_weight * train_balance
            validation_prediction = validation_total - balance_weight * validation_balance
        elif model_type == "fixed_routing_moe":
            train_total = train_fixed_routing_moe_epoch(
                model,
                train_loader,
                optimizer,
                device,
                gradient_clip=float(config["gradient_clip"]),
                amp=bool(config["amp"]),
                scaler=grad_scaler,
                target_log1p=bool(config["target_log1p"]),
            )
            validation_total = evaluate_fixed_routing_moe_loss(
                model,
                validation_loader,
                device,
                amp=bool(config["amp"]),
                target_log1p=bool(config["target_log1p"]),
            )
            train_prediction = train_total
            validation_prediction = validation_total
        else:
            train_total = train_global_lstm_epoch(
                model,
                train_loader,
                optimizer,
                device,
                gradient_clip=float(config["gradient_clip"]),
                amp=bool(config["amp"]),
                scaler=grad_scaler,
                target_log1p=bool(config["target_log1p"]),
            )
            validation_total = evaluate_global_lstm_loss(
                model,
                validation_loader,
                device,
                amp=bool(config["amp"]),
                target_log1p=bool(config["target_log1p"]),
            )
            train_prediction = train_total
            validation_prediction = validation_total

        if scheduler is not None:
            scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        update = update_early_stopping(
            validation_total,
            best_validation_loss,
            early_counter,
            int(config["early_stop_patience"]),
        )
        best_validation_loss = update.best_validation_loss
        early_counter = update.epochs_without_improvement
        if update.improved:
            save_model_checkpoint(model, epoch, best_path)

        row = {
            "epoch": epoch,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "train_prediction_loss": train_prediction,
            "train_load_balance_loss": train_balance if model_type == "moe" else None,
            "train_total_loss": train_total,
            "validation_prediction_loss": validation_prediction,
            "validation_load_balance_loss": (
                validation_balance if model_type == "moe" else None
            ),
            "validation_total_loss": validation_total,
            "best_validation_loss": best_validation_loss,
            "early_stopping_counter": early_counter,
        }
        history.append(row)
        _write_history(output_dir / "training_history.csv", history)
        temperature_text = "" if temperature is None else f" | temp={temperature:.4f}"
        print(
            f"Epoch {epoch:03d} | train={train_total:.6f} | "
            f"validation={validation_total:.6f} | lr={learning_rate:.2e}"
            f"{temperature_text} | early={early_counter}"
        )

        if update.should_stop:
            break
    if best_path.exists():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    return history


def run_training(
    config: Dict[str, Any],
    *,
    basin_ids: Optional[set[str]] = None,
) -> list[Dict[str, Any]]:
    """Prepare data and run the configured model training."""

    model_type = resolve_model_type(config)
    if model_type == "cluster_specific_lstms":
        from data import load_basin_cluster_assignments

        basin_to_cluster, _routing_labels = load_basin_cluster_assignments(
            Path(config["cluster_assignments_file"]),
            str(config.get("cluster_id_column", "gauge_id")),
            str(config.get("cluster_label_column", "cluster_category")),
        )
        requested = config.get("clusters")
        clusters = list(requested) if requested else sorted(set(basin_to_cluster.values()))
        combined_history: list[Dict[str, Any]] = []
        for cluster in clusters:
            cluster_basins = {
                basin_id
                for basin_id, assigned_cluster in basin_to_cluster.items()
                if assigned_cluster == cluster
            }
            if not cluster_basins:
                raise ValueError(f"Requested cluster '{cluster}' has no basin assignments.")
            cluster_config = dict(config)
            cluster_config["model"] = "global_lstm"
            cluster_config["output_dir"] = str(Path(config["output_dir"]) / cluster)
            history = run_training(cluster_config, basin_ids=cluster_basins)
            combined_history.extend(
                [{"cluster": cluster, **row} for row in history]
            )
        return combined_history

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    set_random_seed(int(config["seed"]), bool(config["deterministic"]))
    prepared = prepare_data(
        config,
        use_npy_cache=basin_ids is None,
        write_npy_cache=basin_ids is None,
        basin_ids=basin_ids,
    )
    train_loader = make_data_loader(prepared.datasets[0], config, device, shuffle=True)
    validation_loader = make_data_loader(prepared.datasets[1], config, device, shuffle=False)
    model = build_model(
        config,
        len(prepared.feature_names),
        len(prepared.static_feature_names),
        device,
    )
    optimizer = create_adam_optimizer(model, float(config["learning_rate"]))
    scheduler = create_multistep_scheduler(
        optimizer,
        list(config["lr_milestones"]),
        float(config["lr_gamma"]),
    )
    grad_scaler = create_grad_scaler(device, bool(config["amp"]))
    history = run_epoch_loop(
        model,
        train_loader,
        validation_loader,
        optimizer,
        scheduler,
        grad_scaler,
        config,
        device,
        output_dir,
    )

    if prepared.feature_scaler is not None:
        from data import save_feature_standardizer

        save_feature_standardizer(
            output_dir,
            prepared.feature_scaler,
            prepared.feature_names,
            list(config.get("log_features") or []),
        )
    run_metadata = dict(config)
    run_metadata["resolved_model"] = resolve_model_type(config)
    (output_dir / "config.json").write_text(
        json.dumps(run_metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--static-attributes", type=Path)
    parser.add_argument("--npy-cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    overrides = {
        "device": args.device,
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "npy_cache_dir": args.npy_cache_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = str(value)
    if args.static_attributes is not None:
        config["attributes_file"] = str(args.static_attributes)
    run_training(config)


if __name__ == "__main__":
    main()
