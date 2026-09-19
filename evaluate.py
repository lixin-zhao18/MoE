"""Evaluation for the models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from metrics import (
    compute_kge,
    compute_nse,
    pbias_high_flow,
    pbias_low_flow,
)
from train import (
    build_model,
    load_config,
    make_data_loader,
    prepare_data,
    resolve_model_type,
    resolve_static_attribute_config,
    set_random_seed,
)
from training import autocast_context, compute_moe_temperature


def resolve_checkpoint_temperature(
    config: Dict[str, Any],
    checkpoint: Dict[str, Any],
) -> Optional[float]:
    """Recover the MoE routing temperature from the saved checkpoint epoch."""

    if resolve_model_type(config) != "moe":
        return None

    epoch = checkpoint.get("epoch")
    if epoch is None:
        raise ValueError(
            "MoE checkpoints must contain 'epoch' so the routing temperature "
            "can be recovered from the configured temperature schedule."
        )

    epoch = int(epoch)
    if epoch < 1:
        raise ValueError("Checkpoint epoch must be a positive integer.")

    return compute_moe_temperature(
        epoch,
        float(config["initial_temperature"]),
        float(config["final_temperature"]),
        int(config["temperature_warmup_epochs"]),
        int(config["temperature_decay_end_epoch"]),
    )


def load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Load a model checkpoint with strict key matching."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise TypeError("Checkpoint must contain a state_dict mapping.")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint loading returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return checkpoint


def collect_predictions(
    model: nn.Module,
    dataset: Any,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Collect valid raw-scale predictions and one routing row per basin."""

    model_type = resolve_model_type(config)
    loader = make_data_loader(dataset, config, device, shuffle=False)
    autocast_context_manager = autocast_context(device, bool(config["amp"]))
    prediction_records: list[Dict[str, Any]] = []
    routing_by_basin: Dict[str, Dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            non_blocking = device.type == "cuda"
            encoder = batch["encoder_inputs"].to(device, non_blocking=non_blocking)
            decoder = batch["decoder_inputs"].to(device, non_blocking=non_blocking)
            targets = batch["targets"].cpu().numpy()
            target_mask = batch.get("target_mask")
            if target_mask is not None:
                target_mask_array = target_mask.cpu().numpy().astype(bool)
            else:
                target_mask_array = np.isfinite(targets)
            basin_ids: Sequence[str] = batch["basin_id"]
            start_indices = batch["start_idx"].cpu().numpy()

            with autocast_context_manager():
                if model_type == "moe":
                    static_features = batch["static_features"].to(
                        device,
                        non_blocking=non_blocking,
                    )
                    predictions_raw, _balance_loss, routing = model(
                        encoder,
                        decoder,
                        static_features,
                    )
                elif model_type == "fixed_routing_moe":
                    static_features = batch["static_features"].to(
                        device,
                        non_blocking=non_blocking,
                    )
                    routing_labels = batch["routing_label"].to(
                        device,
                        non_blocking=non_blocking,
                    )
                    predictions_raw, _balance_loss, routing = model(
                        encoder,
                        decoder,
                        static_features,
                        routing_labels,
                    )
                else:
                    predictions_raw = model(encoder, decoder)
                    routing = None

            log_predictions = predictions_raw.detach().cpu().numpy()
            predictions = (
                np.expm1(log_predictions)
                if bool(config["target_log1p"])
                else log_predictions
            )
            routing_probabilities = None
            topk_indices = None
            topk_weights = None
            if routing is not None:
                routing_probabilities = (
                    routing["routing_probabilities"].detach().cpu().numpy()
                )
                topk_indices = routing["topk_indices"].detach().cpu().numpy()
                topk_weights = routing["topk_weights"].detach().cpu().numpy()

            for batch_index, basin_id in enumerate(basin_ids):
                _features, _targets, dates = dataset.basin_arrays[basin_id]
                current_log = np.atleast_1d(log_predictions[batch_index])
                current_predictions = np.atleast_1d(predictions[batch_index])
                current_targets = np.atleast_1d(targets[batch_index])
                current_mask = np.atleast_1d(
                    target_mask_array[batch_index]
                    if target_mask_array.ndim > 1
                    else np.asarray([target_mask_array[batch_index]])
                )
                sequence_length = min(
                    current_predictions.shape[0],
                    current_targets.shape[0],
                    current_mask.shape[0],
                )
                for step_index in range(sequence_length):
                    if not current_mask[step_index]:
                        continue
                    observation = current_targets[step_index]
                    if not np.isfinite(observation):
                        continue
                    date = dates[int(start_indices[batch_index]) + step_index]
                    if not (
                        np.isfinite(current_log[step_index])
                        and np.isfinite(current_predictions[step_index])
                    ):
                        raise ValueError(
                            f"Model produced a non-finite prediction for basin ID "
                            f"{basin_id} on {date:%Y-%m-%d}."
                        )
                    prediction_records.append(
                        {
                            "basin_id": basin_id,
                            "date": date.strftime("%Y-%m-%d"),
                            "prediction_log_space": float(current_log[step_index]),
                            "prediction": float(current_predictions[step_index]),
                            "observation": float(observation),
                        }
                    )

                if routing is not None and basin_id not in routing_by_basin:
                    row: Dict[str, Any] = {
                        "basin_id": basin_id,
                        "top1_expert": int(topk_indices[batch_index, 0]) + 1,
                        "top1_weight": float(topk_weights[batch_index, 0]),
                        "top2_expert": (
                            int(topk_indices[batch_index, 1]) + 1
                            if topk_indices.shape[1] > 1
                            else None
                        ),
                        "top2_weight": (
                            float(topk_weights[batch_index, 1])
                            if topk_weights.shape[1] > 1
                            else None
                        ),
                    }
                    for expert_index, probability in enumerate(
                        routing_probabilities[batch_index],
                        start=1,
                    ):
                        row[f"routing_probability_expert_{expert_index}"] = float(probability)
                    routing_by_basin[basin_id] = row
    return prediction_records, [routing_by_basin[key] for key in sorted(routing_by_basin)]


def compute_per_basin_metrics(
    prediction_records: Sequence[Dict[str, Any]],
    high_flow_fraction: float,
    low_flow_fraction: float,
) -> list[Dict[str, Any]]:
    """Compute the four common evaluation metrics in raw runoff space."""

    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for record in prediction_records:
        grouped.setdefault(str(record["basin_id"]), []).append(record)
    rows: list[Dict[str, Any]] = []
    for basin_id in sorted(grouped):
        records = grouped[basin_id]
        observed = np.asarray([record["observation"] for record in records], dtype=np.float64)
        simulated = np.asarray([record["prediction"] for record in records], dtype=np.float64)
        rows.append(
            {
                "basin_id": basin_id,
                "nse": compute_nse(observed, simulated),
                "kge": compute_kge(observed, simulated),
                "pbias_high_flow": pbias_high_flow(
                    observed,
                    simulated,
                    high_flow_fraction,
                ),
                "pbias_low_flow": pbias_low_flow(
                    observed,
                    simulated,
                    low_flow_fraction,
                ),
                "count": len(records),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(
    config: Dict[str, Any],
    checkpoint_path: Path,
    scaler_path: Optional[Path],
    output_dir: Path,
    split: str,
    *,
    save_daily_predictions: bool = False,
    basin_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model and write compact outputs."""

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
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
        combined = {
            "metadata": {"model": model_type, "split": split},
            "predictions": [],
            "metrics": [],
            "routing": [],
        }
        for cluster in clusters:
            cluster_basins = {
                basin_id
                for basin_id, assigned_cluster in basin_to_cluster.items()
                if assigned_cluster == cluster
            }
            cluster_config = dict(config)
            cluster_config["model"] = "global_lstm"
            cluster_root = checkpoint_path / cluster
            result = run_evaluation(
                cluster_config,
                cluster_root / "model_best.pt",
                cluster_root / "feature_scaler.json",
                output_dir / cluster,
                split,
                save_daily_predictions=save_daily_predictions,
                basin_ids=cluster_basins,
            )
            for row in result["metrics"]:
                row["cluster"] = cluster
            combined["predictions"].extend(result["predictions"])
            combined["metrics"].extend(result["metrics"])
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "per_basin_metrics.csv", combined["metrics"])
        return combined

    device = torch.device(config["device"])
    set_random_seed(int(config["seed"]), bool(config["deterministic"]))
    from data import load_feature_standardizer

    _static_path, _static_id, static_columns, _static_sheet = (
        resolve_static_attribute_config(config)
    )
    expected_names = list(config["dynamic_features"]) + [
        f"attr_{name}" for name in static_columns
    ]
    if scaler_path is None:
        raise ValueError("A feature scaler is required for this model.")

    feature_scaler, feature_names, _log_features = load_feature_standardizer(
        scaler_path,
        expected_feature_names=expected_names,
    )
    prepared = prepare_data(
        config,
        feature_scaler=feature_scaler,
        use_npy_cache=False,
        write_npy_cache=False,
        basin_ids=basin_ids,
    )
    if prepared.feature_names != feature_names:
        raise ValueError("Prepared feature order differs from the saved scaler.")
    model = build_model(
        config,
        len(prepared.feature_names),
        len(prepared.static_feature_names),
        device,
    )
    checkpoint = load_checkpoint_strict(model, checkpoint_path, device)
    temperature = resolve_checkpoint_temperature(config, checkpoint)
    if temperature is not None:
        model.temperature = temperature
    dataset_index = {"train": 0, "validation": 1, "test": 2}[split]
    prediction_records, routing_rows = collect_predictions(
        model,
        prepared.datasets[dataset_index],
        config,
        device,
    )
    metric_rows = compute_per_basin_metrics(
        prediction_records,
        float(config["high_flow_fraction"]),
        float(config["low_flow_fraction"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_basin_metrics.csv", metric_rows)
    if routing_rows:
        _write_csv(output_dir / "routing_summary.csv", routing_rows)
    if save_daily_predictions:
        _write_csv(output_dir / "daily_predictions.csv", prediction_records)
    metadata = {
        "model": model_type,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "routing_temperature": temperature,
        "basin_count": len(metric_rows),
        "prediction_count": len(prediction_records),
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return {
        "metadata": metadata,
        "predictions": prediction_records,
        "metrics": metric_rows,
        "routing": routing_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--static-attributes", type=Path)
    parser.add_argument("--save-daily-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    for key, value in {
        "device": args.device,
        "data_dir": args.data_dir,
    }.items():
        if value is not None:
            config[key] = str(value)
    if args.static_attributes is not None:
        config["attributes_file"] = str(args.static_attributes)
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    model_type = resolve_model_type(config)
    output_dir = args.output_dir or (run_dir / "evaluation")

    if model_type == "cluster_specific_lstms":
        checkpoint = args.checkpoint or run_dir
        scaler = None
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Cluster-specific run directory not found: {checkpoint}"
            )
    else:
        checkpoint = args.checkpoint or (run_dir / "model_best.pt")
        scaler = args.scaler or (run_dir / "feature_scaler.json")

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if not scaler.is_file():
            raise FileNotFoundError(f"Feature scaler not found: {scaler}")
    run_evaluation(
        config,
        checkpoint,
        scaler if scaler is not None else checkpoint,
        output_dir,
        args.split,
        save_daily_predictions=args.save_daily_predictions,
    )


if __name__ == "__main__":
    main()
