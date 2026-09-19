"""Flow-interval bias analysis for Global LSTM and MoE."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np


INTERVALS = (
    ("highest_1pct", "high", 0.00, 0.01),
    ("high_1_5pct", "high", 0.01, 0.05),
    ("high_5_10pct", "high", 0.05, 0.10),
    ("lowest_5pct", "low", 0.00, 0.05),
    ("low_5_10pct", "low", 0.05, 0.10),
    ("low_10_20pct", "low", 0.10, 0.20),
)


def _read_daily_predictions(path: Path) -> Dict[tuple[str, str], tuple[float, float]]:
    """Read basin/date keyed observation and prediction pairs."""

    records: Dict[tuple[str, str], tuple[float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"basin_id", "date", "prediction", "observation"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            key = (str(row["basin_id"]), str(row["date"]))
            if key in records:
                raise ValueError(
                    f"Duplicate basin/date row in {path}: basin={key[0]}, date={key[1]}"
                )
            observation = float(row["observation"])
            prediction = float(row["prediction"])
            if np.isfinite(observation) and np.isfinite(prediction):
                records[key] = (observation, prediction)

    return records


def _align_predictions(
    global_records: Dict[tuple[str, str], tuple[float, float]],
    moe_records: Dict[tuple[str, str], tuple[float, float]],
) -> Dict[str, list[tuple[str, float, float, float]]]:
    """Align Global LSTM and MoE predictions on identical basin/date keys."""

    global_keys = set(global_records)
    moe_keys = set(moe_records)
    if global_keys != moe_keys:
        only_global = len(global_keys - moe_keys)
        only_moe = len(moe_keys - global_keys)
        raise ValueError(
            "Global LSTM and MoE daily-prediction files do not contain identical "
            f"valid basin/date rows (Global-only={only_global}, MoE-only={only_moe})."
        )

    grouped: Dict[str, list[tuple[str, float, float, float]]] = {}
    for basin_id, date in sorted(global_keys):
        global_observation, global_prediction = global_records[(basin_id, date)]
        moe_observation, moe_prediction = moe_records[(basin_id, date)]

        if not np.isclose(
            global_observation,
            moe_observation,
            rtol=1e-7,
            atol=1e-10,
            equal_nan=False,
        ):
            raise ValueError(
                "Observed streamflow differs between model outputs for "
                f"basin={basin_id}, date={date}."
            )

        grouped.setdefault(basin_id, []).append(
            (date, global_observation, global_prediction, moe_prediction)
        )

    return grouped


def _rank_interval_indices(
    observed: np.ndarray,
    tail: str,
    start_fraction: float,
    end_fraction: float,
) -> np.ndarray:
    """Return fixed-count rank indices for one observed-flow interval.

    Cumulative interval boundaries use ceil(n * fraction), matching the
    top/bottom-fraction convention used by the main evaluation metrics.
    """

    n = observed.size
    if n == 0:
        return np.empty(0, dtype=int)

    start = int(np.ceil(n * start_fraction))
    end = int(np.ceil(n * end_fraction))
    start = min(max(start, 0), n)
    end = min(max(end, 0), n)
    if end <= start:
        return np.empty(0, dtype=int)

    if tail == "high":
        order = np.argsort(-observed, kind="mergesort")
    elif tail == "low":
        order = np.argsort(observed, kind="mergesort")
    else:
        raise ValueError("tail must be 'high' or 'low'")

    return order[start:end]


def _percentage_bias(
    observed: np.ndarray,
    simulated: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Return streamflow-volume PBIAS (%) on the selected dates."""

    if indices.size == 0:
        return float("nan")

    observed_sum = float(np.sum(observed[indices]))
    if not np.isfinite(observed_sum) or observed_sum == 0.0:
        return float("nan")

    simulated_sum = float(np.sum(simulated[indices]))
    if not np.isfinite(simulated_sum):
        return float("nan")

    return 100.0 * (simulated_sum - observed_sum) / observed_sum


def compute_per_basin_interval_bias(
    grouped_records: Dict[str, list[tuple[str, float, float, float]]],
) -> list[dict[str, object]]:
    """Compute the six rank-defined flow-interval biases for both models."""

    rows: list[dict[str, object]] = []

    for basin_id in sorted(grouped_records):
        records = sorted(grouped_records[basin_id], key=lambda item: item[0])
        observed = np.asarray([item[1] for item in records], dtype=np.float64)
        global_simulated = np.asarray([item[2] for item in records], dtype=np.float64)
        moe_simulated = np.asarray([item[3] for item in records], dtype=np.float64)

        for label, tail, start_fraction, end_fraction in INTERVALS:
            indices = _rank_interval_indices(
                observed,
                tail,
                start_fraction,
                end_fraction,
            )
            global_pbias = _percentage_bias(observed, global_simulated, indices)
            moe_pbias = _percentage_bias(observed, moe_simulated, indices)

            rows.append(
                {
                    "basin_id": basin_id,
                    "interval": label,
                    "n_days_total": int(observed.size),
                    "n_days_interval": int(indices.size),
                    "global_lstm_pbias": global_pbias,
                    "moe_pbias": moe_pbias,
                    "global_lstm_abs_pbias": (
                        abs(global_pbias) if np.isfinite(global_pbias) else float("nan")
                    ),
                    "moe_abs_pbias": (
                        abs(moe_pbias) if np.isfinite(moe_pbias) else float("nan")
                    ),
                    "abs_bias_reduction": (
                        abs(global_pbias) - abs(moe_pbias)
                        if np.isfinite(global_pbias) and np.isfinite(moe_pbias)
                        else float("nan")
                    ),
                }
            )

    return rows


def summarize_interval_bias(
    per_basin_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Summarize paired absolute-bias performance across catchments."""

    summary_rows: list[dict[str, object]] = []

    for label, _tail, _start_fraction, _end_fraction in INTERVALS:
        subset = [row for row in per_basin_rows if row["interval"] == label]

        global_abs = np.asarray(
            [float(row["global_lstm_abs_pbias"]) for row in subset],
            dtype=np.float64,
        )
        moe_abs = np.asarray(
            [float(row["moe_abs_pbias"]) for row in subset],
            dtype=np.float64,
        )
        valid = np.isfinite(global_abs) & np.isfinite(moe_abs)
        global_abs = global_abs[valid]
        moe_abs = moe_abs[valid]

        if global_abs.size == 0:
            summary_rows.append(
                {
                    "interval": label,
                    "n_catchments": 0,
                    "global_lstm_median_abs_pbias": float("nan"),
                    "moe_median_abs_pbias": float("nan"),
                    "median_abs_bias_reduction": float("nan"),
                    "catchments_improved_pct": float("nan"),
                }
            )
            continue

        summary_rows.append(
            {
                "interval": label,
                "n_catchments": int(global_abs.size),
                "global_lstm_median_abs_pbias": float(np.median(global_abs)),
                "moe_median_abs_pbias": float(np.median(moe_abs)),
                "median_abs_bias_reduction": float(
                    np.median(global_abs) - np.median(moe_abs)
                ),
                "catchments_improved_pct": float(
                    100.0 * np.mean(moe_abs < global_abs)
                ),
            }
        )

    return summary_rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    global_predictions_path: Path,
    moe_predictions_path: Path,
    output_dir: Path,
) -> None:
    """Run the paired Global LSTM–MoE flow-interval analysis."""

    global_records = _read_daily_predictions(global_predictions_path)
    moe_records = _read_daily_predictions(moe_predictions_path)
    grouped_records = _align_predictions(global_records, moe_records)

    per_basin_rows = compute_per_basin_interval_bias(grouped_records)
    summary_rows = summarize_interval_bias(per_basin_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "per_basin_flow_interval_bias.csv",
        per_basin_rows,
    )
    _write_csv(
        output_dir / "flow_interval_summary.csv",
        summary_rows,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-predictions",
        type=Path,
        required=True,
        help="Global LSTM daily_predictions.csv produced by evaluate.py.",
    )
    parser.add_argument(
        "--moe-predictions",
        type=Path,
        required=True,
        help="MoE daily_predictions.csv produced by evaluate.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for interval-analysis outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_analysis(
        args.global_predictions,
        args.moe_predictions,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
