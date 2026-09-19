"""Data preparation and preprocessing utilities for the streamflow models."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional progress display
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


SMALL_STD = 1e-6
DATE_COLUMN_CANDIDATES = ("day", "date", "datetime", "time", "timestamp")

ArrayEntry = Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], Optional[np.ndarray]]
ModelInputEntry = Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]
DateRange = Tuple[pd.Timestamp, pd.Timestamp]
DateRangeSpec = Union[DateRange, Dict[str, DateRange]]

__all__ = [
    "FeatureStandardizer",
    "StreamflowDataset",
    "MoEStreamflowDataset",
    "FixedRoutingStreamflowDataset",
    "build_needed_columns",
    "ensure_parquet_cache",
    "load_streamflow_timeseries",
    "load_static_attribute_table",
    "load_basin_cluster_assignments",
    "canonicalize_basin_id",
    "apply_log_transform",
    "preprocess_basin_frame",
    "prepare_basin_arrays",
    "compute_feature_standardizer",
    "compute_target_standard_deviations",
    "build_model_input_arrays",
    "build_static_feature_map",
    "build_explicit_date_splits",
    "build_streamflow_datasets",
    "save_feature_standardizer",
    "load_feature_standardizer",
    "save_npy_cache",
    "load_npy_cache",
    "is_npy_cache_compatible",
]


def _cast_float_columns_to_float32(frame: pd.DataFrame) -> pd.DataFrame:
    float_columns = frame.select_dtypes(include=["floating"]).columns
    if len(float_columns) > 0:
        frame.loc[:, float_columns] = frame.loc[:, float_columns].astype("float32", copy=False)
    return frame


def _normalize_column_name(name: object) -> str:
    return str(name).strip().lower()


def build_needed_columns(feature_columns: Sequence[str], target_column: str) -> Set[str]:
    """Return normalized dynamic, target, and possible date columns to read."""

    needed = {_normalize_column_name(target_column)}
    needed.update(_normalize_column_name(column) for column in feature_columns)
    needed.update(DATE_COLUMN_CANDIDATES)
    return needed


def _resolve_columns_by_normalized_name(
    columns: Sequence[object],
    normalized_needed: Set[str],
) -> List[str]:
    return [
        str(column)
        for column in columns
        if _normalize_column_name(column) in normalized_needed
    ]


def ensure_parquet_cache(
    data_dir: Path,
    needed_columns: Optional[Set[str]] = None,
    rebuild_existing: bool = False,
) -> None:
    """Create missing Parquet mirrors of basin XLSX files.

    Set ``rebuild_existing`` to refresh existing Parquet files.
    """

    usecols = None
    if needed_columns:
        normalized_needed = {_normalize_column_name(column) for column in needed_columns}
        usecols = lambda column: _normalize_column_name(column) in normalized_needed
    converted = 0
    skipped = 0
    for xlsx_path in sorted(data_dir.glob("*.xlsx")):
        parquet_path = xlsx_path.with_suffix(".parquet")
        if parquet_path.exists() and not rebuild_existing:
            skipped += 1
            continue
        frame = (
            pd.read_excel(xlsx_path, usecols=usecols)
            if usecols is not None
            else pd.read_excel(xlsx_path)
        )
        frame = _cast_float_columns_to_float32(frame)
        frame.to_parquet(parquet_path, index=False)
        del frame
        gc.collect()
        converted += 1
    print(
        "Parquet cache scan finished. "
        f"converted={converted}, skipped_existing={skipped}, rebuild={rebuild_existing}"
    )


def load_streamflow_timeseries(
    data_dir: Path,
    ignore_files: Optional[Sequence[Path]] = None,
    needed_columns: Optional[Set[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load one time-series table per basin from an explicit directory.

    Supported formats are Parquet, Feather, CSV, and XLSX. If several formats
    share a stem, the priority is Parquet, Feather, CSV, then XLSX. Tables are
    sorted by date and expose the date column as ``day``.
    """

    supported_suffixes = (".parquet", ".feather", ".csv", ".xlsx")
    files = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in supported_suffixes
    ]
    ignored = {Path(path).resolve() for path in ignore_files} if ignore_files else set()
    if not files:
        raise FileNotFoundError(
            "No supported timeseries files (*.xlsx, *.csv, *.parquet, *.feather) "
            f"found under {data_dir}"
        )

    priority = {".parquet": 0, ".feather": 1, ".csv": 2, ".xlsx": 3}
    selected: Dict[str, Path] = {}
    for path in files:
        if path.resolve() in ignored:
            continue
        existing = selected.get(path.stem)
        if existing is None or priority[path.suffix.lower()] < priority[existing.suffix.lower()]:
            selected[path.stem] = path

    ordered_files = [selected[key] for key in sorted(selected)]
    basin_frames: Dict[str, pd.DataFrame] = {}
    usecols = None
    normalized_needed: Optional[Set[str]] = None
    if needed_columns:
        normalized_needed = {_normalize_column_name(column) for column in needed_columns}
        usecols = lambda column: _normalize_column_name(column) in normalized_needed

    for path in ordered_files:
        suffix = path.suffix.lower()
        if suffix == ".xlsx":
            frame = (
                pd.read_excel(path, usecols=usecols)
                if usecols is not None
                else pd.read_excel(path)
            )
        elif suffix == ".csv":
            frame = pd.read_csv(path, usecols=usecols) if usecols is not None else pd.read_csv(path)
        elif suffix == ".parquet":
            if normalized_needed:
                columns = None
                try:
                    import pyarrow.parquet as parquet

                    schema_columns = parquet.ParquetFile(path).schema.names
                    columns = _resolve_columns_by_normalized_name(
                        schema_columns,
                        normalized_needed,
                    )
                except Exception:
                    columns = None
                frame = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
            else:
                frame = pd.read_parquet(path)
        elif suffix == ".feather":
            frame = pd.read_feather(path)
        else:  # pragma: no cover - suffixes are filtered above
            continue

        if suffix in {".xlsx", ".parquet"}:
            frame = _cast_float_columns_to_float32(frame)
        frame = frame.rename(columns=lambda value: value.strip() if isinstance(value, str) else value)
        column_lookup = {str(column).lower(): column for column in frame.columns}
        day_column = next(
            (column_lookup[candidate] for candidate in DATE_COLUMN_CANDIDATES if candidate in column_lookup),
            None,
        )
        if day_column is None:
            raise ValueError(
                f"{path} missing a recognisable date column. Expected one of: "
                "day, date, datetime, time, timestamp."
            )
        if day_column != "day":
            frame = frame.rename(columns={day_column: "day"})
        frame["day"] = pd.to_datetime(frame["day"])
        frame = frame.sort_values("day").reset_index(drop=True)
        basin_frames[path.stem] = frame
    return basin_frames


def canonicalize_basin_id(value: Union[str, int, float, None]) -> str:
    """Normalize numeric gauge IDs while preserving non-numeric identifiers."""

    if value is None:
        return ""
    token = str(value).strip()
    if not token:
        return ""
    numeric_candidate = token.replace(".", "", 1)
    if numeric_candidate.isdigit():
        try:
            numeric_value = int(float(token))
        except (TypeError, ValueError):
            return token
        return f"{numeric_value:08d}"
    return token


def _read_table(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return _cast_float_columns_to_float32(pd.read_parquet(path))
    if suffix == ".feather":
        return pd.read_feather(path)
    if suffix in {".xlsx", ".xls"}:
        data = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else None)
        if isinstance(data, dict):
            if sheet_name is not None:
                if sheet_name not in data:
                    raise ValueError(f"Static-attribute workbook missing sheet '{sheet_name}'")
                return _cast_float_columns_to_float32(data[sheet_name])
            return _cast_float_columns_to_float32(data[next(iter(data))])
        return _cast_float_columns_to_float32(data)
    raise ValueError(f"Unsupported static-attribute file type: {suffix}")


def load_static_attribute_table(
    path: Path,
    id_column: str,
    attribute_columns: Sequence[str],
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load the ordered 96-column static basin-attribute table.

    The table contains 64 AEF columns followed by 32 HydroATLAS/basin
    attributes. Attribute values are converted to numeric.
    """

    frame = _read_table(path, sheet_name=sheet_name)
    if id_column not in frame.columns:
        raise ValueError(f"Static-attribute file missing id column '{id_column}'")
    missing = [column for column in attribute_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Static-attribute file missing columns: {', '.join(missing)}")
    subset = frame[[id_column] + list(attribute_columns)].copy()
    subset[id_column] = subset[id_column].map(canonicalize_basin_id)
    subset = subset[subset[id_column].astype(bool)]
    subset = subset.set_index(id_column)
    return subset.apply(pd.to_numeric, errors="coerce")


def apply_log_transform(values: np.ndarray) -> np.ndarray:
    """Clip values below zero and apply ``log1p``."""

    clipped = np.clip(values, a_min=0.0, a_max=None)
    return np.log1p(clipped)


def preprocess_basin_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    interpolate: bool,
    log_feature_columns: Sequence[str],
    interpolate_target: bool = True,
) -> pd.DataFrame:
    """Prepare one basin's dynamic features and runoff target.

    Sentinel values are replaced by NaN. With interpolation enabled, dynamic
    inputs use time interpolation in both directions followed by forward and
    backward filling. Target interpolation is controlled separately. Selected
    meteorological features (precipitation in the configs) are clipped at
    zero and transformed with ``log1p`` after interpolation.
    """

    working = frame.replace([-999, -9999, -9999.0], np.nan)
    required = ["day", target_column] + list(feature_columns)
    missing = [column for column in required if column not in working.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}")
    working = working[required]
    if interpolate:
        temporary = working.set_index("day").sort_index()
        temporary.loc[:, feature_columns] = (
            temporary.loc[:, feature_columns]
            .interpolate(method="time", limit_direction="both")
            .ffill()
            .bfill()
        )
        if interpolate_target:
            temporary.loc[:, target_column] = (
                temporary.loc[:, target_column]
                .interpolate(method="time", limit_direction="both")
                .ffill()
                .bfill()
            )
        working = temporary.reset_index()
    else:
        working = working.dropna(subset=feature_columns).reset_index(drop=True)
    working = working.sort_values("day").reset_index(drop=True)
    for name in log_feature_columns:
        if name in working.columns:
            transformed = apply_log_transform(working[name].to_numpy(dtype=np.float64))
            working[name] = transformed.astype(np.float32)
    return working


@dataclass(frozen=True)
class FeatureStandardizer:
    """Column-wise mean and population standard deviation for model inputs."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def _range_for_basin(date_range: DateRangeSpec, basin_id: str) -> Optional[DateRange]:
    return date_range.get(basin_id) if isinstance(date_range, dict) else date_range


def prepare_basin_arrays(
    basin_frames: Dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    target_column: str,
    interpolate: bool,
    log_feature_columns: Sequence[str],
    static_attributes: Optional[pd.DataFrame],
    interpolate_target: bool = True,
) -> Tuple[Dict[str, ArrayEntry], List[str]]:
    """Convert basin tables to ordered dynamic, target, date, and static arrays."""

    arrays: Dict[str, ArrayEntry] = {}
    feature_names = list(feature_columns)
    attribute_names = (
        [f"attr_{column}" for column in static_attributes.columns]
        if static_attributes is not None
        else []
    )
    print("Processing basin data...")
    for basin_id, frame in tqdm(
        basin_frames.items(),
        desc="Basins",
        dynamic_ncols=True,
        leave=False,
    ):
        if static_attributes is not None and basin_id not in static_attributes.index:
            continue
        processed = preprocess_basin_frame(
            frame,
            feature_columns,
            target_column,
            interpolate,
            log_feature_columns,
            interpolate_target=interpolate_target,
        )
        if processed.empty:
            continue
        dynamic_features = processed[feature_columns].to_numpy(dtype=np.float32)
        static_vector = (
            static_attributes.loc[basin_id].to_numpy(dtype=np.float32)
            if static_attributes is not None
            else None
        )
        targets = processed[target_column].to_numpy(dtype=np.float32)
        dates = list(processed["day"])
        arrays[basin_id] = (dynamic_features, targets, dates, static_vector)
    if not arrays:
        raise RuntimeError(
            "No basins available after processing. Check static-attribute alignment and feature names."
        )
    full_feature_names = feature_names + attribute_names
    print(f"Prepared data for {len(arrays)} basins.")
    return arrays, full_feature_names


def compute_feature_standardizer(
    basin_arrays: Dict[str, ArrayEntry],
    train_range: DateRangeSpec,
) -> FeatureStandardizer:
    """Fit the 111-column feature scaler on training-period rows.

    Dynamic statistics aggregate finite training values across basins. Each
    basin's static vector is weighted by its number of training-period rows.
    Population moments are accumulated in float64 and returned as float32.
    """

    sample = next(iter(basin_arrays.values()))
    dynamic_count = sample[0].shape[1]
    static_sample = sample[3]
    has_static = static_sample is not None
    static_count = len(static_sample) if has_static else 0

    dynamic_sum = np.zeros(dynamic_count, dtype=np.float64)
    dynamic_sum_squared = np.zeros(dynamic_count, dtype=np.float64)
    dynamic_counts = np.zeros(dynamic_count, dtype=np.float64)
    if has_static:
        static_sum = np.zeros(static_count, dtype=np.float64)
        static_sum_squared = np.zeros(static_count, dtype=np.float64)
        static_counts = np.zeros(static_count, dtype=np.float64)

    for basin_id, (dynamic_features, _targets, dates, static_vector) in basin_arrays.items():
        basin_range = _range_for_basin(train_range, basin_id)
        if basin_range is None:
            continue
        start, end = basin_range
        date_index = pd.DatetimeIndex(dates)
        mask = (date_index >= start) & (date_index < end)
        if not np.any(mask):
            continue
        subset = dynamic_features[mask]
        valid = np.isfinite(subset)
        dynamic_sum += np.where(valid, subset, 0.0).sum(axis=0)
        dynamic_sum_squared += np.where(valid, subset ** 2, 0.0).sum(axis=0)
        dynamic_counts += valid.sum(axis=0)
        if has_static and static_vector is not None:
            training_rows = float(mask.sum())
            static_float64 = static_vector.astype(np.float64)
            static_sum += static_float64 * training_rows
            static_sum_squared += (static_float64 ** 2) * training_rows
            static_counts += training_rows

    dynamic_mean = dynamic_sum / dynamic_counts
    dynamic_variance = np.maximum(
        dynamic_sum_squared / dynamic_counts - dynamic_mean ** 2,
        SMALL_STD,
    )
    dynamic_std = np.sqrt(dynamic_variance)
    dynamic_std[dynamic_std < SMALL_STD] = 1.0

    if has_static:
        static_mean = static_sum / static_counts
        static_variance = np.maximum(
            static_sum_squared / static_counts - static_mean ** 2,
            SMALL_STD,
        )
        static_std = np.sqrt(static_variance)
        static_std[static_std < SMALL_STD] = 1.0
        full_mean = np.concatenate([dynamic_mean, static_mean])
        full_std = np.concatenate([dynamic_std, static_std])
    else:
        full_mean = dynamic_mean
        full_std = dynamic_std
    return FeatureStandardizer(full_mean.astype(np.float32), full_std.astype(np.float32))


def compute_target_standard_deviations(
    basin_arrays: Dict[str, ArrayEntry],
    train_range: DateRangeSpec,
    target_log1p: bool,
) -> Dict[str, float]:
    """Compute per-basin training-period target standard deviations.

    Finite raw runoff targets are selected using an end-exclusive training
    range. When ``target_log1p`` is true, values are clipped at zero and log1p
    transformed before the population standard deviation is computed.
    """

    standard_deviations: Dict[str, float] = {}
    for basin_id, (_features, targets, dates, _static) in basin_arrays.items():
        basin_range = _range_for_basin(train_range, basin_id)
        if basin_range is None:
            continue
        start, end = basin_range
        date_index = pd.DatetimeIndex(dates)
        mask = (date_index >= start) & (date_index < end)
        if not np.any(mask):
            continue
        valid_targets = targets[mask]
        valid_targets = valid_targets[np.isfinite(valid_targets)]
        if target_log1p:
            valid_targets = apply_log_transform(valid_targets)
        if valid_targets.size < 2:
            standard_deviations[basin_id] = 1.0
        else:
            standard_deviation = np.std(valid_targets, dtype=np.float64)
            standard_deviations[basin_id] = (
                float(standard_deviation) if standard_deviation > SMALL_STD else 1.0
            )
    return standard_deviations


def build_model_input_arrays(
    basin_arrays: Dict[str, ArrayEntry],
    scaler: FeatureStandardizer,
) -> Dict[str, ModelInputEntry]:
    """Build contiguous scaled timestep inputs in dynamic-then-static order."""

    model_inputs: Dict[str, ModelInputEntry] = {}
    for basin_id, (dynamic_features, targets, dates, static_vector) in basin_arrays.items():
        if static_vector is not None:
            static_block = np.broadcast_to(
                static_vector,
                (dynamic_features.shape[0], static_vector.shape[0]),
            )
            full_features = np.concatenate([dynamic_features, static_block], axis=1)
        else:
            full_features = dynamic_features
        scaled_features = scaler.transform(full_features).astype(np.float32, copy=False)
        model_inputs[basin_id] = (
            np.ascontiguousarray(scaled_features),
            targets.astype(np.float32, copy=False),
            dates,
        )
    return model_inputs


def build_static_feature_map(
    basin_arrays: Dict[str, Union[ArrayEntry, ModelInputEntry]],
    static_table: pd.DataFrame,
    attribute_columns: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Mean-impute and standardize static routing inputs across participating basins.

    Each attribute uses its column mean and population standard deviation.
    """

    shared_basins = [basin_id for basin_id in basin_arrays if basin_id in static_table.index]
    if not shared_basins:
        raise RuntimeError("No overlapping basins between time series and static attributes.")
    static_values = static_table.loc[shared_basins, attribute_columns].copy()
    column_means = static_values.mean(axis=0)
    static_values = static_values.fillna(column_means)
    values = static_values.to_numpy(dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < SMALL_STD] = 1.0
    normalized = (values - mean) / std
    return {
        basin_id: normalized[index] for index, basin_id in enumerate(shared_basins)
    }, list(attribute_columns)


def build_explicit_date_splits(
    train_start: Union[str, pd.Timestamp],
    train_end: Union[str, pd.Timestamp],
    validation_start: Union[str, pd.Timestamp],
    validation_end: Union[str, pd.Timestamp],
    test_start: Union[str, pd.Timestamp],
    test_end: Union[str, pd.Timestamp],
) -> Tuple[DateRange, DateRange, DateRange]:
    """Return train/validation/test ranges with end-exclusive semantics."""

    return (
        (pd.to_datetime(train_start), pd.to_datetime(train_end)),
        (pd.to_datetime(validation_start), pd.to_datetime(validation_end)),
        (pd.to_datetime(test_start), pd.to_datetime(test_end)),
    )


class StreamflowDataset(Dataset):
    """Sliding-window dataset for the global encoder-decoder LSTM.

    Each item contains a 365-row encoder input, prediction-day decoder input,
    raw runoff target, target mask, basin ID, target-row index, and basin target
    standard deviation. Sequence lengths are constructor arguments. Dates are
    recovered from the stored basin arrays using ``basin_id`` and ``start_idx``.
    """

    def __init__(
        self,
        basin_arrays: Dict[str, ModelInputEntry],
        history_length: int,
        forecast_length: int,
        date_range: DateRangeSpec,
        target_standard_deviations: Dict[str, float],
        allow_missing_targets: bool = False,
    ) -> None:
        self.basin_arrays = basin_arrays
        self.history_length = history_length
        self.forecast_length = forecast_length
        self.per_basin_ranges = date_range if isinstance(date_range, dict) else None
        if self.per_basin_ranges is None:
            self.start, self.end = date_range
        else:
            self.start, self.end = None, None
        self.target_scalers = target_standard_deviations
        self.allow_missing_targets = allow_missing_targets
        self.basin_ids = sorted(self.basin_arrays.keys())
        self._basin_to_index = {basin_id: index for index, basin_id in enumerate(self.basin_ids)}
        sample_basin_indices: List[int] = []
        sample_start_indices: List[int] = []

        for basin_id, (features, targets, dates) in basin_arrays.items():
            if basin_id not in self.target_scalers:
                continue
            if len(features) < history_length + forecast_length:
                continue
            if self.per_basin_ranges is not None:
                bounds = self.per_basin_ranges.get(basin_id)
                if bounds is None:
                    continue
            else:
                bounds = (self.start, self.end)
            start_bound, end_bound = bounds
            for target_index in range(
                history_length,
                len(features) - forecast_length + 1,
            ):
                first_target_date = dates[target_index]
                last_target_date = dates[target_index + forecast_length - 1]
                if not (
                    start_bound <= first_target_date < end_bound
                    and start_bound <= last_target_date < end_bound
                ):
                    continue
                target_slice = targets[target_index : target_index + forecast_length]
                finite_mask = np.isfinite(target_slice)
                if self.allow_missing_targets:
                    if not np.any(finite_mask):
                        continue
                elif not np.all(finite_mask):
                    continue
                sample_basin_indices.append(self._basin_to_index[basin_id])
                sample_start_indices.append(target_index)
        if not sample_basin_indices:
            raise RuntimeError("No samples available for the specified configuration.")
        self.sample_basin_idx = np.asarray(sample_basin_indices, dtype=np.int32)
        self.sample_start_idx = np.asarray(sample_start_indices, dtype=np.int32)

    def __len__(self) -> int:
        return int(self.sample_start_idx.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        basin_index = int(self.sample_basin_idx[index])
        basin_id = self.basin_ids[basin_index]
        start_index = int(self.sample_start_idx[index])
        scaled_features, targets, _dates = self.basin_arrays[basin_id]
        encoder = scaled_features[start_index - self.history_length : start_index]
        decoder = scaled_features[start_index : start_index + self.forecast_length]
        target_slice = targets[start_index : start_index + self.forecast_length]
        target_mask = np.isfinite(target_slice).astype(np.float32)
        target_standard_deviation = self.target_scalers[basin_id]
        return {
            "encoder_inputs": torch.from_numpy(np.asarray(encoder, dtype=np.float32)),
            "decoder_inputs": torch.from_numpy(np.asarray(decoder, dtype=np.float32)),
            "targets": torch.from_numpy(np.asarray(target_slice, dtype=np.float32)),
            "target_mask": torch.from_numpy(target_mask),
            "basin_id": basin_id,
            "start_idx": torch.tensor(start_index, dtype=torch.long),
            "target_std": torch.tensor(target_standard_deviation, dtype=torch.float32),
        }


class MoEStreamflowDataset(StreamflowDataset):
    """Streamflow dataset with a separate 96-column static routing input."""

    def __init__(
        self,
        *args,
        static_features: Dict[str, np.ndarray],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.static_features = static_features

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        basin_id = sample["basin_id"]
        static_vector = self.static_features.get(basin_id)
        if static_vector is None:
            raise KeyError(f"Missing static attributes for basin '{basin_id}'")
        sample["static_features"] = torch.from_numpy(
            np.asarray(static_vector, dtype=np.float32)
        )
        return sample


class FixedRoutingStreamflowDataset(MoEStreamflowDataset):
    """MoE dataset that adds a fixed zero-based expert label per basin."""

    def __init__(
        self,
        *args,
        routing_labels: Dict[str, int],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.routing_labels = routing_labels

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(index)
        basin_id = sample["basin_id"]
        if basin_id not in self.routing_labels:
            raise KeyError(f"Missing fixed routing label for basin '{basin_id}'")
        sample["routing_label"] = torch.tensor(
            self.routing_labels[basin_id], dtype=torch.long
        )
        return sample


def build_streamflow_datasets(
    model_input_arrays: Dict[str, ModelInputEntry],
    target_standard_deviations: Dict[str, float],
    splits: Tuple[DateRangeSpec, DateRangeSpec, DateRangeSpec],
    static_features: Dict[str, np.ndarray],
    history_length: int = 365,
    forecast_length: int = 1,
    allow_missing_targets: bool = True,
    routing_labels: Optional[Dict[str, int]] = None,
) -> Tuple[MoEStreamflowDataset, MoEStreamflowDataset, MoEStreamflowDataset]:
    """Build train, validation, and test MoE datasets with identical rules."""

    dataset_class = (
        FixedRoutingStreamflowDataset if routing_labels is not None else MoEStreamflowDataset
    )
    return tuple(
        dataset_class(
            model_input_arrays,
            history_length,
            forecast_length,
            split,
            target_standard_deviations,
            allow_missing_targets=allow_missing_targets,
            static_features=static_features,
            **({"routing_labels": routing_labels} if routing_labels is not None else {}),
        )
        for split in splits
    )


def load_basin_cluster_assignments(
    path: Path,
    basin_id_column: str = "gauge_id",
    cluster_column: str = "cluster_category",
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Load basin clusters and derive stable zero-based routing labels.

    Cluster names are sorted naturally by their numeric suffix, so ``C0`` to
    ``C8`` map to expert indices 0 to 8.
    """

    table = _read_table(Path(path))
    missing = [
        name for name in (basin_id_column, cluster_column) if name not in table.columns
    ]
    if missing:
        raise KeyError(f"Cluster-assignment table is missing columns: {missing}")
    subset = table[[basin_id_column, cluster_column]].dropna().copy()
    subset[basin_id_column] = subset[basin_id_column].map(canonicalize_basin_id)
    subset[cluster_column] = subset[cluster_column].astype(str).str.strip()
    subset = subset[(subset[basin_id_column] != "") & (subset[cluster_column] != "")]
    if subset[basin_id_column].duplicated().any():
        raise ValueError("Each basin must have exactly one cluster assignment.")

    def natural_key(value: str) -> Tuple[str, int]:
        prefix = value.rstrip("0123456789")
        suffix = value[len(prefix) :]
        return prefix, int(suffix) if suffix else -1

    clusters = sorted(subset[cluster_column].unique().tolist(), key=natural_key)
    cluster_to_index = {cluster: index for index, cluster in enumerate(clusters)}
    basin_to_cluster = dict(zip(subset[basin_id_column], subset[cluster_column]))
    routing_labels = {
        basin_id: cluster_to_index[cluster]
        for basin_id, cluster in basin_to_cluster.items()
    }
    return basin_to_cluster, routing_labels


def save_feature_standardizer(
    output_dir: Path,
    scaler: FeatureStandardizer,
    feature_names: Sequence[str],
    log_feature_columns: Sequence[str],
) -> None:
    """Save the feature standardizer."""

    payload = {
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
        "feature_columns": list(feature_names),
    }
    if log_feature_columns:
        payload["log_transformed_features"] = list(log_feature_columns)
    (output_dir / "feature_scaler.json").write_text(json.dumps(payload, indent=2))


def load_feature_standardizer(
    path: Path,
    expected_feature_names: Optional[Sequence[str]] = None,
) -> Tuple[FeatureStandardizer, List[str], List[str]]:
    """Load a saved feature standardizer."""

    payload = json.loads(path.read_text())
    feature_names = list(payload["feature_columns"])
    if expected_feature_names is not None and feature_names != list(expected_feature_names):
        raise ValueError("Feature-scaler column order does not match the expected feature order.")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != std.shape or mean.shape != (len(feature_names),):
        raise ValueError("Feature-scaler dimensions do not match feature_columns.")
    scaler = FeatureStandardizer(mean=mean, std=std)
    return scaler, feature_names, list(payload.get("log_transformed_features", []))


def save_npy_cache(
    model_inputs: Dict[str, ModelInputEntry],
    target_standard_deviations: Dict[str, float],
    feature_names: Sequence[str],
    cache_dir: Path,
    scaler: Optional[FeatureStandardizer] = None,
    *,
    target_column: str,
    dynamic_feature_names: Sequence[str],
    log_feature_names: Sequence[str],
    history_length: int,
    forecast_length: int,
) -> None:
    """Save model-ready arrays in an NPY cache."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    basin_dates: Dict[str, List[str]] = {}
    basin_ids = sorted(model_inputs)
    for basin_id in tqdm(basin_ids, desc="Save NPY cache", dynamic_ncols=True, leave=False):
        scaled_features, targets, dates = model_inputs[basin_id]
        np.save(cache_dir / f"{basin_id}_features.npy", np.asarray(scaled_features, dtype=np.float32))
        np.save(cache_dir / f"{basin_id}_targets.npy", np.asarray(targets, dtype=np.float32))
        basin_dates[basin_id] = [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates]
    (cache_dir / "basin_dates.json").write_text(json.dumps(basin_dates))
    metadata = {
        "basin_ids": basin_ids,
        "feature_names": list(feature_names),
        "target_scalers": target_standard_deviations,
        "target_column": str(target_column),
        "dynamic_features": list(dynamic_feature_names),
        "log_features": list(log_feature_names),
        "history_length": int(history_length),
        "forecast_length": int(forecast_length),
    }
    if scaler is not None:
        metadata["scaler_mean"] = scaler.mean.tolist()
        metadata["scaler_std"] = scaler.std.tolist()
    (cache_dir / "cache_metadata.json").write_text(json.dumps(metadata, indent=2))


def load_npy_cache(
    cache_dir: Path,
) -> Tuple[Dict[str, ModelInputEntry], Dict[str, object]]:
    """Load model inputs from a memory-mapped NPY cache."""

    metadata_path = cache_dir / "cache_metadata.json"
    dates_path = cache_dir / "basin_dates.json"
    if not metadata_path.exists() or not dates_path.exists():
        raise FileNotFoundError("Missing cache metadata or dates file.")
    metadata = json.loads(metadata_path.read_text())
    basin_dates = json.loads(dates_path.read_text())
    basin_ids = sorted(metadata.get("basin_ids", []))
    model_inputs: Dict[str, ModelInputEntry] = {}
    for basin_id in tqdm(basin_ids, desc="Load NPY cache", dynamic_ncols=True, leave=False):
        features_path = cache_dir / f"{basin_id}_features.npy"
        targets_path = cache_dir / f"{basin_id}_targets.npy"
        if not features_path.exists() or not targets_path.exists() or basin_id not in basin_dates:
            continue
        scaled_features = np.load(features_path, mmap_mode="r")
        targets = np.load(targets_path, mmap_mode="r")
        dates = [pd.Timestamp(date) for date in basin_dates[basin_id]]
        model_inputs[basin_id] = (scaled_features, targets, dates)
    if not model_inputs:
        raise RuntimeError("NPY cache exists but no valid basin arrays were loaded.")
    return model_inputs, metadata


def is_npy_cache_compatible(
    cache_metadata: Dict[str, object],
    *,
    expected_feature_names: Optional[Sequence[str]],
    target_column: str,
    dynamic_feature_names: Sequence[str],
    log_feature_names: Sequence[str],
    history_length: int,
    forecast_length: int,
) -> bool:
    """Compare NPY cache metadata with the requested input specification.

    Compare feature order, target and dynamic feature names, log-transformed
    features, and input-window lengths.
    """

    feature_names_match = True
    if expected_feature_names is not None:
        feature_names_match = cache_metadata.get("feature_names") == list(expected_feature_names)
    return (
        feature_names_match
        and cache_metadata.get("target_column") == str(target_column)
        and cache_metadata.get("dynamic_features") == list(dynamic_feature_names)
        and cache_metadata.get("log_features") == list(log_feature_names)
        and int(cache_metadata.get("history_length", -1)) == int(history_length)
        and int(cache_metadata.get("forecast_length", -1)) == int(forecast_length)
    )
