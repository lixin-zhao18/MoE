"""Hydrological evaluation metrics."""

from __future__ import annotations

import math

import numpy as np


__all__ = [
    "compute_nse",
    "compute_kge",
    "pbias_high_flow",
    "pbias_low_flow",
]


def compute_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Return the Nash-Sutcliffe efficiency (NSE)."""

    if observed.size == 0 or simulated.size == 0:
        return float("nan")

    mean_observed = np.mean(observed)
    denominator = np.sum((observed - mean_observed) ** 2)
    if denominator <= 0:
        return float("nan")

    numerator = np.sum((observed - simulated) ** 2)
    return 1.0 - numerator / denominator


def compute_kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Return the Kling-Gupta efficiency (KGE)."""

    if observed.size == 0 or simulated.size == 0:
        return float("nan")

    mean_observed = np.mean(observed)
    mean_simulated = np.mean(simulated)
    std_observed = np.std(observed)
    std_simulated = np.std(simulated)

    if std_observed == 0 or std_simulated == 0 or mean_observed == 0:
        return float("nan")

    correlation = np.corrcoef(observed, simulated)[0, 1]
    variability_ratio = std_simulated / std_observed
    mean_ratio = mean_simulated / mean_observed

    if any(np.isnan([correlation, variability_ratio, mean_ratio])):
        return float("nan")

    return 1.0 - math.sqrt(
        (correlation - 1.0) ** 2
        + (variability_ratio - 1.0) ** 2
        + (mean_ratio - 1.0) ** 2
    )


def pbias_high_flow(
    observed: np.ndarray,
    simulated: np.ndarray,
    top_fraction: float = 0.02,
) -> float:
    """Return PBIASHF (%) for the highest observed flows."""

    if observed.size == 0 or simulated.size == 0:
        return float("nan")

    top_fraction = max(min(top_fraction, 1.0), 0.0)
    if top_fraction <= 0:
        return float("nan")

    count = max(1, int(np.ceil(observed.size * top_fraction)))
    indices = np.argsort(observed)[-count:]
    observed_high = observed[indices]
    simulated_high = simulated[indices]

    denominator = observed_high.sum()
    if denominator == 0:
        return float("nan")

    return 100.0 * (simulated_high.sum() - denominator) / denominator


def pbias_low_flow(
    observed: np.ndarray,
    simulated: np.ndarray,
    bottom_fraction: float = 0.30,
) -> float:
    """Return PBIASLF (%) for the lowest observed flows."""

    if observed.size == 0 or simulated.size == 0:
        return float("nan")

    bottom_fraction = max(min(bottom_fraction, 1.0), 0.0)
    if bottom_fraction <= 0:
        return float("nan")

    count = max(1, int(np.ceil(observed.size * bottom_fraction)))
    indices = np.argsort(observed)[:count]
    observed_low = observed[indices]
    simulated_low = simulated[indices]

    denominator = observed_low.sum()
    if denominator == 0:
        return float("nan")

    return 100.0 * (simulated_low.sum() - denominator) / denominator
