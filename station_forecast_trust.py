"""Leakage-safe historical body centers for forecast-trust diffusion.

The current issuance forecast remains available for all 168 lead hours.  The
alternative center is built only from *training* actual trajectories retrieved
with information available at issuance: the current forecast, the aligned
previous forecast revision, the most recently observed forecast error, and the
calendar date.  Future query actuals are never part of a retrieval key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HOURS = 168
STATIONS = 24


@dataclass(frozen=True)
class ForecastTrustArrays:
    center: np.ndarray
    dispersion: np.ndarray
    neighbor_index: np.ndarray
    distance: np.ndarray
    weight: np.ndarray
    audit: dict[str, object]


def _resample(values: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(0, len(values), bins + 1).round().astype(int)
    blocks = []
    for left, right in zip(edges[:-1], edges[1:]):
        blocks.append(values[left : max(left + 1, right)].mean(axis=0))
    return np.stack(blocks)


def _previous_indices(issue_frame: pd.DataFrame) -> np.ndarray:
    days = pd.to_datetime(issue_frame["issue_date"]).dt.normalize()
    lookup = {day: index for index, day in enumerate(days)}
    return np.asarray(
        [lookup.get(day - pd.Timedelta(days=1), -1) for day in days],
        dtype=np.int64,
    )


def _causal_features(
    forecast: np.ndarray,
    residual: np.ndarray,
    issues: pd.DataFrame,
    capacities: np.ndarray,
    wind_indices: np.ndarray,
    solar_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return issuance-time-only keys and previous-issue availability."""

    previous = _previous_indices(issues)
    capacity_weight = capacities / capacities.sum()
    wind_weight = capacities[wind_indices] / capacities[wind_indices].sum()
    solar_weight = capacities[solar_indices] / capacities[solar_indices].sum()
    features: list[np.ndarray] = []
    for index in range(len(forecast)):
        current = np.asarray(forecast[index], dtype=np.float64)
        daily = _resample(current, 7)
        aggregate_wind = daily[:, wind_indices] @ wind_weight
        aggregate_solar = daily[:, solar_indices] @ solar_weight
        aggregate_all = daily @ capacity_weight
        aggregate_ramp = np.diff(aggregate_all, prepend=aggregate_all[:1])
        revision_mean = np.zeros(STATIONS, dtype=np.float64)
        revision_abs = np.zeros(STATIONS, dtype=np.float64)
        recent_mean = np.zeros(STATIONS, dtype=np.float64)
        recent_abs = np.zeros(STATIONS, dtype=np.float64)
        recent_last = np.zeros(STATIONS, dtype=np.float64)
        available = float(previous[index] >= 0)
        if previous[index] >= 0:
            old = np.asarray(forecast[previous[index]], dtype=np.float64)
            revision = current[:144] - old[24:]
            revision_mean = revision.mean(axis=0)
            revision_abs = np.abs(revision).mean(axis=0)
            recent = np.asarray(residual[previous[index], :24], dtype=np.float64)
            recent_mean = recent.mean(axis=0)
            recent_abs = np.abs(recent).mean(axis=0)
            recent_last = recent[-1]
        day = pd.Timestamp(issues.iloc[index]["issue_date"])
        angle = 2.0 * np.pi * float(day.dayofyear) / 365.25
        feature = np.concatenate(
            [
                daily.reshape(-1),
                aggregate_wind,
                aggregate_solar,
                aggregate_all,
                aggregate_ramp,
                revision_mean,
                revision_abs,
                recent_mean,
                recent_abs,
                recent_last,
                np.asarray([available, np.sin(angle), np.cos(angle)]),
            ]
        )
        features.append(feature)
    return np.stack(features), previous


def build_forecast_trust_arrays(
    data_dir: str | Path,
    split: str,
    top_k: int = 24,
    exclusion_days: int = 6,
    temperature: float = 0.75,
) -> ForecastTrustArrays:
    """Build a train-only historical actual center for one data split."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split={split!r}")
    if top_k < 4:
        raise ValueError("forecast trust top_k must be at least 4")
    if temperature <= 0:
        raise ValueError("forecast trust temperature must be positive")
    root = Path(data_dir)
    stations = pd.read_csv(root / "station_order.csv").sort_values(
        "channel_index"
    )
    capacities = stations["capacity_mw"].to_numpy(float)
    wind_indices = np.flatnonzero(stations["data_type"].to_numpy() == "wind")
    solar_indices = np.flatnonzero(stations["data_type"].to_numpy() == "solar")
    train_forecast = np.load(root / "train_forecast.npy", mmap_mode="r")
    train_actual = np.load(root / "train_actual.npy", mmap_mode="r")
    train_residual = np.load(root / "train_residual.npy", mmap_mode="r")
    train_issues = pd.read_csv(root / "train_issue_dates.csv")
    query_forecast = np.load(root / f"{split}_forecast.npy", mmap_mode="r")
    query_residual = np.load(root / f"{split}_residual.npy", mmap_mode="r")
    query_issues = pd.read_csv(root / f"{split}_issue_dates.csv")
    train_key, train_previous = _causal_features(
        train_forecast,
        train_residual,
        train_issues,
        capacities,
        wind_indices,
        solar_indices,
    )
    query_key, query_previous = _causal_features(
        query_forecast,
        query_residual,
        query_issues,
        capacities,
        wind_indices,
        solar_indices,
    )
    mean = train_key.mean(axis=0)
    std = train_key.std(axis=0)
    std[std < 1e-5] = 1.0
    train_key = np.nan_to_num((train_key - mean) / std)
    query_key = np.nan_to_num((query_key - mean) / std)
    train_days = pd.to_datetime(train_issues["issue_date"]).dt.normalize().to_numpy()
    query_days = pd.to_datetime(query_issues["issue_date"]).dt.normalize().to_numpy()
    count = len(query_key)
    if top_k >= len(train_key):
        raise ValueError("forecast trust top_k must be smaller than train issue count")
    index_array = np.full((count, top_k), -1, dtype=np.int64)
    distance_array = np.zeros((count, top_k), dtype=np.float32)
    weight_array = np.zeros((count, top_k), dtype=np.float32)
    center = np.zeros((count, HOURS, STATIONS), dtype=np.float32)
    dispersion = np.zeros_like(center)
    for query_index in range(count):
        distance = np.mean((train_key - query_key[query_index]) ** 2, axis=1)
        allowed = np.ones(len(train_key), dtype=bool)
        if split == "train":
            separation = np.abs(
                (train_days - query_days[query_index])
                .astype("timedelta64[D]")
                .astype(np.int64)
            )
            allowed &= separation > int(exclusion_days)
        available = np.flatnonzero(allowed)
        if len(available) < top_k:
            raise ValueError(
                f"query {query_index} has only {len(available)} leakage-safe analogs"
            )
        local = available[np.argpartition(distance[available], top_k - 1)[:top_k]]
        local = local[np.argsort(distance[local])]
        selected_distance = distance[local]
        scale = max(float(np.median(selected_distance)), 1e-6)
        logits = -(selected_distance - selected_distance.min()) / (
            float(temperature) * scale
        )
        weight = np.exp(logits - logits.max())
        weight = 0.95 * weight / weight.sum() + 0.05 / top_k
        histories = np.asarray(train_actual[local], dtype=np.float64)
        # Preserve one coherent 168 h trajectory as the alternative body
        # center. Averaging analog actuals would smooth precisely the abrupt
        # changes this experiment is designed to recover. The remaining
        # neighbors estimate uncertainty only.
        historical_center = histories[0]
        historical_variance = np.einsum(
            "k,kts->ts", weight, (histories - historical_center[None]) ** 2
        )
        index_array[query_index] = local
        distance_array[query_index] = (selected_distance / scale).astype(np.float32)
        weight_array[query_index] = weight.astype(np.float32)
        center[query_index] = historical_center.astype(np.float32)
        dispersion[query_index] = np.sqrt(historical_variance + 1e-8).astype(
            np.float32
        )
    effective_k = 1.0 / np.sum(weight_array**2, axis=1)
    train_cv_fraction = None
    if split == "train":
        train_cv_fraction = []
        query_forecast_array = np.asarray(query_forecast, dtype=np.float64)
        query_actual_array = np.asarray(train_actual, dtype=np.float64)
        for day in range(7):
            section = slice(day * 24, (day + 1) * 24)
            delta = center[:, section] - query_forecast_array[:, section]
            needed = query_actual_array[:, section] - query_forecast_array[:, section]
            denominator = float(np.sum(delta**2))
            coefficient = (
                float(np.sum(delta * needed) / denominator)
                if denominator > 0
                else 0.0
            )
            train_cv_fraction.append(float(np.clip(coefficient, 0.02, 0.80)))
    return ForecastTrustArrays(
        center=center,
        dispersion=dispersion,
        neighbor_index=index_array,
        distance=distance_array,
        weight=weight_array,
        audit={
            "method": "causal_nearest_analog_actual_dual_center_v2",
            "split": split,
            "query_count": int(count),
            "top_k": int(top_k),
            "exclusion_days": int(exclusion_days),
            "temperature": float(temperature),
            "key_source": [
                "current_issued_forecast_168h",
                "aligned_previous_issue_forecast_revision",
                "most_recent_observed_forecast_error_24h",
                "issue_calendar",
            ],
            "value_source": "train_actual_168h",
            "center_selection": "nearest_causal_forecast_analogue_complete_actual_trajectory",
            "neighbor_averaging_used_for_center": False,
            "neighbor_set_role": "dispersion_estimation_and_audit_only",
            "mean_effective_k": float(effective_k.mean()),
            "mean_nearest_neighbor_weight": float(weight_array[:, 0].mean()),
            "train_cross_retrieval_least_squares_history_fraction_by_lead_day": (
                train_cv_fraction
            ),
            "query_previous_issue_available_count": int(
                np.sum(query_previous >= 0)
            ),
            "train_previous_issue_available_count": int(
                np.sum(train_previous >= 0)
            ),
            "future_query_actual_used": False,
            "test_target_used": False,
        },
    )
