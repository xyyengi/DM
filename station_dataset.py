"""Dataset utilities for 24-station, 168-hour forecast-vintage experiments."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.eval.physical_projection import _solar_elevation_degrees
from station_jstd_targets import JSTDTargetArrays


EXPECTED_STATIONS = 24
EXPECTED_HOURS = 168


def _same_length_average_numpy(value: np.ndarray, width: int) -> np.ndarray:
    """Reflection-padded moving average over the last axis."""

    value = np.asarray(value, dtype=np.float32)
    width = int(width)
    if not 1 <= width < value.shape[-1]:
        raise ValueError("moving-average width must be in [1, length)")
    left = (width - 1) // 2
    right = width - 1 - left
    padded = np.pad(value, ((0, 0), (left, right)), mode="reflect")
    cumulative = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros((*cumulative.shape[:-1], 1), dtype=np.float64), cumulative],
        axis=-1,
    )
    return ((cumulative[..., width:] - cumulative[..., :-width]) / width).astype(
        np.float32
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)


def validate_station_data_dir(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    required = [
        "export_metadata.json",
        "station_order.csv",
        "station_features.npy",
        "station_adjacency.npy",
    ]
    for split in ("train", "val", "test"):
        required.extend(
            [
                f"{split}_forecast.npy",
                f"{split}_actual.npy",
                f"{split}_residual.npy",
                f"{split}_time_mark.npy",
                f"{split}_lead_mark.npy",
                f"{split}_fill_mask.npy",
                f"{split}_issue_dates.csv",
            ]
        )
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"station data artifacts missing: {missing}")
    return data_dir


def fit_station_residual_scale(
    data_dir: str | Path,
    epsilon: float = 1e-4,
    method: str = "per_station_std",
    condition_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit residual normalization on valid train values only."""
    data_dir = validate_station_data_dir(data_dir)
    residual = np.load(data_dir / "train_residual.npy", mmap_mode="r")
    fill_mask = np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")
    if residual.shape != fill_mask.shape:
        raise ValueError("train residual and fill mask shapes must match")
    scales = []
    counts = []
    for station_index in range(residual.shape[-1]):
        valid = fill_mask[:, :, station_index] == 0
        values = np.asarray(residual[:, :, station_index][valid], dtype=np.float64)
        scale = max(float(values.std()), float(epsilon))
        scales.append(scale)
        counts.append(int(values.size))
    result: dict[str, object] = {
        "method": "per_station_std",
        "fit_split": "train",
        "center": False,
        "epsilon": float(epsilon),
        "scale": scales,
        "valid_value_count": counts,
    }
    if method == "per_station_std":
        return result
    if method != "wind_factorized_condition_std":
        raise ValueError(f"unsupported residual scaling method={method!r}")

    config = dict(condition_config or {})
    forecast = np.asarray(
        np.load(data_dir / "train_forecast.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    residual_array = np.asarray(residual, dtype=np.float64)
    valid = np.asarray(fill_mask) == 0
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    wind_mask = np.zeros(forecast.shape, dtype=bool)
    wind_mask[:, :, wind_indices] = True
    valid &= wind_mask

    issue_frame = pd.read_csv(data_dir / "train_issue_dates.csv")
    issue_days = pd.to_datetime(issue_frame["issue_date"]).dt.normalize()
    lookup = {timestamp: index for index, timestamp in enumerate(issue_days)}
    previous = np.asarray(
        [lookup.get(timestamp - pd.Timedelta(days=1), -1) for timestamp in issue_days],
        dtype=np.int64,
    )
    revision = np.full_like(forecast, np.nan)
    for index, previous_index in enumerate(previous):
        if previous_index >= 0:
            revision[index, :144] = (
                forecast[index, :144] - forecast[previous_index, 24:]
            )

    ramp_lag = int(config.get("ramp_lag", 3))
    if not 1 <= ramp_lag < EXPECTED_HOURS:
        raise ValueError("conditional residual ramp_lag must be between 1 and 167")
    ramp = np.full_like(forecast, np.nan)
    ramp[:, ramp_lag:] = np.abs(
        forecast[:, ramp_lag:] - forecast[:, :-ramp_lag]
    )
    lead_day = np.broadcast_to(
        (np.arange(EXPECTED_HOURS) // 24 + 1)[None, :, None],
        forecast.shape,
    ).astype(np.float64)

    def quantile_edges(values: np.ndarray, quantiles: list[float]) -> np.ndarray:
        selected = values[valid & np.isfinite(values)]
        if not len(selected):
            raise ValueError("no valid train values for conditional residual scaling")
        edges = np.quantile(selected, quantiles)
        edges[0], edges[-1] = -1.0e30, 1.0e30
        for edge_index in range(1, len(edges)):
            if edges[edge_index] <= edges[edge_index - 1]:
                edges[edge_index] = np.nextafter(edges[edge_index - 1], np.inf)
        return edges

    level_quantiles = [
        float(value)
        for value in config.get("forecast_level_quantiles", [0, 0.2, 0.4, 0.6, 0.8, 1])
    ]
    ramp_quantiles = [
        float(value)
        for value in config.get("forecast_ramp_quantiles", [0, 0.25, 0.5, 0.75, 1])
    ]
    revision_quantiles = [
        float(value)
        for value in config.get("forecast_revision_quantiles", [0, 0.25, 0.5, 0.75, 1])
    ]
    for name, quantiles in [
        ("forecast_level", level_quantiles),
        ("forecast_ramp", ramp_quantiles),
        ("forecast_revision", revision_quantiles),
    ]:
        if quantiles[0] != 0.0 or quantiles[-1] != 1.0 or any(
            right <= left for left, right in zip(quantiles, quantiles[1:])
        ):
            raise ValueError(f"invalid {name} quantiles={quantiles}")

    feature_values = {
        "forecast_level": forecast,
        "lead_day": lead_day,
        "forecast_ramp": ramp,
        "forecast_revision": np.abs(revision),
    }
    edges = {
        "forecast_level": quantile_edges(forecast, level_quantiles),
        "lead_day": np.arange(0.5, 8.5, 1.0, dtype=np.float64),
        "forecast_ramp": quantile_edges(ramp, ramp_quantiles),
        "forecast_revision": quantile_edges(np.abs(revision), revision_quantiles),
    }
    bin_index = {
        name: np.digitize(
            feature_values[name], condition_edges[1:-1], right=False
        )
        for name, condition_edges in edges.items()
    }
    factor_values = {
        name: np.ones(len(condition_edges) - 1, dtype=np.float64)
        for name, condition_edges in edges.items()
    }
    base = np.asarray(scales, dtype=np.float64)[None, None, :]
    iterations = int(config.get("factor_iterations", 6))
    factor_clip = config.get("factor_clip", [0.5, 2.0])
    clip_low, clip_high = float(factor_clip[0]), float(factor_clip[1])
    if iterations <= 0 or not 0 < clip_low < clip_high:
        raise ValueError("invalid conditional residual factor fitting settings")

    for _ in range(iterations):
        for name, condition_edges in edges.items():
            other_multiplier = np.ones_like(forecast, dtype=np.float64)
            for other_name, other_factors in factor_values.items():
                if other_name == name:
                    continue
                other_valid = np.isfinite(feature_values[other_name])
                other_multiplier[other_valid] *= other_factors[
                    bin_index[other_name][other_valid]
                ]
            normalized = residual_array / (base * other_multiplier)
            values = feature_values[name]
            counts_by_bin = []
            updated = []
            for current_bin in range(len(condition_edges) - 1):
                selected = (
                    valid
                    & np.isfinite(values)
                    & (bin_index[name] == current_bin)
                )
                sample = normalized[selected]
                counts_by_bin.append(int(sample.size))
                updated.append(float(np.std(sample)) if sample.size else 1.0)
            updated_array = np.clip(
                np.asarray(updated, dtype=np.float64), clip_low, clip_high
            )
            counts_array = np.asarray(counts_by_bin, dtype=np.float64)
            if counts_array.sum() > 0:
                log_center = np.average(
                    np.log(np.maximum(updated_array, epsilon)),
                    weights=counts_array,
                )
                updated_array = np.clip(
                    updated_array / np.exp(log_center), clip_low, clip_high
                )
            factor_values[name] = updated_array

    combined_multiplier = np.ones_like(forecast, dtype=np.float64)
    for name, fitted_factors in factor_values.items():
        available = np.isfinite(feature_values[name])
        combined_multiplier[available] *= fitted_factors[
            bin_index[name][available]
        ]
    unconditional_scales = np.asarray(scales, dtype=np.float64)
    adjusted_scales = unconditional_scales.copy()
    for station_index in wind_indices:
        selected = valid[:, :, station_index]
        standardized = residual_array[:, :, station_index][selected] / (
            unconditional_scales[station_index]
            * combined_multiplier[:, :, station_index][selected]
        )
        adjusted_scales[station_index] *= max(float(np.std(standardized)), epsilon)

    result.update(
        {
            "method": "wind_factorized_condition_std",
            "future_condition_source": "issued_forecast_and_previous_issue_forecast",
            "future_actual_used_as_condition": False,
            "wind_station_indices": wind_indices.tolist(),
            "ramp_lag": ramp_lag,
            "factor_iterations": iterations,
            "factor_clip": [clip_low, clip_high],
            "unconditional_station_scale": unconditional_scales.tolist(),
            "scale": adjusted_scales.tolist(),
            "condition_edges": {
                name: condition_edges.tolist()
                for name, condition_edges in edges.items()
            },
            "condition_factors": {
                name: values.tolist() for name, values in factor_values.items()
            },
            "missing_revision_factor": 1.0,
        }
    )
    return result


def validate_residual_scale(
    residual_scale: Mapping[str, object],
    station_count: int = EXPECTED_STATIONS,
) -> np.ndarray:
    if residual_scale.get("fit_split") != "train":
        raise ValueError("residual scale must be fitted on train")
    if bool(residual_scale.get("center", False)):
        raise ValueError("station experiment uses scale-only residual normalization")
    scale = np.asarray(residual_scale.get("scale"), dtype=np.float32)
    if scale.shape != (station_count,) or not np.isfinite(scale).all():
        raise ValueError(f"invalid residual scale shape/value: {scale.shape}")
    if np.any(scale <= 0):
        raise ValueError("all residual scales must be positive")
    method = residual_scale.get("method", "per_station_std")
    if method not in {"per_station_std", "wind_factorized_condition_std"}:
        raise ValueError(f"unsupported residual scale method={method!r}")
    if method == "wind_factorized_condition_std":
        if bool(residual_scale.get("future_actual_used_as_condition", True)):
            raise ValueError("future actual cannot be used for residual scaling")
        wind_indices = np.asarray(
            residual_scale.get("wind_station_indices"), dtype=np.int64
        )
        if (
            wind_indices.ndim != 1
            or not len(wind_indices)
            or np.any(wind_indices < 0)
            or np.any(wind_indices >= station_count)
        ):
            raise ValueError("invalid wind station indices in residual scale")
        edges = residual_scale.get("condition_edges", {})
        factors = residual_scale.get("condition_factors", {})
        for name in [
            "forecast_level",
            "lead_day",
            "forecast_ramp",
            "forecast_revision",
        ]:
            condition_edges = np.asarray(edges.get(name), dtype=np.float64)
            condition_factors = np.asarray(factors.get(name), dtype=np.float64)
            if (
                condition_edges.ndim != 1
                or len(condition_edges) != len(condition_factors) + 1
                or not np.all(np.diff(condition_edges) > 0)
                or not np.isfinite(condition_factors).all()
                or np.any(condition_factors <= 0)
            ):
                raise ValueError(f"invalid conditional residual factor {name}")
    return scale


def build_station_residual_scale_tensor(
    residual_scale: Mapping[str, object],
    forecast: np.ndarray,
    forecast_revision: np.ndarray | None = None,
    revision_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build generation-known [station, lead] residual scales."""
    base = validate_residual_scale(residual_scale)
    forecast = np.asarray(forecast, dtype=np.float32)
    if forecast.shape != (EXPECTED_STATIONS, EXPECTED_HOURS):
        raise ValueError(f"forecast expected (24,168), got {forecast.shape}")
    output = np.broadcast_to(base[:, None], forecast.shape).copy()
    if residual_scale.get("method", "per_station_std") == "per_station_std":
        return output

    wind_indices = np.asarray(
        residual_scale["wind_station_indices"], dtype=np.int64
    )
    ramp_lag = int(residual_scale["ramp_lag"])
    ramp = np.full_like(forecast, np.nan)
    ramp[:, ramp_lag:] = np.abs(
        forecast[:, ramp_lag:] - forecast[:, :-ramp_lag]
    )
    lead_day = np.broadcast_to(
        (np.arange(EXPECTED_HOURS) // 24 + 1)[None, :], forecast.shape
    ).astype(np.float32)
    if forecast_revision is None:
        forecast_revision = np.zeros_like(forecast)
    if revision_mask is None:
        revision_mask = np.zeros_like(forecast)
    features = {
        "forecast_level": forecast,
        "lead_day": lead_day,
        "forecast_ramp": ramp,
        "forecast_revision": np.abs(np.asarray(forecast_revision, dtype=np.float32)),
    }
    edges = residual_scale["condition_edges"]
    factors = residual_scale["condition_factors"]
    multiplier = np.ones_like(forecast, dtype=np.float32)
    for name, values in features.items():
        condition_edges = np.asarray(edges[name], dtype=np.float64)
        condition_factors = np.asarray(factors[name], dtype=np.float32)
        valid = np.isfinite(values)
        if name == "forecast_revision":
            valid &= np.asarray(revision_mask) > 0
        bins = np.digitize(values[valid], condition_edges[1:-1], right=False)
        multiplier[valid] *= condition_factors[bins]
    output[wind_indices] *= multiplier[wind_indices]
    if not np.isfinite(output).all() or np.any(output <= 0):
        raise ValueError("conditional residual scale tensor is invalid")
    return output


def fit_station_state_thresholds(
    data_dir: str | Path,
    low_quantile: float = 0.20,
    high_quantile: float = 0.90,
    ramp_quantile: float = 0.90,
    ramp_lags: tuple[int, ...] = (3, 6),
    epsilon: float = 1e-4,
) -> dict[str, object]:
    """Fit state-v1 thresholds on unique train target hours only.

    Future state inputs are always computed from forecast.  Train actual is used
    here only to establish fixed risk thresholds, which are frozen for validation,
    test, and deployment generation.
    """
    data_dir = validate_station_data_dir(data_dir)
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError("state low/high quantiles must satisfy 0 < low < high < 1")
    if not 0.0 < ramp_quantile < 1.0:
        raise ValueError("state ramp quantile must be between 0 and 1")
    ramp_lags = tuple(int(value) for value in ramp_lags)
    if not ramp_lags or any(value <= 0 or value >= EXPECTED_HOURS for value in ramp_lags):
        raise ValueError(f"invalid state ramp lags={ramp_lags}")

    actual = np.load(data_dir / "train_actual.npy", mmap_mode="r")
    fill_mask = np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")
    daylight, _ = build_station_daylight_mask(data_dir, "train")
    issues = pd.read_csv(data_dir / "train_issue_dates.csv")
    starts = pd.to_datetime(issues["target_start"]).to_numpy(dtype="datetime64[h]")
    target_hours = starts[:, None] + np.arange(EXPECTED_HOURS).astype("timedelta64[h]")
    flat_hours = target_hours.reshape(-1).astype("datetime64[h]").astype(np.int64)

    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    lows: list[float] = []
    highs: list[float] = []
    counts: list[int] = []
    ramp_scales = {str(lag): [] for lag in ramp_lags}
    for station_index in range(EXPECTED_STATIONS):
        valid = np.asarray(fill_mask[:, :, station_index] == 0).reshape(-1)
        if stations.iloc[station_index].data_type == "solar":
            valid &= daylight[:, :, station_index].reshape(-1)
        values = np.asarray(actual[:, :, station_index], dtype=np.float64).reshape(-1)
        valid_indices = np.flatnonzero(valid & np.isfinite(values))
        valid_hours = flat_hours[valid_indices]
        _, first_positions = np.unique(valid_hours, return_index=True)
        selected = valid_indices[first_positions]
        hours = flat_hours[selected]
        unique_values = values[selected]
        order = np.argsort(hours)
        hours = hours[order]
        unique_values = unique_values[order]
        if unique_values.size < 2:
            raise ValueError(f"insufficient train state values for station {station_index}")
        low = float(np.quantile(unique_values, low_quantile))
        high = float(np.quantile(unique_values, high_quantile))
        if high - low < float(epsilon):
            high = low + float(epsilon)
        lows.append(low)
        highs.append(high)
        counts.append(int(unique_values.size))
        for lag in ramp_lags:
            _, current_index, previous_index = np.intersect1d(
                hours, hours + int(lag), return_indices=True
            )
            differences = np.abs(
                unique_values[current_index] - unique_values[previous_index]
            )
            scale = max(float(np.quantile(differences, ramp_quantile)), float(epsilon))
            ramp_scales[str(lag)].append(scale)

    return {
        "method": "train_actual_unique_target_hour_quantiles",
        "fit_split": "train",
        "future_state_source": "current_issued_forecast_only",
        "future_actual_used_as_condition": False,
        "solar_daylight_only": True,
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "ramp_quantile": float(ramp_quantile),
        "ramp_lags": list(ramp_lags),
        "epsilon": float(epsilon),
        "low_threshold": lows,
        "high_threshold": highs,
        "ramp_abs_scale": ramp_scales,
        "unique_valid_hour_count": counts,
    }


def validate_station_state_thresholds(
    thresholds: Mapping[str, object],
    station_count: int = EXPECTED_STATIONS,
) -> dict[str, object]:
    if thresholds.get("fit_split") != "train":
        raise ValueError("state thresholds must be fitted on train")
    if thresholds.get("future_state_source") != "current_issued_forecast_only":
        raise ValueError("state thresholds declare an invalid future state source")
    if bool(thresholds.get("future_actual_used_as_condition", True)):
        raise ValueError("future actual cannot be used as a state condition")
    low = np.asarray(thresholds.get("low_threshold"), dtype=np.float32)
    high = np.asarray(thresholds.get("high_threshold"), dtype=np.float32)
    if low.shape != (station_count,) or high.shape != (station_count,):
        raise ValueError("state low/high threshold shape does not match station count")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("state low/high thresholds are invalid")
    lags = tuple(int(value) for value in thresholds.get("ramp_lags", []))
    scales = thresholds.get("ramp_abs_scale", {})
    for lag in lags:
        value = np.asarray(scales.get(str(lag)), dtype=np.float32)
        if value.shape != (station_count,) or not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f"invalid state ramp scale for lag={lag}")
    return dict(thresholds)


def fit_station_event_weighting(
    data_dir: str | Path,
    condition_config: Mapping[str, object],
) -> dict[str, object]:
    """Fit train-only thresholds used to upweight rare wind events in the loss."""
    data_dir = validate_station_data_dir(data_dir)
    config = dict(condition_config)
    residual = np.asarray(
        np.load(data_dir / "train_residual.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    actual = np.asarray(
        np.load(data_dir / "train_actual.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    valid = np.asarray(
        np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")
    ) == 0
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacities = stations.capacity_mw.to_numpy(dtype=np.float64)
    wind_capacity = capacities[wind_indices]
    wind_weight = wind_capacity / wind_capacity.sum()
    wind_valid = valid[:, :, wind_indices]
    complete = wind_valid.all(axis=-1)
    aggregate_residual = np.einsum(
        "nts,s->nt", residual[:, :, wind_indices], wind_weight
    )
    aggregate_actual = np.einsum("nts,s->nt", actual[:, :, wind_indices], wind_weight)

    method = str(config.get("event_weighting_method", "target_extremes_v0"))
    if method == "forecast_mismatch_v1":
        level_quantiles = tuple(
            float(value)
            for value in config.get("event_level_quantiles", [0.90, 0.99])
        )
        ramp_quantiles = tuple(
            float(value)
            for value in config.get(
                "event_ramp_mismatch_quantiles", [0.90, 0.99]
            )
        )
        for name, quantiles in (
            ("event_level_quantiles", level_quantiles),
            ("event_ramp_mismatch_quantiles", ramp_quantiles),
        ):
            if len(quantiles) != 2 or not (
                0 <= quantiles[0] < quantiles[1] <= 1
            ):
                raise ValueError(f"{name} must contain two increasing values")
        ramp_lags = tuple(
            int(value) for value in config.get("event_ramp_lags", [1, 3, 6])
        )
        if not ramp_lags or any(
            lag < 1 or lag >= EXPECTED_HOURS for lag in ramp_lags
        ):
            raise ValueError(f"invalid event_ramp_lags={ramp_lags}")
        aggregate_negative = np.maximum(-aggregate_residual[complete], 0.0)
        aggregate_level_thresholds = np.quantile(
            aggregate_negative, level_quantiles
        )
        aggregate_ramp_thresholds: dict[str, list[float]] = {}
        node_ramp_thresholds: dict[str, list[list[float]]] = {}
        for lag in ramp_lags:
            pair_complete = complete[:, lag:] & complete[:, :-lag]
            aggregate_mismatch = np.abs(
                aggregate_residual[:, lag:] - aggregate_residual[:, :-lag]
            )[pair_complete]
            aggregate_ramp_thresholds[str(lag)] = [
                float(value)
                for value in np.quantile(aggregate_mismatch, ramp_quantiles)
            ]
            station_thresholds = []
            for station_index in wind_indices:
                pair_valid = (
                    valid[:, lag:, station_index]
                    & valid[:, :-lag, station_index]
                )
                mismatch = np.abs(
                    residual[:, lag:, station_index]
                    - residual[:, :-lag, station_index]
                )[pair_valid]
                station_thresholds.append(
                    [
                        float(value)
                        for value in np.quantile(mismatch, ramp_quantiles)
                    ]
                )
            node_ramp_thresholds[str(lag)] = station_thresholds
        node_level_thresholds = []
        for station_index in wind_indices:
            station_negative = np.maximum(
                -residual[:, :, station_index][valid[:, :, station_index]], 0.0
            )
            node_level_thresholds.append(
                [
                    float(value)
                    for value in np.quantile(station_negative, level_quantiles)
                ]
            )
        context_hours = int(config.get("event_context_hours", 3))
        if not 0 <= context_hours <= 12:
            raise ValueError("event_context_hours must be in [0,12]")
        return {
            "method": "train_forecast_mismatch_event_weighting_v1",
            "fit_split": "train",
            "used_for": "training_loss_weight_only",
            "future_actual_used_as_condition": False,
            "applied_to_validation_or_generation": False,
            "formula": (
                "level=max(forecast-actual,0); ramp=abs(delta(actual)-delta(forecast))"
            ),
            "max_weight": float(config.get("event_max_weight", 3.0)),
            "level_quantiles": list(level_quantiles),
            "aggregate_level_thresholds": [
                float(value) for value in aggregate_level_thresholds
            ],
            "node_level_thresholds": node_level_thresholds,
            "ramp_mismatch_quantiles": list(ramp_quantiles),
            "ramp_lags": list(ramp_lags),
            "aggregate_ramp_mismatch_thresholds": aggregate_ramp_thresholds,
            "node_ramp_mismatch_thresholds": node_ramp_thresholds,
            "context_hours": context_hours,
            "use_aggregate_events": bool(
                config.get("event_use_aggregate_events", True)
            ),
            "use_node_events": bool(config.get("event_use_node_events", True)),
            "wind_station_indices": [int(value) for value in wind_indices],
            "wind_capacity_weights": [float(value) for value in wind_weight],
        }

    negative = np.maximum(-aggregate_residual[complete], 0.0)
    negative_quantiles = tuple(
        float(value)
        for value in config.get("event_negative_quantiles", [0.90, 0.99])
    )
    if len(negative_quantiles) != 2 or not (
        0 <= negative_quantiles[0] < negative_quantiles[1] <= 1
    ):
        raise ValueError("event_negative_quantiles must contain two increasing values")
    negative_thresholds = np.quantile(negative, negative_quantiles)

    ramp_lags = tuple(
        int(value) for value in config.get("event_ramp_lags", [1, 3, 6])
    )
    ramp_quantiles = tuple(
        float(value)
        for value in config.get("event_ramp_quantiles", [0.90, 0.99])
    )
    if len(ramp_quantiles) != 2 or not (
        0 <= ramp_quantiles[0] < ramp_quantiles[1] <= 1
    ):
        raise ValueError("event_ramp_quantiles must contain two increasing values")
    ramp_thresholds: dict[str, list[float]] = {}
    for lag in ramp_lags:
        if not 1 <= lag < EXPECTED_HOURS:
            raise ValueError(f"invalid event ramp lag={lag}")
        pair_valid = complete[:, lag:] & complete[:, :-lag]
        magnitude = np.abs(
            aggregate_actual[:, lag:] - aggregate_actual[:, :-lag]
        )[pair_valid]
        ramp_thresholds[str(lag)] = [
            float(value) for value in np.quantile(magnitude, ramp_quantiles)
        ]

    return {
        "method": "train_target_aggregate_wind_event_quantiles",
        "fit_split": "train",
        "used_for": "training_loss_weight_only",
        "future_actual_used_as_condition": False,
        "applied_to_validation_or_generation": False,
        "max_weight": float(config.get("event_max_weight", 3.0)),
        "negative_quantiles": list(negative_quantiles),
        "negative_thresholds": [float(value) for value in negative_thresholds],
        "ramp_quantiles": list(ramp_quantiles),
        "ramp_lags": list(ramp_lags),
        "ramp_thresholds": ramp_thresholds,
        "synchronous_residual_threshold": float(
            config.get("event_synchronous_residual_threshold", 0.25)
        ),
        "synchronous_fraction_start": float(
            config.get("event_synchronous_fraction_start", 0.30)
        ),
        "synchronous_fraction_full": float(
            config.get("event_synchronous_fraction_full", 0.80)
        ),
        "wind_station_indices": [int(value) for value in wind_indices],
        "wind_capacity_weights": [float(value) for value in wind_weight],
    }


def validate_station_event_weighting(
    specification: Mapping[str, object],
) -> dict[str, object]:
    specification = dict(specification)
    if specification.get("fit_split") != "train":
        raise ValueError("event weighting must be fitted on train")
    if bool(specification.get("future_actual_used_as_condition", True)):
        raise ValueError("event targets cannot be exposed as generation conditions")
    if bool(specification.get("applied_to_validation_or_generation", True)):
        raise ValueError("event weighting must be disabled outside training")
    maximum = float(specification.get("max_weight", 0.0))
    if not 1.0 <= maximum <= 5.0:
        raise ValueError("event max_weight must be in [1,5]")
    indices = np.asarray(specification.get("wind_station_indices"), dtype=int)
    weights = np.asarray(specification.get("wind_capacity_weights"), dtype=float)
    if indices.ndim != 1 or weights.shape != indices.shape or weights.sum() <= 0:
        raise ValueError("invalid wind station weights")
    method = str(specification.get("method", ""))
    if method == "train_forecast_mismatch_event_weighting_v1":
        context = int(specification.get("context_hours", -1))
        if not 0 <= context <= 12:
            raise ValueError("invalid forecast-mismatch context hours")
        node_level = np.asarray(
            specification.get("node_level_thresholds"), dtype=float
        )
        if node_level.shape != (len(indices), 2):
            raise ValueError("invalid node forecast-mismatch level thresholds")
        if not bool(specification.get("use_aggregate_events", False)) and not bool(
            specification.get("use_node_events", False)
        ):
            raise ValueError("forecast mismatch weighting has no enabled scope")
    return specification


def fit_station_event_replay(
    data_dir: str | Path,
    condition_config: Mapping[str, object],
) -> dict[str, object]:
    """Build train-only, issue-deduplicated wind event replay targets.

    A 168-hour rolling issuance can expose the same physical wind event in
    several training samples.  This routine first identifies each issue's
    worst valid six-hour aggregate forecast over-estimate, then merges event
    timestamps less than ``merge_gap_hours`` apart.  Only the most severe
    issue in each merged group receives replay weight and x0 event targets.
    """

    data_dir = validate_station_data_dir(data_dir)
    config = dict(condition_config)
    forecast = np.asarray(
        np.load(data_dir / "train_forecast.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    actual = np.asarray(
        np.load(data_dir / "train_actual.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    residual = np.asarray(
        np.load(data_dir / "train_residual.npy", mmap_mode="r"),
        dtype=np.float64,
    )
    valid = np.asarray(
        np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")
    ) == 0
    issues = pd.read_csv(data_dir / "train_issue_dates.csv")
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    if len(issues) != len(forecast):
        raise ValueError("train issue dates do not match train arrays")
    if np.max(np.abs(residual - (actual - forecast))) > 1e-6:
        raise ValueError("event replay requires residual=actual-forecast")

    window = int(config.get("event_replay_window_hours", 6))
    merge_gap = int(config.get("event_replay_merge_gap_hours", 24))
    quantiles = tuple(
        float(value)
        for value in config.get("event_replay_quantiles", [0.80, 0.90])
    )
    replay_weights = tuple(
        float(value)
        for value in config.get("event_replay_weights", [2.0, 4.0])
    )
    if not 1 <= window <= 24:
        raise ValueError("event_replay_window_hours must be in [1,24]")
    if not 0 <= merge_gap <= EXPECTED_HOURS:
        raise ValueError("event_replay_merge_gap_hours must be in [0,168]")
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("event_replay_quantiles must contain two values in (0,1)")
    if (
        len(replay_weights) != 2
        or replay_weights[0] < 1.0
        or replay_weights[1] < replay_weights[0]
        or replay_weights[1] > 5.0
    ):
        raise ValueError("event_replay_weights must be increasing within [1,5]")

    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacities = stations.capacity_mw.to_numpy(dtype=np.float64)
    wind_capacity = capacities[wind_indices]
    wind_weight = wind_capacity / wind_capacity.sum()
    aggregate_residual = np.einsum(
        "nts,s->nt", residual[:, :, wind_indices], wind_weight
    )
    complete = valid[:, :, wind_indices].all(axis=-1)

    severity = np.full(len(forecast), np.nan, dtype=np.float64)
    event_start = np.zeros(len(forecast), dtype=np.int64)
    for sample_index in range(len(forecast)):
        mismatch = -aggregate_residual[sample_index]
        rolling = np.convolve(mismatch, np.ones(window) / window, mode="valid")
        rolling_valid = (
            np.convolve(
                complete[sample_index].astype(np.int64),
                np.ones(window, dtype=np.int64),
                mode="valid",
            )
            == window
        )
        rolling = np.where(rolling_valid, rolling, np.nan)
        if np.any(np.isfinite(rolling)):
            start = int(np.nanargmax(rolling))
            event_start[sample_index] = start
            severity[sample_index] = float(rolling[start])

    finite = severity[np.isfinite(severity)]
    if finite.size != len(forecast):
        raise ValueError("every train issue must contain a valid wind event window")
    thresholds = np.quantile(finite, quantiles)
    target_start_column = "target_start" if "target_start" in issues else "issue_date"
    target_starts = pd.to_datetime(issues[target_start_column])
    event_timestamps = target_starts + pd.to_timedelta(event_start, unit="h")

    selected = np.flatnonzero(severity >= thresholds[0])
    selected = selected[np.argsort(event_timestamps.iloc[selected].to_numpy())]
    groups: list[list[int]] = []
    current: list[int] = []
    previous: pd.Timestamp | None = None
    for sample_index in selected:
        timestamp = pd.Timestamp(event_timestamps.iloc[int(sample_index)])
        if previous is None or timestamp - previous > pd.Timedelta(hours=merge_gap):
            if current:
                groups.append(current)
            current = [int(sample_index)]
        else:
            current.append(int(sample_index))
        previous = timestamp
    if current:
        groups.append(current)

    sample_count = len(forecast)
    active = np.zeros(sample_count, dtype=np.float32)
    starts = np.zeros(sample_count, dtype=np.int64)
    tiers = np.zeros(sample_count, dtype=np.int64)
    sample_weights = np.ones(sample_count, dtype=np.float64)
    sync_weights = np.zeros((sample_count, EXPECTED_STATIONS), dtype=np.float32)
    station_tail_threshold = np.full(EXPECTED_STATIONS, np.nan, dtype=np.float64)
    for station_index in wind_indices:
        station_values = residual[:, :, station_index][valid[:, :, station_index]]
        station_tail_threshold[station_index] = float(
            np.quantile(station_values, 0.10)
        )

    catalog: list[dict[str, object]] = []
    for event_number, members in enumerate(groups, start=1):
        representative = max(members, key=lambda value: severity[value])
        tier_index = int(severity[representative] >= thresholds[1])
        tier = int(round(quantiles[tier_index] * 100))
        start = int(event_start[representative])
        stop = start + window
        station_mean = residual[representative, start:stop][
            :, wind_indices
        ].mean(axis=0)
        synchronous = station_mean <= station_tail_threshold[wind_indices]
        if not np.any(synchronous):
            synchronous[int(np.argmin(station_mean))] = True
        active[representative] = 1.0
        starts[representative] = start
        tiers[representative] = tier
        sample_weights[representative] = replay_weights[tier_index]
        sync_weights[representative, wind_indices] = synchronous.astype(np.float32)
        catalog.append(
            {
                "event_id": f"train_q{tier}_event_{event_number:03d}",
                "tier": tier,
                "threshold": float(thresholds[tier_index]),
                "representative_sample_index": int(representative),
                "representative_issue_date": str(issues.iloc[representative]["issue_date"]),
                "event_timestamp": pd.Timestamp(
                    event_timestamps.iloc[representative]
                ).isoformat(),
                "lead_start": start,
                "lead_end": stop - 1,
                "severity": float(severity[representative]),
                "member_issue_count": int(len(members)),
                "member_sample_indices": [int(value) for value in members],
                "synchronous_station_count": int(np.sum(synchronous)),
                "replay_weight": float(replay_weights[tier_index]),
            }
        )

    total_weight = float(sample_weights.sum())
    expected_event_draws = float(sample_weights[active > 0].sum() / total_weight * sample_count)
    return {
        "method": "train_independent_wind_event_replay_x0_v1",
        "fit_split": "train",
        "used_for": ["training_sampler", "training_x0_event_loss"],
        "future_actual_used_as_condition": False,
        "applied_to_validation_or_generation": False,
        "ordinary_epsilon_loss_reweighted": False,
        "event_definition": "maximum_valid_6h_capacity_weighted_forecast_minus_actual",
        "event_window_hours": window,
        "merge_gap_hours": merge_gap,
        "quantiles": list(quantiles),
        "severity_thresholds": [float(value) for value in thresholds],
        "replay_weights": list(replay_weights),
        "sample_replay_weights": [float(value) for value in sample_weights],
        "sample_event_active": [float(value) for value in active],
        "sample_event_start": [int(value) for value in starts],
        "sample_event_tier": [int(value) for value in tiers],
        "sample_sync_station_weight": sync_weights.tolist(),
        "wind_station_indices": [int(value) for value in wind_indices],
        "wind_capacity_weights": [float(value) for value in wind_weight],
        "wind_station_lower_tail_q10": [
            None if not np.isfinite(value) else float(value)
            for value in station_tail_threshold
        ],
        "independent_event_count": int(len(catalog)),
        "q90_event_count": int(sum(row["tier"] == 90 for row in catalog)),
        "overlapping_issue_count": int(sum(len(row["member_sample_indices"]) for row in catalog)),
        "representative_issue_count": int(active.sum()),
        "expected_event_draws_per_epoch": expected_event_draws,
        "catalog": catalog,
    }


def fit_station_forecast_mismatch_replay(
    data_dir: str | Path,
    condition_config: Mapping[str, object],
) -> dict[str, object]:
    """Fit independent, multi-duration forecast-missed wind ramp events.

    Unlike the sustained-low-output replay above, this target is activated when
    the realised 1/3/6 h aggregate ramp is large but the issued forecast has the
    wrong sign or less than half its magnitude.  Labels are training targets;
    they are never exposed to validation/test generation.
    """

    data_dir = validate_station_data_dir(data_dir)
    config = dict(condition_config)
    forecast = np.asarray(
        np.load(data_dir / "train_forecast.npy", mmap_mode="r"), dtype=np.float64
    )
    actual = np.asarray(
        np.load(data_dir / "train_actual.npy", mmap_mode="r"), dtype=np.float64
    )
    valid = np.asarray(np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")) == 0
    issues = pd.read_csv(data_dir / "train_issue_dates.csv")
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacity = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
    weight = capacity / capacity.sum()
    actual_wind = np.einsum("nts,s->nt", actual[:, :, wind_indices], weight)
    forecast_wind = np.einsum("nts,s->nt", forecast[:, :, wind_indices], weight)
    complete = valid[:, :, wind_indices].all(axis=-1)

    lags = tuple(int(value) for value in config.get("mismatch_ramp_lags", [1, 3, 6]))
    window = int(config.get("mismatch_event_window_hours", 12))
    merge_gap = int(config.get("mismatch_merge_gap_hours", 12))
    quantiles = tuple(
        float(value) for value in config.get("mismatch_replay_quantiles", [0.80, 0.90])
    )
    replay_weights = tuple(
        float(value) for value in config.get("mismatch_replay_weights", [2.0, 4.0])
    )
    magnitude_fraction = float(config.get("mismatch_forecast_magnitude_fraction", 0.5))
    if not lags or any(lag not in {1, 3, 6, 12} for lag in lags):
        raise ValueError("mismatch_ramp_lags must be selected from 1/3/6/12 h")
    if not 3 <= window <= 24 or not 0 <= merge_gap <= 48:
        raise ValueError("invalid mismatch event window/merge gap")
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("mismatch replay requires two increasing quantiles")
    if len(replay_weights) != 2 or not 1 <= replay_weights[0] <= replay_weights[1] <= 5:
        raise ValueError("mismatch replay weights must be increasing within [1,5]")
    if not 0 < magnitude_fraction < 1:
        raise ValueError("mismatch forecast magnitude fraction must be in (0,1)")

    ramp_thresholds: dict[str, float] = {}
    event_score = np.zeros_like(actual_wind)
    event_lag = np.zeros_like(actual_wind, dtype=np.int16)
    for lag in lags:
        actual_ramp = actual_wind[:, lag:] - actual_wind[:, :-lag]
        forecast_ramp = forecast_wind[:, lag:] - forecast_wind[:, :-lag]
        pair_valid = complete[:, lag:] & complete[:, :-lag]
        threshold = float(np.quantile(np.abs(actual_ramp[pair_valid]), 0.90))
        ramp_thresholds[str(lag)] = threshold
        missed = pair_valid & (np.abs(actual_ramp) >= threshold) & (
            (np.sign(actual_ramp) != np.sign(forecast_ramp))
            | (np.abs(forecast_ramp) < magnitude_fraction * np.abs(actual_ramp))
        )
        score = np.where(
            missed,
            np.abs(actual_ramp - forecast_ramp) / max(threshold, 1e-6),
            0.0,
        )
        replace = score > event_score[:, lag:]
        event_score[:, lag:] = np.maximum(event_score[:, lag:], score)
        event_lag[:, lag:] = np.where(replace, lag, event_lag[:, lag:])

    severity = event_score.max(axis=1)
    center = event_score.argmax(axis=1)
    positive = severity[severity > 0]
    if positive.size < 4:
        raise ValueError("too few training forecast-mismatch events")
    thresholds = np.quantile(positive, quantiles)
    target_start_column = "target_start" if "target_start" in issues else "issue_date"
    target_starts = pd.to_datetime(issues[target_start_column])
    timestamps = target_starts + pd.to_timedelta(center, unit="h")
    selected = np.flatnonzero(severity >= thresholds[0])
    selected = selected[np.argsort(timestamps.iloc[selected].to_numpy())]
    groups: list[list[int]] = []
    for sample_index in selected:
        stamp = pd.Timestamp(timestamps.iloc[int(sample_index)])
        if not groups:
            groups.append([int(sample_index)])
            continue
        previous = pd.Timestamp(timestamps.iloc[groups[-1][-1]])
        if stamp - previous <= pd.Timedelta(hours=merge_gap):
            groups[-1].append(int(sample_index))
        else:
            groups.append([int(sample_index)])

    count = len(forecast)
    active = np.zeros(count, dtype=np.float32)
    starts = np.zeros(count, dtype=np.int64)
    tiers = np.zeros(count, dtype=np.int64)
    sample_weights = np.ones(count, dtype=np.float64)
    sync_weights = np.zeros((count, EXPECTED_STATIONS), dtype=np.float32)
    catalog: list[dict[str, object]] = []
    half_left = window // 3
    for event_number, members in enumerate(groups, start=1):
        representative = max(members, key=lambda value: severity[value])
        event_center = int(center[representative])
        start = min(max(event_center - half_left, 0), EXPECTED_HOURS - window)
        lag = int(event_lag[representative, event_center])
        tier_index = int(severity[representative] >= thresholds[1])
        tier = int(round(quantiles[tier_index] * 100))
        station_actual_ramp = (
            actual[representative, event_center, wind_indices]
            - actual[representative, event_center - lag, wind_indices]
        )
        station_forecast_ramp = (
            forecast[representative, event_center, wind_indices]
            - forecast[representative, event_center - lag, wind_indices]
        )
        station_mismatch = np.abs(station_actual_ramp - station_forecast_ramp)
        sync = station_mismatch >= np.quantile(station_mismatch, 0.50)
        active[representative] = 1.0
        starts[representative] = start
        tiers[representative] = tier
        sample_weights[representative] = replay_weights[tier_index]
        sync_weights[representative, wind_indices] = sync.astype(np.float32)
        catalog.append(
            {
                "event_id": f"train_mismatch_q{tier}_{event_number:03d}",
                "tier": tier,
                "representative_sample_index": int(representative),
                "representative_issue_date": str(issues.iloc[representative]["issue_date"]),
                "event_timestamp": pd.Timestamp(timestamps.iloc[representative]).isoformat(),
                "lead_center": event_center,
                "lead_start": start,
                "lead_end": start + window - 1,
                "ramp_lag_hours": lag,
                "severity": float(severity[representative]),
                "member_sample_indices": members,
                "synchronous_station_count": int(sync.sum()),
                "replay_weight": float(replay_weights[tier_index]),
            }
        )
    total_weight = float(sample_weights.sum())
    return {
        "method": "train_independent_forecast_missed_ramp_replay_v1",
        "fit_split": "train",
        "used_for": ["training_sampler", "mismatch_expert", "training_x0_event_loss"],
        "future_actual_used_as_condition": False,
        "applied_to_validation_or_generation": False,
        "ordinary_epsilon_loss_reweighted": False,
        "event_definition": "large_actual_1_3_6h_ramp_wrong_sign_or_under_half_in_issued_forecast",
        "event_window_hours": window,
        "merge_gap_hours": merge_gap,
        "ramp_lags": list(lags),
        "actual_ramp_abs_q90_thresholds": ramp_thresholds,
        "forecast_magnitude_fraction": magnitude_fraction,
        "quantiles": list(quantiles),
        "severity_thresholds": [float(value) for value in thresholds],
        "replay_weights": list(replay_weights),
        "sample_replay_weights": sample_weights.tolist(),
        "sample_event_active": active.tolist(),
        "sample_event_start": starts.tolist(),
        "sample_event_tier": tiers.tolist(),
        "sample_sync_station_weight": sync_weights.tolist(),
        "wind_station_indices": wind_indices.tolist(),
        "wind_capacity_weights": weight.tolist(),
        "independent_event_count": int(len(catalog)),
        "q90_event_count": int(sum(row["tier"] == 90 for row in catalog)),
        "overlapping_issue_count": int(sum(len(row["member_sample_indices"]) for row in catalog)),
        "representative_issue_count": int(active.sum()),
        "expected_event_draws_per_epoch": float(
            sample_weights[active > 0].sum() / total_weight * count
        ),
        "catalog": catalog,
    }


def fit_station_unified_event_replay(
    data_dir: str | Path,
    condition_config: Mapping[str, object],
) -> dict[str, object]:
    """Unify sustained drops and missed 1/3/6 h ramps for one event expert."""

    deep = fit_station_event_replay(data_dir, condition_config)
    mismatch = fit_station_forecast_mismatch_replay(data_dir, condition_config)
    if int(deep["event_window_hours"]) != int(mismatch["event_window_hours"]):
        raise ValueError("unified replay requires matching deep/mismatch windows")
    deep_active = np.asarray(deep["sample_event_active"], dtype=np.float32)
    mismatch_active = np.asarray(mismatch["sample_event_active"], dtype=np.float32)
    deep_tier = np.asarray(deep["sample_event_tier"], dtype=np.int64)
    mismatch_tier = np.asarray(mismatch["sample_event_tier"], dtype=np.int64)
    choose_mismatch = (mismatch_active > 0) & (
        (deep_active == 0) | (mismatch_tier > deep_tier)
    )
    active = np.maximum(deep_active, mismatch_active)
    starts = np.where(
        choose_mismatch,
        np.asarray(mismatch["sample_event_start"], dtype=np.int64),
        np.asarray(deep["sample_event_start"], dtype=np.int64),
    )
    tiers = np.where(choose_mismatch, mismatch_tier, deep_tier)
    weights = np.maximum(
        np.asarray(deep["sample_replay_weights"], dtype=np.float64),
        np.asarray(mismatch["sample_replay_weights"], dtype=np.float64),
    )
    deep_sync = np.asarray(deep["sample_sync_station_weight"], dtype=np.float32)
    mismatch_sync = np.asarray(
        mismatch["sample_sync_station_weight"], dtype=np.float32
    )
    sync = np.where(choose_mismatch[:, None], mismatch_sync, deep_sync)
    count = len(active)
    total_weight = float(weights.sum())
    return {
        "method": "train_unified_wind_event_replay_v1",
        "fit_split": "train",
        "used_for": [
            "training_sampler",
            "unified_event_expert",
            "training_x0_event_loss",
            "event_selector_supervision",
        ],
        "future_actual_used_as_condition": False,
        "applied_to_validation_or_generation": False,
        "ordinary_epsilon_loss_reweighted": False,
        "event_definition": "union_of_sustained_forecast_overestimate_and_forecast_missed_1_3_6h_ramps",
        "event_window_hours": int(deep["event_window_hours"]),
        "sample_replay_weights": weights.tolist(),
        "sample_event_active": active.tolist(),
        "sample_event_start": starts.tolist(),
        "sample_event_tier": tiers.tolist(),
        "sample_sync_station_weight": sync.tolist(),
        "wind_station_indices": deep["wind_station_indices"],
        "wind_capacity_weights": deep["wind_capacity_weights"],
        "independent_event_count": int(active.sum()),
        "sustained_drop_representative_count": int(deep_active.sum()),
        "forecast_mismatch_representative_count": int(mismatch_active.sum()),
        "overlap_representative_count": int(
            np.sum((deep_active > 0) & (mismatch_active > 0))
        ),
        "expected_event_draws_per_epoch": float(
            weights[active > 0].sum() / total_weight * count
        ),
        "deep_replay": deep,
        "mismatch_replay": mismatch,
    }


def validate_station_event_replay(
    specification: Mapping[str, object],
    sample_count: int | None = None,
) -> dict[str, object]:
    specification = dict(specification)
    if specification.get("method") not in {
        "train_independent_wind_event_replay_x0_v1",
        "train_independent_forecast_missed_ramp_replay_v1",
        "train_unified_wind_event_replay_v1",
    }:
        raise ValueError("unsupported event replay method")
    if specification.get("fit_split") != "train":
        raise ValueError("event replay must be fitted on train")
    if bool(specification.get("future_actual_used_as_condition", True)):
        raise ValueError("event replay targets cannot be generation conditions")
    if bool(specification.get("applied_to_validation_or_generation", True)):
        raise ValueError("event replay must be disabled outside training")
    weights = np.asarray(specification.get("sample_replay_weights"), dtype=float)
    active = np.asarray(specification.get("sample_event_active"), dtype=float)
    starts = np.asarray(specification.get("sample_event_start"), dtype=int)
    sync = np.asarray(specification.get("sample_sync_station_weight"), dtype=float)
    expected_count = int(sample_count) if sample_count is not None else len(weights)
    if weights.shape != (expected_count,) or active.shape != (expected_count,):
        raise ValueError("event replay sample arrays do not match train sample count")
    if starts.shape != (expected_count,) or sync.shape != (expected_count, EXPECTED_STATIONS):
        raise ValueError("event replay target arrays have invalid shapes")
    if np.any(weights < 1.0) or np.any(weights > 5.0):
        raise ValueError("event replay sample weights must stay in [1,5]")
    window = int(specification.get("event_window_hours", 0))
    if np.any(starts[active > 0] < 0) or np.any(
        starts[active > 0] + window > EXPECTED_HOURS
    ):
        raise ValueError("event replay windows leave the 168-hour target")
    return specification


def _dilate_event_severity(values: np.ndarray, radius: int) -> np.ndarray:
    """Max-pool event severity over a symmetric temporal context."""
    output = np.asarray(values, dtype=np.float64).copy()
    source = output.copy()
    for shift in range(1, int(radius) + 1):
        output[..., shift:] = np.maximum(output[..., shift:], source[..., :-shift])
        output[..., :-shift] = np.maximum(output[..., :-shift], source[..., shift:])
    return output


def build_station_event_loss_weights(
    actual: np.ndarray,
    residual: np.ndarray,
    valid_mask: np.ndarray,
    specification: Mapping[str, object] | None,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return [S,L] epsilon weights and [L] aggregate-event weights."""
    station_count, length = residual.shape
    loss_weight = np.ones((station_count, length), dtype=np.float32)
    time_weight = np.ones(length, dtype=np.float32)
    if specification is None or split != "train":
        return loss_weight, time_weight
    spec = validate_station_event_weighting(specification)
    indices = np.asarray(spec["wind_station_indices"], dtype=int)
    capacity_weight = np.asarray(spec["wind_capacity_weights"], dtype=np.float64)
    aggregate_residual = np.einsum(
        "st,s->t", residual[indices], capacity_weight
    )
    aggregate_actual = np.einsum("st,s->t", actual[indices], capacity_weight)
    complete = valid_mask[indices].min(axis=0) > 0.5

    def severity(value: np.ndarray, thresholds: list[float]) -> np.ndarray:
        start, full = (float(thresholds[0]), float(thresholds[1]))
        denominator = max(full - start, 1e-6)
        return np.clip((value - start) / denominator, 0.0, 1.0)

    if spec.get("method") == "train_forecast_mismatch_event_weighting_v1":
        aggregate_event = severity(
            np.maximum(-aggregate_residual, 0.0),
            spec["aggregate_level_thresholds"],
        )
        node_event = np.zeros((len(indices), length), dtype=np.float64)
        node_level_thresholds = np.asarray(
            spec["node_level_thresholds"], dtype=np.float64
        )
        for local_index, station_index in enumerate(indices):
            node_event[local_index] = severity(
                np.maximum(-residual[station_index], 0.0),
                node_level_thresholds[local_index],
            )
        for lag in (int(value) for value in spec["ramp_lags"]):
            aggregate_mismatch = np.zeros(length, dtype=np.float64)
            aggregate_mismatch[lag:] = np.abs(
                aggregate_residual[lag:] - aggregate_residual[:-lag]
            )
            aggregate_event = np.maximum(
                aggregate_event,
                severity(
                    aggregate_mismatch,
                    spec["aggregate_ramp_mismatch_thresholds"][str(lag)],
                ),
            )
            node_thresholds = np.asarray(
                spec["node_ramp_mismatch_thresholds"][str(lag)],
                dtype=np.float64,
            )
            for local_index, station_index in enumerate(indices):
                mismatch = np.zeros(length, dtype=np.float64)
                mismatch[lag:] = np.abs(
                    residual[station_index, lag:]
                    - residual[station_index, :-lag]
                )
                node_event[local_index] = np.maximum(
                    node_event[local_index],
                    severity(mismatch, node_thresholds[local_index]),
                )
        aggregate_event = _dilate_event_severity(
            aggregate_event, int(spec["context_hours"])
        )
        node_event = _dilate_event_severity(
            node_event, int(spec["context_hours"])
        )
        aggregate_event *= complete
        node_event *= valid_mask[indices]
        maximum = float(spec["max_weight"])
        if bool(spec["use_aggregate_events"]):
            time_weight = (1.0 + (maximum - 1.0) * aggregate_event).astype(
                np.float32
            )
        if not bool(spec["use_node_events"]):
            node_event.fill(0.0)
        combined = node_event
        if bool(spec["use_aggregate_events"]):
            combined = np.maximum(combined, aggregate_event[None])
        loss_weight[indices] = 1.0 + (maximum - 1.0) * combined
        denominator = float((loss_weight * valid_mask).sum())
        numerator = float(valid_mask.sum())
        if denominator > 0:
            loss_weight *= numerator / denominator
        return loss_weight.astype(np.float32), time_weight.astype(np.float32)

    event = severity(
        np.maximum(-aggregate_residual, 0.0), spec["negative_thresholds"]
    )
    for lag in (int(value) for value in spec["ramp_lags"]):
        ramp = np.zeros(length, dtype=np.float64)
        ramp[lag:] = np.abs(aggregate_actual[lag:] - aggregate_actual[:-lag])
        event = np.maximum(
            event, severity(ramp, spec["ramp_thresholds"][str(lag)])
        )
    synchronous = np.mean(
        residual[indices]
        <= -float(spec["synchronous_residual_threshold"]),
        axis=0,
    )
    sync_start = float(spec["synchronous_fraction_start"])
    sync_full = float(spec["synchronous_fraction_full"])
    event = np.maximum(
        event,
        np.clip(
            (synchronous - sync_start) / max(sync_full - sync_start, 1e-6),
            0.0,
            1.0,
        ),
    )
    event *= complete
    time_weight = (
        1.0 + (float(spec["max_weight"]) - 1.0) * event
    ).astype(np.float32)
    loss_weight[indices] = time_weight[None]
    # Keep the average valid epsilon contribution at one, changing emphasis
    # rather than the optimizer's effective learning rate.
    denominator = float((loss_weight * valid_mask).sum())
    numerator = float(valid_mask.sum())
    if denominator > 0:
        loss_weight *= numerator / denominator
    return loss_weight, time_weight


class StationForecastDataset(Dataset):
    """One item is one forecast issuance with 24 stations and 168 lead hours."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        residual_scale: Mapping[str, object],
        condition_config: Mapping[str, object] | None = None,
        state_thresholds: Mapping[str, object] | None = None,
        event_weighting: Mapping[str, object] | None = None,
        event_replay: Mapping[str, object] | None = None,
        jstd_targets: JSTDTargetArrays | None = None,
        retrieval_arrays: object | None = None,
        forecast_trust_arrays: object | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split={split!r}")
        self.data_dir = validate_station_data_dir(data_dir)
        self.split = split
        self.forecast = np.load(self.data_dir / f"{split}_forecast.npy")
        self.actual = np.load(self.data_dir / f"{split}_actual.npy")
        self.residual = np.load(self.data_dir / f"{split}_residual.npy")
        self.time_mark = np.load(self.data_dir / f"{split}_time_mark.npy")
        self.lead_mark = np.load(self.data_dir / f"{split}_lead_mark.npy")
        self.fill_mask = np.load(self.data_dir / f"{split}_fill_mask.npy")
        self.residual_scale = dict(residual_scale)
        self.scale = validate_residual_scale(self.residual_scale)
        self.condition_config = dict(condition_config or {})
        self.event_weighting = (
            validate_station_event_weighting(event_weighting)
            if event_weighting is not None
            else None
        )
        # The train-fitted object may be passed to construct all loaders, but
        # labels and sampler weights are never attached outside train.
        self.event_replay = (
            validate_station_event_replay(event_replay, len(self.forecast))
            if event_replay is not None and split == "train"
            else None
        )
        self.jstd_targets = jstd_targets if split in {"train", "val"} else None
        if self.jstd_targets is not None:
            if self.jstd_targets.split != split:
                raise ValueError("JSTD target split does not match dataset split")
            expected_n = len(self.forecast)
            if self.jstd_targets.event_active.shape != (expected_n,):
                raise ValueError("JSTD event_active must be [N]")
            if self.jstd_targets.time_support.shape != (expected_n, EXPECTED_HOURS):
                raise ValueError("JSTD time_support must be [N,168]")
            if self.jstd_targets.station_support.shape != (
                expected_n,
                EXPECTED_STATIONS,
                EXPECTED_HOURS,
            ):
                raise ValueError("JSTD station_support must be [N,24,168]")
        self.retrieval_arrays = retrieval_arrays
        if retrieval_arrays is not None:
            expected_queries = len(self.forecast)
            for name in ("residual", "distance", "prior_weight", "train_index"):
                value = np.asarray(getattr(retrieval_arrays, name))
                if value.shape[0] != expected_queries:
                    raise ValueError(
                        f"retrieval {name} query count {value.shape[0]} "
                        f"does not match {split}={expected_queries}"
                    )
            for name in (
                "time_mask",
                "event_type",
                "duration",
                "target_start",
                "source_start",
            ):
                if hasattr(retrieval_arrays, name):
                    value = np.asarray(getattr(retrieval_arrays, name))
                    if value.shape[0] != expected_queries:
                        raise ValueError(
                            f"retrieval {name} query count {value.shape[0]} "
                            f"does not match {split}={expected_queries}"
                        )
        self.forecast_trust_arrays = forecast_trust_arrays
        if forecast_trust_arrays is not None:
            expected = (len(self.forecast), EXPECTED_HOURS, EXPECTED_STATIONS)
            for name in ("center", "dispersion"):
                value = np.asarray(getattr(forecast_trust_arrays, name))
                if value.shape != expected:
                    raise ValueError(
                        f"forecast trust {name} expected {expected}, got {value.shape}"
                    )
        self.use_state_encoder = bool(
            self.condition_config.get("use_state_encoder", False)
        )
        self.ramp_lags = tuple(
            int(value)
            for value in self.condition_config.get("forecast_ramp_lags", [1, 3, 6])
        )
        if not self.ramp_lags or any(
            lag <= 0 or lag >= EXPECTED_HOURS for lag in self.ramp_lags
        ):
            raise ValueError(f"invalid forecast_ramp_lags={self.ramp_lags}")
        self.recent_error_hours = int(
            self.condition_config.get("recent_error_hours", 24)
        )
        if not 1 <= self.recent_error_hours <= EXPECTED_HOURS:
            raise ValueError("recent_error_hours must be between 1 and 168")
        self.state_thresholds = None
        self.state_daylight = None
        self.state_ramp_lags = tuple(
            int(value)
            for value in self.condition_config.get("state_ramp_lags", [3, 6])
        )
        self.state_clip = float(self.condition_config.get("state_clip", 3.0))
        if self.use_state_encoder:
            if state_thresholds is None:
                raise ValueError("state_thresholds are required when use_state_encoder=true")
            self.state_thresholds = validate_station_state_thresholds(state_thresholds)
            if tuple(self.state_thresholds["ramp_lags"]) != self.state_ramp_lags:
                raise ValueError("state config ramp lags do not match fitted thresholds")
            self.state_daylight, _ = build_station_daylight_mask(self.data_dir, split)
        issue_frame = pd.read_csv(self.data_dir / f"{split}_issue_dates.csv")
        if len(issue_frame) != len(self.forecast):
            raise ValueError("issue date count does not match forecast sample count")
        issue_days = pd.to_datetime(issue_frame["issue_date"]).dt.normalize()
        lookup = {timestamp: index for index, timestamp in enumerate(issue_days)}
        self.previous_issue_index = np.asarray(
            [lookup.get(timestamp - pd.Timedelta(days=1), -1) for timestamp in issue_days],
            dtype=np.int64,
        )
        self.condition_audit = {
            "split": split,
            "sample_count": int(len(self.forecast)),
            "previous_issue_available_count": int(
                np.sum(self.previous_issue_index >= 0)
            ),
            "revision_overlap_hours": 144,
            "recent_error_hours": self.recent_error_hours,
            "forecast_ramp_lags": list(self.ramp_lags),
            "use_state_encoder": self.use_state_encoder,
            "state_feature_names": [
                "low_output_severity",
                "high_output_severity",
                "ramp_up_severity",
                "ramp_down_severity",
            ] if self.use_state_encoder else [],
            "state_ramp_lags": list(self.state_ramp_lags) if self.use_state_encoder else [],
            "state_source": "current_issued_forecast_only" if self.use_state_encoder else None,
            "residual_scaling_method": str(
                self.residual_scale.get("method", "per_station_std")
            ),
            "future_actual_used_as_condition": False,
            "event_weighting_enabled": self.event_weighting is not None,
            "event_weighting_applied": bool(
                self.event_weighting is not None and split == "train"
            ),
            "event_weighting_uses_target_as_condition": False,
            "event_replay_enabled": event_replay is not None,
            "event_replay_applied": bool(event_replay is not None and split == "train"),
            "event_replay_uses_target_as_condition": False,
            "jstd_targets_enabled": self.jstd_targets is not None,
            "jstd_targets_used_as_condition": False,
            "event_replay_independent_event_count": (
                int(event_replay["independent_event_count"])
                if event_replay is not None and split == "train"
                else 0
            ),
            "historical_retrieval_enabled": retrieval_arrays is not None,
            "historical_retrieval": (
                dict(retrieval_arrays.audit) if retrieval_arrays is not None else None
            ),
            "forecast_trust_enabled": forecast_trust_arrays is not None,
            "forecast_trust": (
                dict(forecast_trust_arrays.audit)
                if forecast_trust_arrays is not None
                else None
            ),
        }
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        expected = (len(self.forecast), EXPECTED_HOURS, EXPECTED_STATIONS)
        for name in ["forecast", "actual", "residual", "fill_mask"]:
            value = getattr(self, name)
            if value.shape != expected:
                raise ValueError(f"{self.split}_{name} expected {expected}, got {value.shape}")
        if self.time_mark.shape != (len(self.forecast), EXPECTED_HOURS, 8):
            raise ValueError(f"invalid {self.split}_time_mark shape {self.time_mark.shape}")
        if self.lead_mark.shape != (len(self.forecast), EXPECTED_HOURS, 2):
            raise ValueError(f"invalid {self.split}_lead_mark shape {self.lead_mark.shape}")
        residual_error = np.max(
            np.abs(
                np.asarray(self.residual, dtype=np.float32)
                - (
                    np.asarray(self.actual, dtype=np.float32)
                    - np.asarray(self.forecast, dtype=np.float32)
                )
            )
        )
        if residual_error > 1e-6:
            raise ValueError(
                "station residual must equal actual - forecast; "
                f"max error={residual_error}"
            )

    def __len__(self) -> int:
        return int(self.forecast.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        forecast = np.asarray(self.forecast[index], dtype=np.float32).T.copy()
        actual = np.asarray(self.actual[index], dtype=np.float32).T.copy()
        residual = np.asarray(self.residual[index], dtype=np.float32).T.copy()
        valid_mask = 1.0 - np.asarray(
            self.fill_mask[index], dtype=np.float32
        ).T.copy()
        forecast_ramps = np.zeros(
            (EXPECTED_STATIONS, len(self.ramp_lags), EXPECTED_HOURS),
            dtype=np.float32,
        )
        for channel, lag in enumerate(self.ramp_lags):
            forecast_ramps[:, channel, lag:] = (
                forecast[:, lag:] - forecast[:, :-lag]
            )

        node_state = np.zeros((EXPECTED_STATIONS, 4, EXPECTED_HOURS), dtype=np.float32)
        if self.use_state_encoder:
            thresholds = self.state_thresholds
            low = np.asarray(thresholds["low_threshold"], dtype=np.float32)[:, None]
            high = np.asarray(thresholds["high_threshold"], dtype=np.float32)[:, None]
            low_scale = np.maximum(low, float(thresholds["epsilon"]))
            high_scale = np.maximum(1.0 - high, float(thresholds["epsilon"]))
            node_state[:, 0] = np.maximum(0.0, (low - forecast) / low_scale)
            node_state[:, 1] = np.maximum(0.0, (forecast - high) / high_scale)
            daylight = np.asarray(self.state_daylight[index], dtype=bool).T
            node_state[:, :2] *= daylight[:, None, :]
            for lag in self.state_ramp_lags:
                ramp = forecast[:, lag:] - forecast[:, :-lag]
                scale = np.asarray(
                    thresholds["ramp_abs_scale"][str(lag)], dtype=np.float32
                )[:, None]
                valid = daylight[:, lag:] & daylight[:, :-lag]
                node_state[:, 2, lag:] = np.maximum(
                    node_state[:, 2, lag:],
                    np.where(valid, np.maximum(ramp, 0.0) / scale, 0.0),
                )
                node_state[:, 3, lag:] = np.maximum(
                    node_state[:, 3, lag:],
                    np.where(valid, np.maximum(-ramp, 0.0) / scale, 0.0),
                )
            np.clip(node_state, 0.0, self.state_clip, out=node_state)

        forecast_revision = np.zeros_like(forecast)
        revision_mask = np.zeros_like(forecast)
        recent_error = np.zeros(
            (EXPECTED_STATIONS, self.recent_error_hours), dtype=np.float32
        )
        recent_error_mask = np.zeros((EXPECTED_STATIONS, 1), dtype=np.float32)
        previous_index = int(self.previous_issue_index[index])
        if previous_index >= 0:
            overlap = EXPECTED_HOURS - 24
            previous_forecast = np.asarray(
                self.forecast[previous_index], dtype=np.float32
            ).T
            forecast_revision[:, :overlap] = (
                forecast[:, :overlap] - previous_forecast[:, 24:]
            )
            revision_mask[:, :overlap] = 1.0
            previous_residual = np.asarray(
                self.residual[previous_index], dtype=np.float32
            ).T
            recent_error[:] = previous_residual[:, : self.recent_error_hours]
            recent_error_mask[:] = 1.0
        scale_tensor = build_station_residual_scale_tensor(
            self.residual_scale,
            forecast,
            forecast_revision=forecast_revision,
            revision_mask=revision_mask,
        )
        target = residual / scale_tensor
        jstd_slow_target = _same_length_average_numpy(target, 12)
        jstd_fast_target = target - jstd_slow_target
        jstd_slow24_target = _same_length_average_numpy(target, 24)
        historical_center = (
            np.asarray(
                self.forecast_trust_arrays.center[index], dtype=np.float32
            ).T.copy()
            if self.forecast_trust_arrays is not None
            else forecast.copy()
        )
        historical_dispersion = (
            np.asarray(
                self.forecast_trust_arrays.dispersion[index], dtype=np.float32
            ).T.copy()
            if self.forecast_trust_arrays is not None
            else np.zeros_like(forecast)
        )
        retrieval_residual = (
            np.asarray(self.retrieval_arrays.residual[index], dtype=np.float32).copy()
            if self.retrieval_arrays is not None
            else np.zeros((1, EXPECTED_STATIONS, EXPECTED_HOURS), dtype=np.float32)
        )
        # Historical values are stored in physical residual units.  The event
        # expert denoises the same normalized target as the body model, so every
        # candidate is expressed using the query issue's causal residual scale.
        if self.retrieval_arrays is not None and hasattr(
            self.retrieval_arrays, "time_mask"
        ):
            retrieval_residual /= scale_tensor[None, :, :]
        loss_weight, event_time_weight = build_station_event_loss_weights(
            actual,
            residual,
            valid_mask,
            self.event_weighting,
            self.split,
        )
        event_active = np.float32(0.0)
        event_replay_weight = np.float32(1.0)
        event_start = np.int64(0)
        event_window_mask = np.zeros(EXPECTED_HOURS, dtype=np.float32)
        event_sync_station_weight = np.zeros(EXPECTED_STATIONS, dtype=np.float32)
        if self.event_replay is not None:
            event_replay_weight = np.float32(
                self.event_replay["sample_replay_weights"][index]
            )
            event_active = np.float32(
                self.event_replay["sample_event_active"][index]
            )
            event_start = np.int64(
                self.event_replay["sample_event_start"][index]
            )
            if event_active > 0:
                stop = int(event_start) + int(
                    self.event_replay["event_window_hours"]
                )
                event_window_mask[int(event_start):stop] = 1.0
                event_sync_station_weight[:] = np.asarray(
                    self.event_replay["sample_sync_station_weight"][index],
                    dtype=np.float32,
                )
        jstd_event_active = np.float32(0.0)
        jstd_event_time_support = np.zeros(EXPECTED_HOURS, dtype=np.float32)
        jstd_event_station_support = np.zeros(
            (EXPECTED_STATIONS, EXPECTED_HOURS), dtype=np.float32
        )
        jstd_sample_weight = np.float32(1.0)
        if self.jstd_targets is not None:
            jstd_event_active = np.float32(self.jstd_targets.event_active[index])
            jstd_event_time_support[:] = self.jstd_targets.time_support[index]
            jstd_event_station_support[:] = self.jstd_targets.station_support[index]
            jstd_sample_weight = np.float32(self.jstd_targets.sample_weights[index])
        return {
            "sample_index": torch.tensor(index, dtype=torch.long),
            "forecast": torch.from_numpy(forecast),
            "actual": torch.from_numpy(actual),
            "residual": torch.from_numpy(residual),
            "residual_target": torch.from_numpy(target),
            "residual_scale": torch.from_numpy(scale_tensor),
            "historical_center": torch.from_numpy(historical_center),
            "historical_dispersion": torch.from_numpy(historical_dispersion),
            "calendar": torch.from_numpy(
                np.asarray(self.time_mark[index], dtype=np.float32).T.copy()
            ),
            "lead": torch.from_numpy(
                np.asarray(self.lead_mark[index], dtype=np.float32).T.copy()
            ),
            "valid_mask": torch.from_numpy(valid_mask),
            "forecast_ramps": torch.from_numpy(forecast_ramps),
            "forecast_revision": torch.from_numpy(forecast_revision),
            "revision_mask": torch.from_numpy(revision_mask),
            "recent_error": torch.from_numpy(recent_error),
            "recent_error_mask": torch.from_numpy(recent_error_mask),
            "node_state": torch.from_numpy(node_state),
            "loss_weight": torch.from_numpy(loss_weight),
            "event_time_weight": torch.from_numpy(event_time_weight),
            "event_active": torch.tensor(event_active, dtype=torch.float32),
            "event_replay_weight": torch.tensor(
                event_replay_weight, dtype=torch.float32
            ),
            "event_start": torch.tensor(event_start, dtype=torch.long),
            "event_window_mask": torch.from_numpy(event_window_mask),
            "event_sync_station_weight": torch.from_numpy(
                event_sync_station_weight
            ),
            "jstd_event_active": torch.tensor(
                jstd_event_active, dtype=torch.float32
            ),
            "jstd_event_time_support": torch.from_numpy(
                jstd_event_time_support
            ),
            "jstd_event_station_support": torch.from_numpy(
                jstd_event_station_support
            ),
            "jstd_sample_weight": torch.tensor(
                jstd_sample_weight, dtype=torch.float32
            ),
            "jstd_slow_target": torch.from_numpy(jstd_slow_target),
            "jstd_fast_target": torch.from_numpy(jstd_fast_target),
            "jstd_slow24_target": torch.from_numpy(jstd_slow24_target),
            "retrieval_residual": torch.from_numpy(
                retrieval_residual
            ),
            "retrieval_distance": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.distance[index]
                    if self.retrieval_arrays is not None
                    else np.zeros(1),
                    dtype=np.float32,
                ).copy()
            ),
            "retrieval_prior_weight": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.prior_weight[index]
                    if self.retrieval_arrays is not None
                    else np.ones(1),
                    dtype=np.float32,
                ).copy()
            ),
            "retrieval_train_index": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.train_index[index]
                    if self.retrieval_arrays is not None
                    else np.full(1, -1),
                    dtype=np.int64,
                ).copy()
            ),
            "retrieval_time_mask": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.time_mask[index]
                    if self.retrieval_arrays is not None
                    and hasattr(self.retrieval_arrays, "time_mask")
                    else np.zeros((1, EXPECTED_HOURS)),
                    dtype=np.float32,
                ).copy()
            ),
            "retrieval_event_type": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.event_type[index]
                    if self.retrieval_arrays is not None
                    and hasattr(self.retrieval_arrays, "event_type")
                    else np.full(1, -1),
                    dtype=np.int64,
                ).copy()
            ),
            "retrieval_duration": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.duration[index]
                    if self.retrieval_arrays is not None
                    and hasattr(self.retrieval_arrays, "duration")
                    else np.zeros(1),
                    dtype=np.int64,
                ).copy()
            ),
            "retrieval_target_start": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.target_start[index]
                    if self.retrieval_arrays is not None
                    and hasattr(self.retrieval_arrays, "target_start")
                    else np.zeros(1),
                    dtype=np.int64,
                ).copy()
            ),
            "retrieval_source_start": torch.from_numpy(
                np.asarray(
                    self.retrieval_arrays.source_start[index]
                    if self.retrieval_arrays is not None
                    and hasattr(self.retrieval_arrays, "source_start")
                    else np.zeros(1),
                    dtype=np.int64,
                ).copy()
            ),
        }


def load_station_static_data(data_dir: str | Path) -> dict[str, torch.Tensor]:
    data_dir = validate_station_data_dir(data_dir)
    features = np.load(data_dir / "station_features.npy").astype(np.float32)
    adjacency = np.load(data_dir / "station_adjacency.npy").astype(np.float32)
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    capacities = stations["capacity_mw"].to_numpy(dtype=np.float32)
    if features.shape != (EXPECTED_STATIONS, 5):
        raise ValueError(f"station_features expected (24,5), got {features.shape}")
    if adjacency.shape != (EXPECTED_STATIONS, EXPECTED_STATIONS):
        raise ValueError(f"station_adjacency expected (24,24), got {adjacency.shape}")
    if not np.allclose(adjacency, adjacency.T, atol=1e-6):
        raise ValueError("station adjacency must be symmetric")
    if not np.allclose(features[:, :2].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("station wind/solar features must be one-hot")
    if capacities.shape != (EXPECTED_STATIONS,) or np.any(capacities <= 0):
        raise ValueError("station capacities must be positive and match station order")
    return {
        "station_features": torch.from_numpy(features),
        "station_adjacency": torch.from_numpy(adjacency),
        "station_capacities": torch.from_numpy(capacities),
    }


def build_station_daylight_mask(
    data_dir: str | Path,
    split: str,
    elevation_threshold_deg: float = -0.833,
    timestamp_offset_minutes: float = 30.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build [N,168,24] station-specific daylight without using power values."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split={split!r}")
    data_dir = validate_station_data_dir(data_dir)
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    issues = pd.read_csv(data_dir / f"{split}_issue_dates.csv")
    mask = np.ones((len(issues), EXPECTED_HOURS, EXPECTED_STATIONS), dtype=bool)
    china_timezone = timezone(timedelta(hours=8))
    solar_indices = stations.index[stations.data_type.eq("solar")].to_numpy()
    for issue_index, target_start in enumerate(issues.target_start):
        start = datetime.fromisoformat(str(target_start)).replace(tzinfo=china_timezone)
        timestamps = [
            start + timedelta(hours=lead, minutes=float(timestamp_offset_minutes))
            for lead in range(EXPECTED_HOURS)
        ]
        for station_index in solar_indices:
            station = stations.iloc[station_index]
            mask[issue_index, :, station_index] = [
                _solar_elevation_degrees(
                    timestamp,
                    float(station.latitude),
                    float(station.longitude),
                )
                > float(elevation_threshold_deg)
                for timestamp in timestamps
            ]
    audit = {
        "method": "station_specific_noaa_solar_elevation",
        "split": split,
        "timezone": "Asia/Shanghai (UTC+08:00)",
        "elevation_threshold_deg": float(elevation_threshold_deg),
        "timestamp_offset_minutes": float(timestamp_offset_minutes),
        "solar_station_count": int(len(solar_indices)),
        "solar_daylight_fraction": float(mask[:, :, solar_indices].mean()),
        "uses_power_or_actual": False,
    }
    return mask, audit


def get_station_dataloader(
    data_dir: str | Path,
    split: str,
    residual_scale: Mapping[str, object],
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    condition_config: Mapping[str, object] | None = None,
    state_thresholds: Mapping[str, object] | None = None,
    event_weighting: Mapping[str, object] | None = None,
    event_replay: Mapping[str, object] | None = None,
    jstd_targets: JSTDTargetArrays | None = None,
    retrieval_arrays: object | None = None,
    forecast_trust_arrays: object | None = None,
) -> tuple[DataLoader, StationForecastDataset]:
    dataset = StationForecastDataset(
        data_dir,
        split,
        residual_scale,
        condition_config=condition_config,
        state_thresholds=state_thresholds,
        event_weighting=event_weighting,
        event_replay=event_replay,
        jstd_targets=jstd_targets,
        retrieval_arrays=retrieval_arrays,
        forecast_trust_arrays=forecast_trust_arrays,
    )
    generator = torch.Generator()
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}[split]
    generator.manual_seed(int(seed) + split_offset)
    sampler = None
    if split == "train" and jstd_targets is not None:
        sampler = WeightedRandomSampler(
            torch.as_tensor(jstd_targets.sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    elif split == "train" and event_replay is not None:
        replay = validate_station_event_replay(event_replay, len(dataset))
        sampler = WeightedRandomSampler(
            torch.as_tensor(replay["sample_replay_weights"], dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    worker_count = int(num_workers)
    loader_options: dict[str, object] = {}
    if worker_count > 0:
        loader_options["persistent_workers"] = bool(persistent_workers)
        loader_options["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=_seed_worker,
        **loader_options,
    )
    return loader, dataset


def write_residual_scale(path: str | Path, residual_scale: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(dict(residual_scale), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_station_state_thresholds(
    path: str | Path, thresholds: Mapping[str, object]
) -> None:
    Path(path).write_text(
        json.dumps(dict(thresholds), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_station_event_weighting(
    path: str | Path, specification: Mapping[str, object]
) -> None:
    Path(path).write_text(
        json.dumps(dict(specification), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_station_event_replay(
    path: str | Path, specification: Mapping[str, object]
) -> None:
    Path(path).write_text(
        json.dumps(dict(specification), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
