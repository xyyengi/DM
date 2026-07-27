"""Leak-free standardization for diffusion residual targets.

The legacy/default mode uses one train-only mean and standard deviation per
channel.  The optional ``solar_forecast_conditional`` mode keeps wind and load
unchanged while using a smooth train-only solar location/scale curve indexed by
the solar forecast.  No validation/test actuals or residuals are used to fit it.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


CHANNEL_ORDER = ("wind", "solar", "load")
SUPPORTED_RESIDUAL_DEFINITIONS = {"forecast_minus_actual", "actual_minus_forecast"}
GLOBAL_MODE = "global_channelwise"
SOLAR_CONDITIONAL_MODE = "solar_forecast_conditional"
SUPPORTED_STANDARDIZATION_MODES = {GLOBAL_MODE, SOLAR_CONDITIONAL_MODE}


def to_internal_residual(values: np.ndarray, residual_definition: str) -> np.ndarray:
    """Return residuals in the project convention: forecast - actual."""
    if residual_definition not in SUPPORTED_RESIDUAL_DEFINITIONS:
        raise ValueError(f"Unsupported residual_definition={residual_definition!r}")
    result = np.asarray(values, dtype=np.float64)
    return -result if residual_definition == "actual_minus_forecast" else result


def recover_stride_one_unique_hours(windows: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Recover one continuous series from stride-one overlapping windows.

    The overlap is verified before fitting statistics so duplicated hours cannot
    silently receive extra weight.
    """
    values = np.asarray(windows)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"Expected [windows,time,channels], got {values.shape}")
    if values.shape[0] > 1 and not np.allclose(
        values[:-1, 1:, :], values[1:, :-1, :], atol=atol, rtol=0.0
    ):
        raise ValueError("Training residual windows are not consistent stride-one overlaps")
    return np.concatenate([values[0], values[1:, -1, :]], axis=0)


def _smooth_curve(values: np.ndarray, passes: int, log_space: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if log_space:
        if np.any(result <= 0):
            raise ValueError("log-space smoothing requires positive values")
        result = np.log(result)
    for _ in range(int(passes)):
        if result.size < 2:
            break
        padded = np.pad(result, (1, 1), mode="edge")
        result = (
            0.25 * padded[:-2]
            + 0.50 * padded[1:-1]
            + 0.25 * padded[2:]
        )
    return np.exp(result) if log_space else result


def _fit_solar_forecast_conditioning(
    unique_forecast: np.ndarray,
    unique_residual: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    settings: Mapping[str, object],
    epsilon: float,
) -> dict:
    n_bins = int(settings.get("n_bins", 12))
    min_bin_count = int(settings.get("min_bin_count", 128))
    shrinkage_count = float(settings.get("shrinkage_count", 128.0))
    smooth_passes = int(settings.get("smooth_passes", 2))
    min_scale_ratio = float(settings.get("min_scale_ratio", 0.25))
    max_scale_ratio = float(settings.get("max_scale_ratio", 4.0))
    daylight_threshold = float(
        settings.get("daylight_forecast_threshold_normalized", 1e-8)
    )
    if n_bins < 2:
        raise ValueError("solar conditional standardization requires n_bins >= 2")
    if min_bin_count < 2:
        raise ValueError("min_bin_count must be >= 2")
    if shrinkage_count < 0:
        raise ValueError("shrinkage_count must be non-negative")
    if smooth_passes < 0:
        raise ValueError("smooth_passes must be non-negative")
    if not 0 < min_scale_ratio <= max_scale_ratio:
        raise ValueError("invalid conditional solar scale-ratio bounds")
    if not np.isfinite(daylight_threshold) or daylight_threshold < 0:
        raise ValueError("invalid daylight forecast threshold")

    solar_forecast = np.asarray(unique_forecast[:, 1], dtype=np.float64)
    solar_residual = np.asarray(unique_residual[:, 1], dtype=np.float64)
    daylight = solar_forecast > daylight_threshold
    if np.count_nonzero(daylight) < n_bins * min_bin_count:
        raise ValueError(
            "not enough train-only daylight points for conditional solar "
            f"standardization: {np.count_nonzero(daylight)} < "
            f"{n_bins * min_bin_count}"
        )
    forecast_daylight = solar_forecast[daylight]
    residual_daylight = solar_residual[daylight]
    edges = np.quantile(forecast_daylight, np.linspace(0.0, 1.0, n_bins + 1))
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            "solar forecast quantile edges are not strictly increasing; "
            "reduce n_bins or audit train daylight values"
        )
    bin_index = np.clip(np.digitize(forecast_daylight, edges[1:-1]), 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    centers = np.zeros(n_bins, dtype=np.float64)
    raw_mean = np.zeros(n_bins, dtype=np.float64)
    raw_std = np.zeros(n_bins, dtype=np.float64)
    for index in range(n_bins):
        selected = bin_index == index
        counts[index] = int(np.count_nonzero(selected))
        if counts[index] < min_bin_count:
            raise ValueError(
                f"solar conditional bin {index} has {counts[index]} points; "
                f"minimum is {min_bin_count}"
            )
        centers[index] = np.median(forecast_daylight[selected])
        raw_mean[index] = np.mean(residual_daylight[selected])
        raw_std[index] = np.std(residual_daylight[selected])
    if np.any(raw_std < epsilon) or not np.isfinite(raw_std).all():
        raise ValueError(f"invalid raw conditional solar std={raw_std}")

    weight = counts / (counts + shrinkage_count)
    shrunk_mean = weight * raw_mean + (1.0 - weight) * global_mean[1]
    shrunk_std = weight * raw_std + (1.0 - weight) * global_std[1]
    smooth_mean = _smooth_curve(shrunk_mean, smooth_passes, log_space=False)
    smooth_std = _smooth_curve(shrunk_std, smooth_passes, log_space=True)
    lower = max(epsilon, min_scale_ratio * global_std[1])
    upper = max(lower, max_scale_ratio * global_std[1])
    smooth_std = np.clip(smooth_std, lower, upper)
    if np.any(np.diff(centers) <= 0):
        raise ValueError("conditional solar forecast knots must be increasing")

    return {
        "channel": "solar",
        "forecast_units": "normalized_power",
        "fit_split": "train",
        "fit_scope": "unique_stride_one_daylight_hours",
        "daylight_forecast_threshold_normalized": daylight_threshold,
        "n_daylight_unique_hours": int(np.count_nonzero(daylight)),
        "n_bins": n_bins,
        "bin_edges": edges.tolist(),
        "forecast_knots": centers.tolist(),
        "raw_bin_mean": raw_mean.tolist(),
        "raw_bin_std": raw_std.tolist(),
        "mean_knots": smooth_mean.tolist(),
        "std_knots": smooth_std.tolist(),
        "bin_counts": counts.tolist(),
        "regularization": {
            "min_bin_count": min_bin_count,
            "shrinkage_count": shrinkage_count,
            "smooth_passes": smooth_passes,
            "min_scale_ratio": min_scale_ratio,
            "max_scale_ratio": max_scale_ratio,
            "global_solar_mean": float(global_mean[1]),
            "global_solar_std": float(global_std[1]),
        },
        "night_fallback": "global_solar_mean_std",
    }


def fit_residual_standardizer(
    train_residual_windows: np.ndarray,
    residual_definition: str,
    normalization_divisors: np.ndarray,
    epsilon: float = 1e-6,
    train_forecast_windows: np.ndarray | None = None,
    mode: str = GLOBAL_MODE,
    solar_conditioning: Mapping[str, object] | None = None,
) -> dict:
    """Fit global or solar-conditional statistics on train unique hours only."""
    if mode not in SUPPORTED_STANDARDIZATION_MODES:
        raise ValueError(
            f"Unsupported residual standardization mode={mode!r}; "
            f"expected one of {sorted(SUPPORTED_STANDARDIZATION_MODES)}"
        )
    raw = np.asarray(train_residual_windows)[..., :3]
    internal = to_internal_residual(raw, residual_definition)
    divisors = np.asarray(normalization_divisors, dtype=np.float64)
    if divisors.shape != (3,) or not np.isfinite(divisors).all() or np.any(divisors <= 0):
        raise ValueError(f"Invalid normalization_divisors={divisors}")
    unique = recover_stride_one_unique_hours(internal) / divisors.reshape(1, 3)
    mean = np.mean(unique, axis=0)
    std = np.std(unique, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < epsilon):
        raise ValueError(f"Invalid fitted residual statistics mean={mean}, std={std}")
    result = {
        "enabled": True,
        "mode": mode,
        "fit_split": "train",
        "fit_scope": "unique_stride_one_hours",
        "n_unique_hours": int(unique.shape[0]),
        "channel_order": list(CHANNEL_ORDER),
        "residual_definition": "forecast_minus_actual",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epsilon": float(epsilon),
    }
    if mode == SOLAR_CONDITIONAL_MODE:
        if train_forecast_windows is None:
            raise ValueError(
                "train_forecast_windows is required for "
                "solar_forecast_conditional standardization"
            )
        forecast_raw = np.asarray(train_forecast_windows)[..., :3]
        if forecast_raw.shape != raw.shape:
            raise ValueError(
                "train forecast/residual window shapes must match, got "
                f"{forecast_raw.shape} and {raw.shape}"
            )
        unique_forecast = (
            recover_stride_one_unique_hours(forecast_raw)
            / divisors.reshape(1, 3)
        )
        result["solar_conditioning"] = _fit_solar_forecast_conditioning(
            unique_forecast,
            unique,
            mean,
            std,
            solar_conditioning or {},
            epsilon,
        )
    return result


def validate_standardizer(stats: Mapping[str, object]) -> dict:
    """Validate and normalize a serialized residual standardizer."""
    if not bool(stats.get("enabled", False)):
        raise ValueError("Residual standardizer must have enabled=true")
    mode = str(stats.get("mode", GLOBAL_MODE))
    if mode not in SUPPORTED_STANDARDIZATION_MODES:
        raise ValueError(f"Unsupported residual standardization mode={mode!r}")
    mean = np.asarray(stats.get("mean"), dtype=np.float64)
    std = np.asarray(stats.get("std"), dtype=np.float64)
    epsilon = float(stats.get("epsilon", 1e-6))
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError(f"Residual standardizer expects three channels, got mean={mean}, std={std}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < epsilon):
        raise ValueError(f"Invalid residual standardizer mean={mean}, std={std}")
    channel_order = tuple(stats.get("channel_order", CHANNEL_ORDER))
    if channel_order != CHANNEL_ORDER:
        raise ValueError(f"Unexpected residual channel order {channel_order}")
    result = {
        **dict(stats),
        "enabled": True,
        "mode": mode,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epsilon": epsilon,
        "channel_order": list(CHANNEL_ORDER),
        "residual_definition": "forecast_minus_actual",
    }
    if mode == SOLAR_CONDITIONAL_MODE:
        conditional = dict(stats.get("solar_conditioning") or {})
        knots = np.asarray(conditional.get("forecast_knots"), dtype=np.float64)
        mean_knots = np.asarray(conditional.get("mean_knots"), dtype=np.float64)
        std_knots = np.asarray(conditional.get("std_knots"), dtype=np.float64)
        if (
            knots.ndim != 1
            or knots.size < 2
            or mean_knots.shape != knots.shape
            or std_knots.shape != knots.shape
        ):
            raise ValueError("invalid conditional solar knot shapes")
        if (
            not np.isfinite(knots).all()
            or not np.isfinite(mean_knots).all()
            or not np.isfinite(std_knots).all()
            or np.any(std_knots < epsilon)
            or np.any(np.diff(knots) <= 0)
        ):
            raise ValueError("invalid conditional solar knot values")
        threshold = float(
            conditional.get("daylight_forecast_threshold_normalized", 1e-8)
        )
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid conditional solar daylight threshold")
        conditional.update(
            {
                "forecast_knots": knots.tolist(),
                "mean_knots": mean_knots.tolist(),
                "std_knots": std_knots.tolist(),
                "daylight_forecast_threshold_normalized": threshold,
            }
        )
        result["solar_conditioning"] = conditional
    return result


def _conditional_solar_location_scale(
    forecast: np.ndarray,
    forecast_channel_axis: int,
    target_shape: tuple[int, ...],
    checked: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    conditional = checked["solar_conditioning"]
    solar_forecast = np.take(
        np.asarray(forecast, dtype=np.float64), 1, axis=forecast_channel_axis
    )
    while solar_forecast.ndim < len(target_shape):
        solar_forecast = np.expand_dims(solar_forecast, axis=1)
    try:
        solar_forecast = np.broadcast_to(solar_forecast, target_shape)
    except ValueError as exc:
        raise ValueError(
            "forecast without its channel axis cannot broadcast to the residual "
            f"solar slice: {solar_forecast.shape} vs {target_shape}"
        ) from exc
    knots = np.asarray(conditional["forecast_knots"], dtype=np.float64)
    mean_knots = np.asarray(conditional["mean_knots"], dtype=np.float64)
    std_knots = np.asarray(conditional["std_knots"], dtype=np.float64)
    location = np.interp(solar_forecast, knots, mean_knots)
    scale = np.interp(solar_forecast, knots, std_knots)
    threshold = float(conditional["daylight_forecast_threshold_normalized"])
    night = solar_forecast <= threshold
    location = np.where(night, float(checked["mean"][1]), location)
    scale = np.where(night, float(checked["std"][1]), scale)
    return location, scale


def standardize_residual(
    values: np.ndarray,
    stats: Mapping[str, object],
    channel_axis: int,
    forecast: np.ndarray | None = None,
    forecast_channel_axis: int | None = None,
) -> np.ndarray:
    checked = validate_standardizer(stats)
    mean = np.asarray(checked["mean"], dtype=np.float64)
    std = np.asarray(checked["std"], dtype=np.float64)
    array = np.asarray(values)
    shape = [1] * np.ndim(values)
    shape[channel_axis] = 3
    result = (array - mean.reshape(shape)) / std.reshape(shape)
    if checked["mode"] == SOLAR_CONDITIONAL_MODE:
        if forecast is None or forecast_channel_axis is None:
            raise ValueError(
                "forecast and forecast_channel_axis are required for "
                "solar_forecast_conditional standardization"
            )
        solar_values = np.take(array, 1, axis=channel_axis)
        location, scale = _conditional_solar_location_scale(
            forecast, forecast_channel_axis, solar_values.shape, checked
        )
        solar_index = [slice(None)] * array.ndim
        solar_index[channel_axis] = 1
        result[tuple(solar_index)] = (solar_values - location) / scale
    return result


def inverse_standardize_residual(
    values: np.ndarray,
    stats: Mapping[str, object],
    channel_axis: int,
    forecast: np.ndarray | None = None,
    forecast_channel_axis: int | None = None,
) -> np.ndarray:
    checked = validate_standardizer(stats)
    mean = np.asarray(checked["mean"], dtype=np.float64)
    std = np.asarray(checked["std"], dtype=np.float64)
    array = np.asarray(values)
    shape = [1] * np.ndim(values)
    shape[channel_axis] = 3
    result = array * std.reshape(shape) + mean.reshape(shape)
    if checked["mode"] == SOLAR_CONDITIONAL_MODE:
        if forecast is None or forecast_channel_axis is None:
            raise ValueError(
                "forecast and forecast_channel_axis are required for inverse "
                "solar_forecast_conditional standardization"
            )
        solar_values = np.take(array, 1, axis=channel_axis)
        location, scale = _conditional_solar_location_scale(
            forecast, forecast_channel_axis, solar_values.shape, checked
        )
        solar_index = [slice(None)] * array.ndim
        solar_index[channel_axis] = 1
        result[tuple(solar_index)] = solar_values * scale + location
    return result
