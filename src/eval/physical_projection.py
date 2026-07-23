"""Deterministic projection of generated power scenarios onto physical bounds."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_SOLAR_NIGHT_THRESHOLD_MW = 1.0
DEFAULT_SHANDONG_LATITUDE_BOUNDS = (34.0, 39.0)
DEFAULT_SHANDONG_LONGITUDE_BOUNDS = (114.0, 123.0)
DEFAULT_SOLAR_ELEVATION_THRESHOLD_DEG = -0.833
DEFAULT_TIMESTAMP_OFFSET_MINUTES = 30.0


def _percentage(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) * 100.0 / mask.size)


def _solar_elevation_degrees(
    timestamp: datetime,
    latitude_deg: float,
    longitude_deg: float,
) -> float:
    """Approximate apparent solar elevation using the NOAA solar equations."""
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    local_minutes = (
        timestamp.hour * 60.0
        + timestamp.minute
        + timestamp.second / 60.0
        + timestamp.microsecond / 60_000_000.0
    )
    days_in_year = 366 if (
        timestamp.year % 4 == 0
        and (timestamp.year % 100 != 0 or timestamp.year % 400 == 0)
    ) else 365
    fractional_hour = local_minutes / 60.0
    gamma = (
        2.0
        * math.pi
        / days_in_year
        * (timestamp.timetuple().tm_yday - 1 + (fractional_hour - 12.0) / 24.0)
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.00148 * math.sin(3.0 * gamma)
    )
    utc_offset_hours = timestamp.utcoffset().total_seconds() / 3600.0
    time_offset = (
        equation_of_time + 4.0 * float(longitude_deg) - 60.0 * utc_offset_hours
    )
    true_solar_minutes = (local_minutes + time_offset) % 1440.0
    hour_angle_deg = true_solar_minutes / 4.0 - 180.0
    latitude = math.radians(float(latitude_deg))
    hour_angle = math.radians(hour_angle_deg)
    cos_zenith = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude) * math.cos(declination) * math.cos(hour_angle)
    )
    cos_zenith = min(1.0, max(-1.0, cos_zenith))
    return 90.0 - math.degrees(math.acos(cos_zenith))


def conservative_shandong_daylight(
    timestamps: Sequence[datetime],
    latitude_bounds: tuple[float, float] = DEFAULT_SHANDONG_LATITUDE_BOUNDS,
    longitude_bounds: tuple[float, float] = DEFAULT_SHANDONG_LONGITUDE_BOUNDS,
    elevation_threshold_deg: float = DEFAULT_SOLAR_ELEVATION_THRESHOLD_DEG,
) -> np.ndarray:
    """Return daylight if any point in a broad Shandong envelope sees sun."""
    lat_min, lat_max = map(float, latitude_bounds)
    lon_min, lon_max = map(float, longitude_bounds)
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(f"invalid latitude bounds: {latitude_bounds}")
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise ValueError(f"invalid longitude bounds: {longitude_bounds}")
    threshold = float(elevation_threshold_deg)
    if not np.isfinite(threshold):
        raise ValueError("elevation_threshold_deg must be finite")

    latitudes = (lat_min, (lat_min + lat_max) / 2.0, lat_max)
    longitudes = (lon_min, (lon_min + lon_max) / 2.0, lon_max)
    daylight = []
    for timestamp in timestamps:
        max_elevation = max(
            _solar_elevation_degrees(timestamp, latitude, longitude)
            for latitude in latitudes
            for longitude in longitudes
        )
        daylight.append(max_elevation > threshold)
    return np.asarray(daylight, dtype=bool)


def daylight_mask_from_export_metadata(
    export_metadata_path: str | Path,
    data_split: str,
    window_count: int,
    sequence_length: int,
    latitude_bounds: tuple[float, float] = DEFAULT_SHANDONG_LATITUDE_BOUNDS,
    longitude_bounds: tuple[float, float] = DEFAULT_SHANDONG_LONGITUDE_BOUNDS,
    elevation_threshold_deg: float = DEFAULT_SOLAR_ELEVATION_THRESHOLD_DEG,
    timestamp_offset_minutes: float = DEFAULT_TIMESTAMP_OFFSET_MINUTES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a stride-one [N,L] daylight mask without using val/test power."""
    metadata_path = Path(export_metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    split = metadata.get("splits", {}).get(str(data_split))
    if split is None:
        raise KeyError(f"split {data_split!r} missing from {metadata_path}")
    expected_windows = int(split["windows"])
    expected_hours = int(split["hours"])
    window_count = int(window_count)
    sequence_length = int(sequence_length)
    if window_count != expected_windows:
        raise ValueError(
            f"{data_split} window count {window_count} != metadata {expected_windows}"
        )
    if expected_hours != window_count + sequence_length - 1:
        raise ValueError(
            f"{data_split} hours {expected_hours} are inconsistent with "
            f"{window_count} stride-one windows of length {sequence_length}"
        )
    offset = float(timestamp_offset_minutes)
    if not np.isfinite(offset):
        raise ValueError("timestamp_offset_minutes must be finite")
    start = datetime.fromisoformat(str(split["start_local"]))
    unique_timestamps = [
        start + timedelta(hours=index, minutes=offset)
        for index in range(expected_hours)
    ]
    unique_daylight = conservative_shandong_daylight(
        unique_timestamps,
        latitude_bounds=latitude_bounds,
        longitude_bounds=longitude_bounds,
        elevation_threshold_deg=elevation_threshold_deg,
    )
    indices = (
        np.arange(window_count, dtype=np.int64)[:, None]
        + np.arange(sequence_length, dtype=np.int64)[None, :]
    )
    mask = unique_daylight[indices]
    audit = {
        "method": "astronomical_any_point_in_broad_shandong_envelope",
        "export_metadata_path": str(metadata_path.resolve()),
        "data_split": str(data_split),
        "latitude_bounds_deg": list(map(float, latitude_bounds)),
        "longitude_bounds_deg": list(map(float, longitude_bounds)),
        "solar_elevation_threshold_deg": float(elevation_threshold_deg),
        "timestamp_offset_minutes": offset,
        "daylight_point_pct": float(np.mean(mask) * 100.0),
        "unique_daylight_hours": int(np.count_nonzero(unique_daylight)),
        "unique_total_hours": int(unique_daylight.size),
    }
    return mask, audit


def daylight_mask_from_train_support(
    data_path: str | Path,
    data_split: str,
    window_count: int,
    sequence_length: int,
    fit_split: str = "train",
    support_quantile: float = 0.95,
    minimum_fraction_of_train_peak: float = 1.0e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit allowable solar clock hours on train unique hours and reuse them."""
    root = Path(data_path)
    metadata_path = root / "export_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    splits = metadata.get("splits", {})
    fit_metadata = splits.get(str(fit_split))
    target_metadata = splits.get(str(data_split))
    if fit_metadata is None or target_metadata is None:
        raise KeyError(
            f"fit/target split {fit_split!r}/{data_split!r} missing from {metadata_path}"
        )
    fit_windows = np.load(root / f"{fit_split}_actual.npy")
    if fit_windows.ndim != 3 or fit_windows.shape[2] < 2:
        raise ValueError(
            f"{fit_split}_actual.npy must be [N,L,C>=2], got {fit_windows.shape}"
        )
    if fit_windows.shape[0] != int(fit_metadata["windows"]):
        raise ValueError("fit window count does not match export metadata")
    if fit_windows.shape[1] != int(sequence_length):
        raise ValueError(
            f"fit sequence length {fit_windows.shape[1]} != {sequence_length}"
        )
    fit_unique = np.concatenate(
        [np.asarray(fit_windows[0]), np.asarray(fit_windows[1:, -1, :])],
        axis=0,
    )
    if fit_unique.shape[0] != int(fit_metadata["hours"]):
        raise ValueError("reconstructed fit hours do not match export metadata")
    quantile = float(support_quantile)
    minimum_fraction = float(minimum_fraction_of_train_peak)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("support_quantile must be in (0,1]")
    if not np.isfinite(minimum_fraction) or minimum_fraction < 0.0:
        raise ValueError("minimum_fraction_of_train_peak must be non-negative")

    fit_start = datetime.fromisoformat(str(fit_metadata["start_local"]))
    fit_hours = (
        fit_start.hour + np.arange(fit_unique.shape[0], dtype=np.int64)
    ) % 24
    solar = fit_unique[:, 1]
    train_peak = float(np.max(solar))
    threshold = max(np.finfo(np.float32).eps, train_peak * minimum_fraction)
    hourly_quantiles = np.asarray([
        np.quantile(solar[fit_hours == hour], quantile)
        if np.any(fit_hours == hour) else 0.0
        for hour in range(24)
    ])
    supported_hours = np.flatnonzero(hourly_quantiles > threshold)
    if supported_hours.size == 0:
        raise ValueError("train solar support is empty")

    expected_windows = int(target_metadata["windows"])
    expected_hours = int(target_metadata["hours"])
    window_count = int(window_count)
    sequence_length = int(sequence_length)
    if window_count != expected_windows:
        raise ValueError(
            f"{data_split} window count {window_count} != metadata {expected_windows}"
        )
    if expected_hours != window_count + sequence_length - 1:
        raise ValueError(
            f"{data_split} hours {expected_hours} are inconsistent with "
            f"{window_count} stride-one windows of length {sequence_length}"
        )
    target_start = datetime.fromisoformat(str(target_metadata["start_local"]))
    target_hours = (
        target_start.hour + np.arange(expected_hours, dtype=np.int64)
    ) % 24
    unique_daylight = np.isin(target_hours, supported_hours)
    indices = (
        np.arange(window_count, dtype=np.int64)[:, None]
        + np.arange(sequence_length, dtype=np.int64)[None, :]
    )
    mask = unique_daylight[indices]
    audit = {
        "method": "train_unique_hour_solar_clock_support",
        "data_path": str(root.resolve()),
        "fit_split": str(fit_split),
        "data_split": str(data_split),
        "support_quantile": quantile,
        "minimum_fraction_of_train_peak": minimum_fraction,
        "support_threshold_normalized": threshold,
        "train_solar_peak_normalized": train_peak,
        "supported_local_hours": supported_hours.tolist(),
        "hourly_support_quantiles_normalized": hourly_quantiles.tolist(),
        "daylight_point_pct": float(np.mean(mask) * 100.0),
        "validation_or_test_actual_used_for_fit": False,
    }
    return mask, audit


def physical_boundary_rates(
    scenarios: np.ndarray,
    upper_bounds_mw: np.ndarray,
) -> dict[str, float]:
    """Return raw scalar boundary rates for [N,S,3,L] scenarios."""
    values = np.asarray(scenarios)
    bounds = np.asarray(upper_bounds_mw, dtype=np.float64)
    if values.ndim != 4 or values.shape[2] != 3:
        raise ValueError(f"scenarios must be [N,S,3,L], got {values.shape}")
    if bounds.shape != (3,) or not np.isfinite(bounds).all() or np.any(bounds <= 0):
        raise ValueError(f"upper_bounds_mw must contain three positive values, got {bounds}")

    wind = values[:, :, 0, :]
    solar = values[:, :, 1, :]
    load = values[:, :, 2, :]
    invalid = (
        (wind < 0.0)
        | (wind > bounds[0])
        | (solar < 0.0)
        | (solar > bounds[1])
        | (load < 0.0)
    )
    return {
        "wind_below_zero_pct": _percentage(wind < 0.0),
        "wind_above_upper_bound_pct": _percentage(wind > bounds[0]),
        "solar_below_zero_pct": _percentage(solar < 0.0),
        "solar_above_upper_bound_pct": _percentage(solar > bounds[1]),
        "load_below_zero_pct": _percentage(load < 0.0),
        "any_physical_violation_pct": _percentage(invalid),
    }


def project_power_scenarios(
    scenarios: np.ndarray,
    forecast_mw: np.ndarray,
    upper_bounds_mw: np.ndarray,
    solar_night_threshold_mw: float = DEFAULT_SOLAR_NIGHT_THRESHOLD_MW,
    solar_daylight_mask: np.ndarray | None = None,
    solar_daylight_metadata: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project scenarios without mutating the raw input.

    Wind and solar are clipped to their normalization upper bounds, load is
    floored at zero, and solar is set to zero using a supplied timestamp-derived
    daylight mask. The forecast threshold remains a compatibility fallback.
    """
    raw = np.asarray(scenarios)
    forecast = np.asarray(forecast_mw)
    bounds = np.asarray(upper_bounds_mw, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[2] != 3:
        raise ValueError(f"scenarios must be [N,S,3,L], got {raw.shape}")
    expected_forecast = (raw.shape[0], 3, raw.shape[3])
    if forecast.shape != expected_forecast:
        raise ValueError(
            f"forecast_mw must be {expected_forecast}, got {forecast.shape}"
        )
    if not np.isfinite(raw).all() or not np.isfinite(forecast).all():
        raise ValueError("scenarios and forecast_mw must be finite")
    if bounds.shape != (3,) or not np.isfinite(bounds).all() or np.any(bounds <= 0):
        raise ValueError(f"upper_bounds_mw must contain three positive values, got {bounds}")
    threshold = float(solar_night_threshold_mw)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("solar_night_threshold_mw must be finite and non-negative")

    projected = np.array(raw, copy=True)
    projected[:, :, 0, :] = np.clip(projected[:, :, 0, :], 0.0, bounds[0])
    projected[:, :, 1, :] = np.clip(projected[:, :, 1, :], 0.0, bounds[1])
    projected[:, :, 2, :] = np.maximum(projected[:, :, 2, :], 0.0)

    if solar_daylight_mask is None:
        solar_night_mask = forecast[:, 1, :] <= threshold
        solar_night_rule = "forecast_mw <= threshold"
        daylight_audit: dict[str, Any] = {
            "method": "forecast_threshold_fallback",
            "solar_night_threshold_mw": threshold,
        }
    else:
        daylight = np.asarray(solar_daylight_mask)
        expected_mask = (raw.shape[0], raw.shape[3])
        if daylight.shape != expected_mask:
            raise ValueError(
                f"solar_daylight_mask must be {expected_mask}, got {daylight.shape}"
            )
        if daylight.dtype != np.bool_:
            if not np.isin(daylight, [0, 1]).all():
                raise ValueError(
                    "solar_daylight_mask must be boolean or contain only 0/1"
                )
            daylight = daylight.astype(bool)
        solar_night_mask = ~daylight
        solar_night_rule = "not timestamp-derived daylight mask"
        daylight_audit = dict(solar_daylight_metadata or {})
        daylight_audit.setdefault("method", "caller_supplied_daylight_mask")
        daylight_audit["daylight_point_pct"] = float(np.mean(daylight) * 100.0)
    projected[:, :, 1, :] = np.where(
        solar_night_mask[:, None, :],
        0.0,
        projected[:, :, 1, :],
    )

    raw_rates = physical_boundary_rates(raw, bounds)
    projected_rates = physical_boundary_rates(projected, bounds)
    changed = projected != raw
    report = {
        "method": "clip_wind_solar_floor_load_and_zero_solar_at_night",
        "channel_order": ["wind", "solar", "load"],
        "upper_bounds_mw": bounds.tolist(),
        "solar_night_rule": solar_night_rule,
        "solar_daylight_audit": daylight_audit,
        "solar_night_point_pct": float(np.mean(solar_night_mask) * 100.0),
        "changed_scalar_count": int(np.count_nonzero(changed)),
        "changed_scalar_pct": _percentage(changed),
        "raw_boundary_rates": raw_rates,
        "projected_boundary_rates": projected_rates,
        "raw_preserved": True,
    }
    return projected.astype(raw.dtype, copy=False), report
