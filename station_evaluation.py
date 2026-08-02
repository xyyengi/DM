"""Evaluation metrics for station-level probabilistic wind/solar scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


DEFAULT_INTERVAL_LEVELS = (0.80, 0.90, 0.95)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def ensemble_crps(samples: np.ndarray, actual: np.ndarray) -> float:
    """Ensemble CRPS with samples on axis 1, using an O(K log K) formula."""
    samples = np.asarray(samples, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if samples.shape[0] != actual.shape[0] or samples.shape[2:] != actual.shape[1:]:
        raise ValueError("sample/actual shapes do not align")
    members = samples.shape[1]
    term1 = np.mean(np.abs(samples - actual[:, None, ...]), axis=1)
    sorted_samples = np.sort(samples, axis=1)
    coefficients = (2 * np.arange(1, members + 1) - members - 1).astype(np.float64)
    coefficient_shape = (1, members) + (1,) * (samples.ndim - 2)
    half_pair_term = np.sum(
        sorted_samples * coefficients.reshape(coefficient_shape), axis=1
    ) / (members**2)
    return float(np.mean(term1 - half_pair_term))


def interval_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    nominal: float = 0.90,
) -> tuple[float, float, float, float]:
    alpha = (1.0 - nominal) / 2.0
    lower = np.quantile(samples, alpha, axis=1)
    upper = np.quantile(samples, 1.0 - alpha, axis=1)
    covered = (actual >= lower) & (actual <= upper)
    below = actual < lower
    above = actual > upper
    return (
        float(covered.mean()),
        float(np.mean(upper - lower)),
        float(below.mean()),
        float(above.mean()),
    )


def point_metrics(samples: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    mean_scenario = np.mean(samples, axis=1)
    error = mean_scenario - actual
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def _select_valid_points(
    samples: np.ndarray,
    actual: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if valid_mask is None:
        return samples, actual
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != actual.shape:
        raise ValueError("valid_mask must have the same shape as actual")
    if not np.any(valid_mask):
        raise ValueError("valid_mask does not select any evaluation points")
    # Move the member axis behind the observation axes so the observation mask
    # selects the same points from every ensemble member.
    selected_samples = np.moveaxis(samples, 1, -1)[valid_mask]
    selected_actual = actual[valid_mask]
    return selected_samples, selected_actual


def metric_bundle(
    samples: np.ndarray,
    actual: np.ndarray,
    interval_levels: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    samples, actual = _select_valid_points(samples, actual, valid_mask)
    mae, rmse = point_metrics(samples, actual)
    result = {
        "crps": ensemble_crps(samples, actual),
        "scenario_mean_mae": mae,
        "scenario_mean_rmse": rmse,
    }
    for nominal in interval_levels:
        if not 0.0 < nominal < 1.0:
            raise ValueError(f"interval level must be in (0, 1), got {nominal}")
        label = str(int(round(100 * nominal)))
        coverage, width, below, above = interval_metrics(
            samples, actual, nominal=nominal
        )
        result[f"coverage_{label}"] = coverage
        result[f"width_{label}"] = width
        result[f"below_{label}"] = below
        result[f"above_{label}"] = above
    return result


def energy_score(samples: np.ndarray, actual: np.ndarray) -> float:
    scores = []
    for issue_index in range(samples.shape[0]):
        ensemble = samples[issue_index].reshape(samples.shape[1], -1).astype(np.float64)
        observation = actual[issue_index].reshape(-1).astype(np.float64)
        first = np.linalg.norm(ensemble - observation[None, :], axis=1).mean()
        # 0.5 * mean over ordered pairs equals sum of unordered distances / K^2.
        second = pdist(ensemble, metric="euclidean").sum() / (ensemble.shape[0] ** 2)
        scores.append(first - second)
    return float(np.mean(scores))


def adjacency_variogram_score(
    samples: np.ndarray,
    actual: np.ndarray,
    adjacency: np.ndarray,
    power: float = 0.5,
) -> float:
    rows, cols = np.where(np.triu(adjacency, k=1) > 0)
    weights = adjacency[rows, cols].astype(np.float64)
    total = 0.0
    weight_total = 0.0
    for i, j, weight in zip(rows, cols, weights, strict=True):
        observed = np.abs(actual[:, :, i] - actual[:, :, j]) ** power
        generated = np.mean(
            np.abs(samples[:, :, :, i] - samples[:, :, :, j]) ** power,
            axis=1,
        )
        total += float(weight) * float(np.sum((observed - generated) ** 2))
        weight_total += float(weight) * observed.size
    return total / max(weight_total, 1e-12)


def _correlation(values: np.ndarray) -> np.ndarray:
    values = values.reshape(-1, values.shape[-1]).astype(np.float64)
    return np.corrcoef(values, rowvar=False)


def spatial_correlation_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    adjacency: np.ndarray,
    station_types: np.ndarray,
) -> dict[str, float]:
    actual_correlation = _correlation(actual)
    generated_correlation = _correlation(
        samples.transpose(0, 2, 1, 3).reshape(-1, samples.shape[-1])
    )
    difference = generated_correlation - actual_correlation
    upper = np.triu_indices(actual.shape[-1], k=1)
    edge = np.triu(adjacency, k=1) > 0
    result = {
        "spatial_corr_rmse_all_pairs": float(
            np.sqrt(np.mean(difference[upper] ** 2))
        ),
        "spatial_corr_rmse_adjacent_pairs": float(
            np.sqrt(np.mean(difference[edge] ** 2))
        ),
    }
    for label, mask in [
        ("wind_wind", np.equal.outer(station_types, "wind")),
        ("solar_solar", np.equal.outer(station_types, "solar")),
        ("wind_solar", np.not_equal.outer(station_types, station_types)),
    ]:
        pair_mask = np.triu(mask, k=1)
        result[f"spatial_corr_rmse_{label}"] = float(
            np.sqrt(np.mean(difference[pair_mask] ** 2))
        )
    return result


def temporal_acf(values: np.ndarray, lag: int) -> float:
    left = values[..., :-lag, :].reshape(-1)
    right = values[..., lag:, :].reshape(-1)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def temporal_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    station_types: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for station_type in ["wind", "solar"]:
        indices = np.flatnonzero(station_types == station_type)
        actual_type = actual[:, :, indices]
        generated_type = samples[:, :, :, indices]
        for lag in [1, 6, 24, 48]:
            actual_acf = temporal_acf(actual_type, lag)
            generated_acf = temporal_acf(generated_type, lag)
            result[f"{station_type}_acf_actual_lag{lag}"] = actual_acf
            result[f"{station_type}_acf_generated_lag{lag}"] = generated_acf
            result[f"{station_type}_acf_abs_error_lag{lag}"] = abs(
                generated_acf - actual_acf
            )
    return result


def ramp_metric_bundle(
    samples: np.ndarray,
    actual: np.ndarray,
    lag: int,
    interval_levels: tuple[float, ...],
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    scenario_ramp = samples[:, :, lag:, ...] - samples[:, :, :-lag, ...]
    actual_ramp = actual[:, lag:, ...] - actual[:, :-lag, ...]
    ramp_mask = None
    if valid_mask is not None:
        ramp_mask = valid_mask[:, lag:, ...] & valid_mask[:, :-lag, ...]
    return metric_bundle(
        scenario_ramp,
        actual_ramp,
        interval_levels=interval_levels,
        valid_mask=ramp_mask,
    )


def extreme_ramp_metric_bundle(
    samples: np.ndarray,
    actual: np.ndarray,
    lag: int,
    interval_levels: tuple[float, ...],
    valid_mask: np.ndarray | None = None,
    quantile: float = 0.90,
) -> dict[str, float]:
    scenario_ramp = samples[:, :, lag:, ...] - samples[:, :, :-lag, ...]
    actual_ramp = actual[:, lag:, ...] - actual[:, :-lag, ...]
    candidate = np.ones_like(actual_ramp, dtype=bool)
    if valid_mask is not None:
        candidate &= valid_mask[:, lag:, ...] & valid_mask[:, :-lag, ...]
    threshold = float(np.quantile(np.abs(actual_ramp[candidate]), quantile))
    extreme_mask = candidate & (np.abs(actual_ramp) >= threshold)
    result = metric_bundle(
        scenario_ramp,
        actual_ramp,
        interval_levels=interval_levels,
        valid_mask=extreme_mask,
    )
    result["absolute_ramp_threshold"] = threshold
    result["selected_point_count"] = int(np.sum(extreme_mask))
    return result


def daily_peak_metric_bundle(
    samples: np.ndarray,
    actual: np.ndarray,
    interval_levels: tuple[float, ...],
) -> dict[str, float]:
    if actual.shape[1] != 168:
        raise ValueError("daily peak evaluation expects a 168-hour window")
    issue_count, member_count = samples.shape[:2]
    actual_days = actual.reshape(issue_count, 7, 24)
    peak_hour = np.argmax(actual_days, axis=2)
    issue_index = np.repeat(np.arange(issue_count), 7)
    day_index = np.tile(np.arange(7), issue_count)
    hour_index = peak_hour.reshape(-1) + day_index * 24
    peak_actual = actual[issue_index, hour_index]
    peak_samples = samples[issue_index, :, hour_index].reshape(-1, member_count)
    result = metric_bundle(
        peak_samples, peak_actual, interval_levels=interval_levels
    )
    result["peak_count"] = int(peak_actual.size)
    for level in interval_levels:
        label = int(round(100 * level))
        upper = np.quantile(peak_samples, (1.0 + level) / 2.0, axis=1)
        exceedance = np.maximum(peak_actual - upper, 0.0)
        result[f"upper_exceedance_mean_{label}"] = float(np.mean(exceedance))
        result[f"upper_exceedance_mean_when_missed_{label}"] = _mean_or_zero(
            exceedance[exceedance > 0]
        )
    return result


def evaluate_station_scenarios(
    samples: np.ndarray,
    raw_samples: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    stations: pd.DataFrame,
    adjacency: np.ndarray,
    daylight_mask: np.ndarray | None = None,
    interval_levels: tuple[float, ...] = DEFAULT_INTERVAL_LEVELS,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Evaluate arrays shaped samples=[N,K,T,S], actual/forecast=[N,T,S]."""
    if samples.ndim != 4 or actual.ndim != 3:
        raise ValueError("expected samples [N,K,T,S] and actual [N,T,S]")
    if samples.shape[0] != actual.shape[0] or samples.shape[2:] != actual.shape[1:]:
        raise ValueError("scenario and actual shapes do not align")
    if raw_samples.shape != samples.shape or forecast.shape != actual.shape:
        raise ValueError("raw/projected/forecast shapes do not align")
    station_types = stations.data_type.to_numpy()
    capacities = stations.capacity_mw.to_numpy(dtype=np.float64)
    if daylight_mask is not None:
        daylight_mask = np.asarray(daylight_mask, dtype=bool)
        if daylight_mask.shape != actual.shape:
            raise ValueError("daylight_mask must have shape [N,T,S]")

    interval_metric_names = [
        f"{metric}_{int(round(100 * level))}"
        for level in interval_levels
        for metric in ["coverage", "width", "below", "above"]
    ]
    summary_metric_names = [
        "crps",
        *interval_metric_names,
        "scenario_mean_mae",
        "scenario_mean_rmse",
    ]

    station_rows = []
    for station_index, station in stations.iterrows():
        metrics = metric_bundle(
            samples[:, :, :, station_index],
            actual[:, :, station_index],
            interval_levels=interval_levels,
        )
        daylight_metrics = {}
        if station.data_type == "solar" and daylight_mask is not None:
            values = metric_bundle(
                samples[:, :, :, station_index],
                actual[:, :, station_index],
                interval_levels=interval_levels,
                valid_mask=daylight_mask[:, :, station_index],
            )
            daylight_metrics = {
                f"{name}_daylight": value for name, value in values.items()
            }
        station_rows.append(
            {
                "channel_index": int(station.channel_index),
                "station_id": int(station.station_id),
                "station_type": station.data_type,
                "station_name": station.FARM_NAME,
                **metrics,
                **daylight_metrics,
            }
        )
    station_frame = pd.DataFrame(station_rows)

    lead_rows = []
    for station_type in ["wind", "solar", "all"]:
        if station_type == "all":
            station_indices = np.arange(len(stations))
        else:
            station_indices = np.flatnonzero(station_types == station_type)
        for lead_day in range(1, 8):
            hour_slice = slice((lead_day - 1) * 24, lead_day * 24)
            metrics = metric_bundle(
                samples[:, :, hour_slice, :][:, :, :, station_indices],
                actual[:, hour_slice, :][:, :, station_indices],
                interval_levels=interval_levels,
            )
            lead_rows.append(
                {
                    "station_type": station_type,
                    "lead_day": lead_day,
                    **metrics,
                }
            )
    if daylight_mask is not None:
        for lead_day in range(1, 8):
            hour_slice = slice((lead_day - 1) * 24, lead_day * 24)
            station_indices = np.flatnonzero(station_types == "solar")
            metrics = metric_bundle(
                samples[:, :, hour_slice, :][:, :, :, station_indices],
                actual[:, hour_slice, :][:, :, station_indices],
                interval_levels=interval_levels,
                valid_mask=daylight_mask[:, hour_slice, :][:, :, station_indices],
            )
            lead_rows.append(
                {
                    "station_type": "solar_daylight",
                    "lead_day": lead_day,
                    **metrics,
                }
            )
    lead_frame = pd.DataFrame(lead_rows)

    summary: dict[str, object] = {
        "array_shapes": {
            "samples": list(samples.shape),
            "actual": list(actual.shape),
            "forecast": list(forecast.shape),
        },
        "interval_levels": list(interval_levels),
        "station_average": {},
        "aggregate_mw": {},
        "joint": {},
        "ramps": {},
        "extreme_ramps": {},
        "extreme_high_daily_peak_mw": {},
        "physical": {
            "raw_below_zero_rate": float(np.mean(raw_samples < 0)),
            "raw_above_one_rate": float(np.mean(raw_samples > 1)),
            "projected_below_zero_rate": float(np.mean(samples < 0)),
            "projected_above_one_rate": float(np.mean(samples > 1)),
        },
    }
    wind_indices = np.flatnonzero(station_types == "wind")
    solar_indices = np.flatnonzero(station_types == "solar")
    raw_wind = raw_samples[:, :, :, wind_indices]
    summary["physical"].update(
        {
            "raw_wind_below_zero_rate": float(np.mean(raw_wind < 0)),
            "raw_wind_above_one_rate": float(np.mean(raw_wind > 1)),
        }
    )
    if daylight_mask is not None:
        solar_daylight = daylight_mask[:, :, solar_indices]
        raw_solar_by_point = np.moveaxis(
            raw_samples[:, :, :, solar_indices], 1, -1
        )
        raw_solar_daylight = raw_solar_by_point[solar_daylight]
        raw_solar_night = raw_solar_by_point[~solar_daylight]
        actual_solar_night = actual[:, :, solar_indices][~solar_daylight]
        summary["physical"].update(
            {
                "raw_solar_daylight_below_zero_rate": float(
                    np.mean(raw_solar_daylight < 0)
                ),
                "raw_solar_daylight_above_one_rate": float(
                    np.mean(raw_solar_daylight > 1)
                ),
                "raw_solar_night_nonzero_rate": _mean_or_zero(
                    np.abs(raw_solar_night) > 1e-6
                ),
                "raw_solar_night_mean_absolute_pu": _mean_or_zero(
                    np.abs(raw_solar_night)
                ),
                "actual_solar_night_positive_rate": _mean_or_zero(
                    actual_solar_night > 0
                ),
                "actual_solar_night_mean_pu": _mean_or_zero(
                    actual_solar_night
                ),
            }
        )
    for station_type in ["wind", "solar", "all"]:
        subset = (
            station_frame
            if station_type == "all"
            else station_frame.loc[station_frame.station_type.eq(station_type)]
        )
        summary["station_average"][station_type] = {
            name: float(subset[name].mean())
            for name in summary_metric_names
        }

    if daylight_mask is not None:
        summary["station_average"]["solar_daylight"] = metric_bundle(
            samples[:, :, :, solar_indices],
            actual[:, :, solar_indices],
            interval_levels=interval_levels,
            valid_mask=daylight_mask[:, :, solar_indices],
        )

    for station_type, station_indices in [
        ("wind", np.flatnonzero(station_types == "wind")),
        ("solar_daylight", np.flatnonzero(station_types == "solar")),
    ]:
        selected_samples = samples[:, :, :, station_indices]
        selected_actual = actual[:, :, station_indices]
        selected_mask = (
            daylight_mask[:, :, station_indices]
            if station_type == "solar_daylight" and daylight_mask is not None
            else None
        )
        summary["ramps"][station_type] = {}
        summary["extreme_ramps"][station_type] = {}
        for lag in [1, 3, 6]:
            summary["ramps"][station_type][f"lag_{lag}h"] = ramp_metric_bundle(
                selected_samples,
                selected_actual,
                lag,
                interval_levels,
                valid_mask=selected_mask,
            )
            summary["extreme_ramps"][station_type][
                f"lag_{lag}h"
            ] = extreme_ramp_metric_bundle(
                selected_samples,
                selected_actual,
                lag,
                interval_levels,
                valid_mask=selected_mask,
            )

    for station_type in ["wind", "solar", "renewable"]:
        if station_type == "renewable":
            station_indices = np.arange(len(stations))
        else:
            station_indices = np.flatnonzero(station_types == station_type)
        selected_capacity = capacities[station_indices]
        scenario_mw = np.sum(
            samples[:, :, :, station_indices] * selected_capacity[None, None, None, :],
            axis=-1,
        )
        actual_mw = np.sum(
            actual[:, :, station_indices] * selected_capacity[None, None, :], axis=-1
        )
        summary["aggregate_mw"][station_type] = metric_bundle(
            scenario_mw, actual_mw, interval_levels=interval_levels
        )
        if station_type in {"wind", "solar"}:
            peak_label = (
                "solar_daylight" if station_type == "solar" else "wind"
            )
            summary["extreme_high_daily_peak_mw"][peak_label] = (
                daily_peak_metric_bundle(
                    scenario_mw, actual_mw, interval_levels=interval_levels
                )
            )
        if station_type == "solar" and daylight_mask is not None:
            aggregate_daylight = np.any(
                daylight_mask[:, :, station_indices], axis=-1
            )
            summary["aggregate_mw"]["solar_daylight"] = metric_bundle(
                scenario_mw,
                actual_mw,
                interval_levels=interval_levels,
                valid_mask=aggregate_daylight,
            )

    summary["joint"] = {
        "energy_score_pu": energy_score(samples, actual),
        "adjacency_variogram_score": adjacency_variogram_score(
            samples, actual, adjacency
        ),
        **spatial_correlation_metrics(samples, actual, adjacency, station_types),
        **temporal_metrics(samples, actual, station_types),
    }
    return summary, station_frame, lead_frame


def save_evaluation(
    output_dir: str | Path,
    summary: dict[str, object],
    station_frame: pd.DataFrame,
    lead_frame: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    station_frame.to_csv(output_dir / "station_metrics.csv", index=False)
    lead_frame.to_csv(output_dir / "lead_metrics.csv", index=False)
