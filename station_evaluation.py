"""Evaluation metrics for station-level probabilistic wind/solar scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


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


def metric_bundle(samples: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    coverage, width, below, above = interval_metrics(samples, actual, nominal=0.90)
    mae, rmse = point_metrics(samples, actual)
    return {
        "crps": ensemble_crps(samples, actual),
        "coverage_90": coverage,
        "width_90": width,
        "below_90": below,
        "above_90": above,
        "scenario_mean_mae": mae,
        "scenario_mean_rmse": rmse,
    }


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


def evaluate_station_scenarios(
    samples: np.ndarray,
    raw_samples: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    stations: pd.DataFrame,
    adjacency: np.ndarray,
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

    station_rows = []
    for station_index, station in stations.iterrows():
        metrics = metric_bundle(
            samples[:, :, :, station_index], actual[:, :, station_index]
        )
        station_rows.append(
            {
                "channel_index": int(station.channel_index),
                "station_id": int(station.station_id),
                "station_type": station.data_type,
                "station_name": station.FARM_NAME,
                **metrics,
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
            )
            lead_rows.append(
                {
                    "station_type": station_type,
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
        "station_average": {},
        "aggregate_mw": {},
        "joint": {},
        "physical": {
            "raw_below_zero_rate": float(np.mean(raw_samples < 0)),
            "raw_above_one_rate": float(np.mean(raw_samples > 1)),
            "projected_below_zero_rate": float(np.mean(samples < 0)),
            "projected_above_one_rate": float(np.mean(samples > 1)),
        },
    }
    for station_type in ["wind", "solar", "all"]:
        subset = (
            station_frame
            if station_type == "all"
            else station_frame.loc[station_frame.station_type.eq(station_type)]
        )
        summary["station_average"][station_type] = {
            name: float(subset[name].mean())
            for name in [
                "crps",
                "coverage_90",
                "width_90",
                "below_90",
                "above_90",
                "scenario_mean_mae",
                "scenario_mean_rmse",
            ]
        }

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
            scenario_mw, actual_mw
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
