"""Train-only diagnosis and export of Station24 historical spatial priors.

This tool never reads validation/test tensors.  It keeps every statistical source
and every exported adjacency separate so a later model ablation can change only
one historical prior at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.physical_projection import _solar_elevation_degrees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/station24_historical_spatial_prior_diagnostic.yaml",
    )
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--residual-scale-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_type(types: np.ndarray, i: int, j: int) -> str:
    if types[i] == types[j]:
        return f"{types[i]}-{types[j]}"
    return "wind-solar"


def safe_corr(x: np.ndarray, y: np.ndarray, minimum: int, method: str) -> tuple[float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    count = int(np.sum(valid))
    if count < minimum:
        return float("nan"), count
    x = np.asarray(x[valid], dtype=np.float64)
    y = np.asarray(y[valid], dtype=np.float64)
    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError(f"unsupported correlation method={method}")
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-15:
        return 0.0, count
    return float(np.dot(x, y) / denominator), count


def pairwise_correlation(
    values: np.ndarray,
    minimum: int,
    method: str = "pearson",
) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, values.shape[-1])
    stations = flat.shape[-1]
    correlation = np.eye(stations, dtype=np.float64)
    counts = np.zeros((stations, stations), dtype=np.int64)
    for i in range(stations):
        counts[i, i] = int(np.sum(np.isfinite(flat[:, i])))
        for j in range(i + 1, stations):
            value, count = safe_corr(flat[:, i], flat[:, j], minimum, method)
            correlation[i, j] = correlation[j, i] = value
            counts[i, j] = counts[j, i] = count
    return correlation, counts


def issue_target_timestamps(issue_frame: pd.DataFrame, hours: int) -> np.ndarray:
    starts = pd.to_datetime(issue_frame["target_start"]).to_numpy(dtype="datetime64[h]")
    return starts[:, None] + np.arange(hours).astype("timedelta64[h]")[None, :]


def deduplicate_actual_hours(
    actual: np.ndarray,
    valid: np.ndarray,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    flat_time = timestamps.reshape(-1).astype("datetime64[h]")
    unique_time, inverse = np.unique(flat_time, return_inverse=True)
    output = np.full((len(unique_time), actual.shape[-1]), np.nan, dtype=np.float64)
    maximum_duplicate_spread = 0.0
    for station in range(actual.shape[-1]):
        values = np.asarray(actual[..., station], dtype=np.float64).reshape(-1)
        station_valid = np.asarray(valid[..., station], dtype=bool).reshape(-1)
        selected_index = inverse[station_valid]
        selected_values = values[station_valid]
        counts = np.bincount(selected_index, minlength=len(unique_time))
        sums = np.bincount(
            selected_index, weights=selected_values, minlength=len(unique_time)
        )
        present = counts > 0
        output[present, station] = sums[present] / counts[present]
        minima = np.full(len(unique_time), np.inf, dtype=np.float64)
        maxima = np.full(len(unique_time), -np.inf, dtype=np.float64)
        np.minimum.at(minima, selected_index, selected_values)
        np.maximum.at(maxima, selected_index, selected_values)
        maximum_duplicate_spread = max(
            maximum_duplicate_spread,
            float(np.max(maxima[present] - minima[present])),
        )
    return output, unique_time, {
        "issue_lead_observations": int(actual.shape[0] * actual.shape[1]),
        "unique_target_hours": int(len(unique_time)),
        "maximum_duplicate_actual_spread": maximum_duplicate_spread,
    }


def daylight_for_unique_hours(
    unique_time: np.ndarray,
    stations: pd.DataFrame,
    threshold: float,
    offset_minutes: float,
) -> np.ndarray:
    local_zone = timezone(timedelta(hours=8))
    mask = np.ones((len(unique_time), len(stations)), dtype=bool)
    solar = stations.index[stations.data_type.eq("solar")].to_numpy(int)
    for time_index, raw in enumerate(unique_time):
        timestamp = pd.Timestamp(raw).to_pydatetime().replace(tzinfo=local_zone)
        timestamp += timedelta(minutes=offset_minutes)
        for station_index in solar:
            row = stations.iloc[station_index]
            elevation = _solar_elevation_degrees(
                timestamp, float(row.latitude), float(row.longitude)
            )
            mask[time_index, station_index] = elevation > threshold
    return mask


def map_unique_mask_to_issue(
    unique_time: np.ndarray,
    timestamps: np.ndarray,
    unique_mask: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(value.astype("datetime64[h]").astype(np.int64)): index
        for index, value in enumerate(unique_time)
    }
    flat = timestamps.reshape(-1).astype("datetime64[h]").astype(np.int64)
    indices = np.asarray([lookup[int(value)] for value in flat], dtype=np.int64)
    return unique_mask[indices].reshape(timestamps.shape + (unique_mask.shape[-1],))


def previous_issue_indices(issue_frame: pd.DataFrame) -> np.ndarray:
    days = pd.to_datetime(issue_frame["issue_date"]).dt.normalize()
    lookup = {timestamp: index for index, timestamp in enumerate(days)}
    return np.asarray(
        [lookup.get(timestamp - pd.Timedelta(days=1), -1) for timestamp in days],
        dtype=np.int64,
    )


def conditional_standardized_residual(
    forecast: np.ndarray,
    residual: np.ndarray,
    issue_frame: pd.DataFrame,
    scale_specification: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if scale_specification.get("fit_split") != "train":
        raise ValueError("residual scale is not fitted on train")
    if scale_specification.get("method") != "wind_factorized_condition_std":
        raise ValueError("diagnostic requires wind_factorized_condition_std")
    if bool(scale_specification.get("future_actual_used_as_condition", True)):
        raise ValueError("residual scale declares future actual as a condition")
    base = np.asarray(scale_specification["scale"], dtype=np.float64)
    if base.shape != (forecast.shape[-1],) or np.any(base <= 0):
        raise ValueError("invalid station residual scale")
    scale = np.broadcast_to(base[None, None, :], forecast.shape).copy()
    previous = previous_issue_indices(issue_frame)
    revision = np.full_like(forecast, np.nan, dtype=np.float64)
    for index, previous_index in enumerate(previous):
        if previous_index >= 0:
            revision[index, :144] = (
                forecast[index, :144] - forecast[previous_index, 24:]
            )
    ramp_lag = int(scale_specification["ramp_lag"])
    ramp = np.full_like(forecast, np.nan, dtype=np.float64)
    ramp[:, ramp_lag:] = np.abs(
        forecast[:, ramp_lag:] - forecast[:, :-ramp_lag]
    )
    lead_day = np.broadcast_to(
        (np.arange(forecast.shape[1]) // 24 + 1)[None, :, None], forecast.shape
    ).astype(np.float64)
    feature_values = {
        "forecast_level": forecast,
        "lead_day": lead_day,
        "forecast_ramp": ramp,
        "forecast_revision": np.abs(revision),
    }
    multiplier = np.ones_like(forecast, dtype=np.float64)
    for name, values in feature_values.items():
        edges = np.asarray(scale_specification["condition_edges"][name], dtype=np.float64)
        factors = np.asarray(
            scale_specification["condition_factors"][name], dtype=np.float64
        )
        available = np.isfinite(values)
        bins = np.digitize(values[available], edges[1:-1], right=False)
        multiplier[available] *= factors[bins]
    wind = np.asarray(scale_specification["wind_station_indices"], dtype=np.int64)
    scale[..., wind] *= multiplier[..., wind]
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("constructed conditional residual scale is invalid")
    standardized = np.asarray(residual, dtype=np.float64) / scale
    audit = {
        "method": scale_specification["method"],
        "fit_split": scale_specification["fit_split"],
        "previous_issue_available_count": int(np.sum(previous >= 0)),
        "future_actual_used_as_condition": False,
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
    }
    return standardized, scale, audit


def apply_solar_daylight_nan(
    values: np.ndarray,
    valid: np.ndarray,
    daylight: np.ndarray,
    station_types: np.ndarray,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    allowed = np.asarray(valid, dtype=bool).copy()
    solar = np.flatnonzero(station_types == "solar")
    allowed[..., solar] &= daylight[..., solar]
    result[~allowed] = np.nan
    return result


def tail_matrices(
    standardized: np.ndarray,
    station_types: np.ndarray,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stations = standardized.shape[-1]
    low_probability = np.full((stations, stations), np.nan, dtype=np.float64)
    high_probability = np.full((stations, stations), np.nan, dtype=np.float64)
    low_lift = np.full((stations, stations), np.nan, dtype=np.float64)
    high_lift = np.full((stations, stations), np.nan, dtype=np.float64)
    low_threshold = np.full(stations, np.nan, dtype=np.float64)
    high_threshold = np.full(stations, np.nan, dtype=np.float64)
    wind = np.flatnonzero(station_types == "wind")
    flat = standardized.reshape(-1, stations)
    for i in wind:
        values = flat[:, i]
        values = values[np.isfinite(values)]
        low_threshold[i] = float(np.quantile(values, low_quantile))
        high_threshold[i] = float(np.quantile(values, high_quantile))
    for i in wind:
        for j in wind:
            valid = np.isfinite(flat[:, i]) & np.isfinite(flat[:, j])
            left_low = flat[valid, i] < low_threshold[i]
            right_low = flat[valid, j] < low_threshold[j]
            left_high = flat[valid, i] > high_threshold[i]
            right_high = flat[valid, j] > high_threshold[j]
            low_joint = float(np.mean(left_low & right_low))
            high_joint = float(np.mean(left_high & right_high))
            low_denominator = float(np.mean(left_low) * np.mean(right_low))
            high_denominator = float(np.mean(left_high) * np.mean(right_high))
            low_probability[i, j] = low_joint
            high_probability[i, j] = high_joint
            low_lift[i, j] = low_joint / max(low_denominator, 1e-12)
            high_lift[i, j] = high_joint / max(high_denominator, 1e-12)
    return (
        low_probability,
        high_probability,
        low_lift,
        high_lift,
        low_threshold,
        high_threshold,
    )


def lead_day_correlations(
    values: np.ndarray,
    days: list[int],
    minimum: int,
) -> np.ndarray:
    result = []
    for day in days:
        section = values[:, (day - 1) * 24 : day * 24]
        result.append(pairwise_correlation(section, minimum, "pearson")[0])
    return np.stack(result)


def lagged_correlations(
    values: np.ndarray,
    minimum_lag: int,
    maximum_lag: int,
    minimum_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lags = np.arange(minimum_lag, maximum_lag + 1, dtype=np.int64)
    matrices = np.empty((len(lags), values.shape[-1], values.shape[-1]), dtype=np.float64)
    for lag_index, lag in enumerate(lags):
        matrix = np.eye(values.shape[-1], dtype=np.float64)
        for i in range(values.shape[-1]):
            for j in range(values.shape[-1]):
                if i == j:
                    continue
                if lag >= 0:
                    left = values[:, : values.shape[1] - lag if lag else None, i]
                    right = values[:, lag:, j]
                else:
                    left = values[:, -lag:, i]
                    right = values[:, : values.shape[1] + lag, j]
                matrix[i, j] = safe_corr(
                    left.reshape(-1), right.reshape(-1), minimum_observations, "pearson"
                )[0]
        matrices[lag_index] = matrix
    safe = np.where(np.isfinite(matrices), matrices, -np.inf)
    best = np.argmax(safe, axis=0)
    optimal_lag = lags[best]
    optimal_correlation = np.take_along_axis(matrices, best[None], axis=0)[0]
    np.fill_diagonal(optimal_lag, 0)
    np.fill_diagonal(optimal_correlation, 1.0)
    return matrices, optimal_lag, optimal_correlation


def sample_block_indices(length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= block_size <= length:
        raise ValueError("bootstrap block size is outside data length")
    blocks = int(math.ceil(length / block_size))
    maximum_start = length - block_size
    starts = rng.integers(0, maximum_start + 1, size=blocks)
    indices = np.concatenate(
        [np.arange(start, start + block_size, dtype=np.int64) for start in starts]
    )
    return indices[:length]


def bootstrap_correlations(
    values: np.ndarray,
    repetitions: int,
    block_size: int,
    seed: int,
    minimum: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((repetitions, values.shape[-1], values.shape[-1]), dtype=np.float32)
    for repetition in range(repetitions):
        indices = sample_block_indices(values.shape[0], block_size, rng)
        output[repetition] = pairwise_correlation(
            values[indices], minimum, "pearson"
        )[0]
    return output


def graph_from_correlation(
    correlation: np.ndarray,
    ci_lower: np.ndarray,
    positive_probability: np.ndarray,
    station_types: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    weights = np.asarray(correlation, dtype=np.float64).copy()
    np.fill_diagonal(weights, 0.0)
    weights[~np.isfinite(weights)] = 0.0
    if bool(config["positive_only"]):
        weights[weights <= float(config["minimum_weight"])] = 0.0
    stable = (
        (positive_probability >= float(config["stable_positive_probability"]))
        & (ci_lower > float(config["stable_ci_lower_bound"]))
    )
    weights[~stable] = 0.0
    if config["topology_policy"] == "within_type":
        weights[station_types[:, None] != station_types[None, :]] = 0.0
    elif config["topology_policy"] != "all_pairs":
        raise ValueError("unsupported topology policy")
    top_k = int(config["top_k"])
    selected = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        candidates = np.flatnonzero(weights[i] > 0)
        if len(candidates):
            keep = candidates[np.argsort(weights[i, candidates])[-top_k:]]
            selected[i, keep] = weights[i, keep]
    if config["symmetrization"] == "maximum":
        selected = np.maximum(selected, selected.T)
    elif config["symmetrization"] == "mean":
        selected = 0.5 * (selected + selected.T)
    else:
        raise ValueError("unsupported graph symmetrization")
    off_diagonal_edges = int(np.sum(np.triu(selected > 0, 1)))
    np.fill_diagonal(selected, float(config["self_loop_weight"]))
    if config["normalization"] == "symmetric_degree":
        degree = selected.sum(axis=1)
        inverse = np.where(degree > 0, degree ** -0.5, 0.0)
        selected = inverse[:, None] * selected * inverse[None, :]
    elif config["normalization"] != "none":
        raise ValueError("unsupported graph normalization")
    return selected.astype(np.float32), {
        "off_diagonal_undirected_edge_count": off_diagonal_edges,
        "top_k_before_symmetrization": top_k,
        "topology_policy": config["topology_policy"],
        "normalization": config["normalization"],
    }


def graph_from_tail_lift(
    lift: np.ndarray,
    station_types: np.ndarray,
    config: dict[str, object],
    minimum_lift: float,
) -> tuple[np.ndarray, dict[str, object]]:
    correlation_like = np.where(np.isfinite(lift), np.maximum(np.log(lift), 0.0), 0.0)
    correlation_like[lift <= minimum_lift] = 0.0
    wind_pair = (station_types[:, None] == "wind") & (station_types[None, :] == "wind")
    correlation_like[~wind_pair] = 0.0
    stable = correlation_like > 0
    return graph_from_correlation(
        correlation_like,
        np.where(stable, 1.0, -1.0),
        np.where(stable, 1.0, 0.0),
        station_types,
        config,
    )


def matrix_group_summary(matrix: np.ndarray, station_types: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for label in ["wind-wind", "solar-solar", "wind-solar"]:
        values = []
        for i in range(len(station_types)):
            for j in range(i + 1, len(station_types)):
                if pair_type(station_types, i, j) == label and np.isfinite(matrix[i, j]):
                    values.append(float(matrix[i, j]))
        array = np.asarray(values, dtype=np.float64)
        result[label] = {
            "pair_count": int(len(array)),
            "mean": float(np.mean(array)),
            "mean_absolute": float(np.mean(np.abs(array))),
            "median": float(np.median(array)),
            "positive_fraction": float(np.mean(array > 0)),
            "negative_fraction": float(np.mean(array < 0)),
        }
    return result


def legacy_residual_correlation(
    residual: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray,
    station_types: np.ndarray,
    threshold: float,
) -> np.ndarray:
    matrix = np.eye(residual.shape[-1], dtype=np.float64)
    for i in range(residual.shape[-1]):
        for j in range(i + 1, residual.shape[-1]):
            mask = np.ones(residual.shape[:2], dtype=bool)
            if station_types[i] == "solar":
                mask &= np.maximum(forecast[..., i], actual[..., i]) > threshold
            if station_types[j] == "solar":
                mask &= np.maximum(forecast[..., j], actual[..., j]) > threshold
            value = safe_corr(residual[..., i][mask], residual[..., j][mask], 2, "pearson")[0]
            matrix[i, j] = matrix[j, i] = value
    return matrix


def reference_audit(
    legacy_raw: np.ndarray,
    raw_residual: np.ndarray,
    tail_low_lift: np.ndarray,
    lead_raw: np.ndarray,
    geo: np.ndarray,
    station_types: np.ndarray,
    config: dict[str, object],
) -> dict[str, object]:
    summary = matrix_group_summary(legacy_raw, station_types)
    upper = np.triu_indices(len(station_types), 1)
    edge = np.triu(geo, 1) > 0
    wind = np.flatnonzero(station_types == "wind")
    wind_pairs = np.triu(np.equal.outer(station_types, "wind"), 1)
    day_vectors = [matrix[wind_pairs] for matrix in lead_raw]
    pattern = []
    for i in range(len(day_vectors)):
        for j in range(i + 1, len(day_vectors)):
            pattern.append(np.corrcoef(day_vectors[i], day_vectors[j])[0, 1])
    observed = {
        "wind_wind_raw_residual_mean": summary["wind-wind"]["mean"],
        "solar_solar_raw_residual_mean": summary["solar-solar"]["mean"],
        "wind_solar_raw_residual_mean": summary["wind-solar"]["mean"],
        "geo_edge_absolute_raw_residual_mean": float(np.mean(np.abs(legacy_raw[edge]))),
        "non_geo_edge_absolute_raw_residual_mean": float(
            np.mean(np.abs(legacy_raw[upper][~edge[upper]]))
        ),
        "wind_low_tail_lift_mean": float(
            np.nanmean(tail_low_lift[np.ix_(wind, wind)][np.triu_indices(len(wind), 1)])
        ),
        "wind_day1_absolute_residual_mean": float(np.mean(np.abs(lead_raw[0][wind_pairs]))),
        "wind_day7_absolute_residual_mean": float(np.mean(np.abs(lead_raw[-1][wind_pairs]))),
        "lead_day_edge_pattern_mean_correlation": float(np.mean(pattern)),
    }
    expected = config["expected"]
    tolerances = {
        "wind_wind_raw_residual_mean": float(config["correlation_tolerance"]),
        "solar_solar_raw_residual_mean": float(config["correlation_tolerance"]),
        "wind_solar_raw_residual_mean": float(config["correlation_tolerance"]),
        "geo_edge_absolute_raw_residual_mean": float(config["geo_edge_correlation_tolerance"]),
        "non_geo_edge_absolute_raw_residual_mean": float(config["geo_edge_correlation_tolerance"]),
        "wind_low_tail_lift_mean": float(config["tail_lift_tolerance"]),
        "wind_day1_absolute_residual_mean": float(config["lead_day_correlation_tolerance"]),
        "wind_day7_absolute_residual_mean": float(config["lead_day_correlation_tolerance"]),
        "lead_day_edge_pattern_mean_correlation": float(config["lead_day_correlation_tolerance"]),
    }
    checks = {}
    for key, reference in expected.items():
        delta = observed[key] - float(reference)
        checks[key] = {
            "observed": observed[key],
            "expected": float(reference),
            "absolute_delta": abs(delta),
            "tolerance": tolerances[key],
            "passed": abs(delta) <= tolerances[key],
        }
    return {"observed": observed, "checks": checks, "passed": all(v["passed"] for v in checks.values())}


def plot_heatmaps(
    output: Path,
    matrices: list[tuple[str, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for axis, (title, matrix) in zip(axes.flat, matrices):
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("Station channel")
        axis.set_ylabel("Station channel")
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_geo_scatter(
    output: Path,
    pair_frame: pd.DataFrame,
) -> None:
    colors = {"wind-wind": "#2878B5", "solar-solar": "#E76F51", "wind-solar": "#2A9D8F"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for label, group in pair_frame.groupby("pair_type"):
        axes[0].scatter(group.distance_km, group.residual_std_pearson, s=22, alpha=0.7, label=label, color=colors[label])
        axes[1].scatter(group.geo_weight, group.residual_std_pearson, s=22, alpha=0.7, label=label, color=colors[label])
    axes[0].set(xlabel="Geographic distance (km)", ylabel="Standardized residual correlation", title="Distance versus residual relation")
    axes[1].set(xlabel="Geographic adjacency weight", ylabel="Standardized residual correlation", title="Geographic graph versus historical relation")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_lead_stability(
    output: Path,
    lead_matrices: np.ndarray,
    station_types: np.ndarray,
) -> None:
    wind_pairs = np.triu(np.equal.outer(station_types, "wind"), 1)
    solar_pairs = np.triu(np.equal.outer(station_types, "solar"), 1)
    rows = []
    for day, matrix in enumerate(lead_matrices, start=1):
        rows.append((day, np.mean(np.abs(matrix[wind_pairs])), np.mean(np.abs(matrix[solar_pairs]))))
    frame = pd.DataFrame(rows, columns=["lead_day", "wind", "solar"])
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(frame.lead_day, frame.wind, marker="o", label="wind-wind")
    axis.plot(frame.lead_day, frame.solar, marker="o", label="solar-solar")
    axis.set(xlabel="Lead day", ylabel="Mean absolute standardized residual correlation", title="Lead-day historical edge strength")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["data"]["split"] != "train":
        raise SystemExit("historical prior diagnostic is train-only")
    if config["data"]["allow_validation_actual"] or config["data"]["allow_test_actual"]:
        raise SystemExit("validation/test actual access must remain disabled")
    data_path = Path(args.data_path or config["data"]["data_path"])
    scale_path = Path(args.residual_scale_path or config["data"]["residual_scale_path"])
    if not scale_path.is_file():
        raise FileNotFoundError(f"residual scale not found: {scale_path}")
    output_root = Path(config["output"]["root"])
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"{config['output']['run_name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    required = [
        "train_forecast.npy", "train_actual.npy", "train_residual.npy",
        "train_fill_mask.npy", "train_issue_dates.csv", "station_order.csv",
        "station_distance.npy", "station_adjacency.npy", "export_metadata.json",
    ]
    missing = [name for name in required if not (data_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing train diagnostic artifacts: {missing}")
    loaded_files = {name: sha256_file(data_path / name) for name in required}
    forecast = np.asarray(np.load(data_path / "train_forecast.npy", mmap_mode="r"), dtype=np.float64)
    actual = np.asarray(np.load(data_path / "train_actual.npy", mmap_mode="r"), dtype=np.float64)
    residual = np.asarray(np.load(data_path / "train_residual.npy", mmap_mode="r"), dtype=np.float64)
    fill_mask = np.asarray(np.load(data_path / "train_fill_mask.npy", mmap_mode="r"))
    if forecast.shape != (290, 168, 24) or actual.shape != forecast.shape:
        raise ValueError(f"unexpected train shape {forecast.shape}")
    if not np.allclose(residual, actual - forecast, atol=1e-6):
        raise ValueError("train residual is not actual minus forecast")
    valid = fill_mask == 0
    stations = pd.read_csv(data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    station_types = stations.data_type.to_numpy()
    distance = np.load(data_path / "station_distance.npy")
    geo = np.load(data_path / "station_adjacency.npy")
    issue_frame = pd.read_csv(data_path / "train_issue_dates.csv")
    timestamps = issue_target_timestamps(issue_frame, forecast.shape[1])

    actual_unique, unique_time, dedup_audit = deduplicate_actual_hours(actual, valid, timestamps)
    daylight_unique = daylight_for_unique_hours(
        unique_time,
        stations,
        float(config["correlation"]["solar_elevation_threshold_deg"]),
        float(config["correlation"]["timestamp_offset_minutes"]),
    )
    daylight_issue = map_unique_mask_to_issue(unique_time, timestamps, daylight_unique)
    actual_unique_mask = np.isfinite(actual_unique)
    solar = np.flatnonzero(station_types == "solar")
    actual_unique_mask[:, solar] &= daylight_unique[:, solar]
    actual_for_corr = actual_unique.copy()
    actual_for_corr[~actual_unique_mask] = np.nan

    scale_specification = json.loads(scale_path.read_text(encoding="utf-8"))
    standardized, scale_tensor, scale_audit = conditional_standardized_residual(
        forecast, residual, issue_frame, scale_specification
    )
    raw_for_corr = apply_solar_daylight_nan(residual, valid, daylight_issue, station_types)
    standardized_for_corr = apply_solar_daylight_nan(standardized, valid, daylight_issue, station_types)
    forecast_for_corr = apply_solar_daylight_nan(forecast, valid, daylight_issue, station_types)
    minimum = int(config["correlation"]["minimum_pair_observations"])

    actual_pearson, actual_counts = pairwise_correlation(actual_for_corr, minimum, "pearson")
    actual_spearman, _ = pairwise_correlation(actual_for_corr, minimum, "spearman")
    forecast_pearson, forecast_counts = pairwise_correlation(forecast_for_corr, minimum, "pearson")
    forecast_spearman, _ = pairwise_correlation(forecast_for_corr, minimum, "spearman")
    raw_pearson, raw_counts = pairwise_correlation(raw_for_corr, minimum, "pearson")
    raw_spearman, _ = pairwise_correlation(raw_for_corr, minimum, "spearman")
    standardized_pearson, standardized_counts = pairwise_correlation(standardized_for_corr, minimum, "pearson")
    standardized_spearman, _ = pairwise_correlation(standardized_for_corr, minimum, "spearman")

    tail = tail_matrices(
        np.where(valid, standardized, np.nan),
        station_types,
        float(config["tail"]["low_quantile"]),
        float(config["tail"]["high_quantile"]),
    )
    low_probability, high_probability, low_lift, high_lift, low_threshold, high_threshold = tail
    days = [int(value) for value in config["lead_day"]["days"]]
    lead_std = lead_day_correlations(standardized_for_corr, days, minimum)
    lead_raw = lead_day_correlations(raw_for_corr, days, minimum)
    lagged, optimal_lag, optimal_lag_correlation = lagged_correlations(
        standardized_for_corr,
        int(config["lag"]["minimum_hours"]),
        int(config["lag"]["maximum_hours"]),
        minimum,
    )

    repetitions = int(args.bootstrap_repetitions or config["bootstrap"]["repetitions"])
    actual_bootstrap = bootstrap_correlations(
        actual_for_corr,
        repetitions,
        int(config["bootstrap"]["calendar_day_block_size"]) * 24,
        int(config["bootstrap"]["seed"]),
        minimum,
    )
    residual_bootstrap = bootstrap_correlations(
        standardized_for_corr,
        repetitions,
        int(config["bootstrap"]["issue_block_size"]),
        int(config["bootstrap"]["seed"]) + 1,
        minimum,
    )
    alpha = (1.0 - float(config["bootstrap"]["confidence_level"])) / 2.0
    actual_lower = np.quantile(actual_bootstrap, alpha, axis=0)
    actual_upper = np.quantile(actual_bootstrap, 1.0 - alpha, axis=0)
    actual_positive = np.mean(actual_bootstrap > 0, axis=0)
    residual_lower = np.quantile(residual_bootstrap, alpha, axis=0)
    residual_upper = np.quantile(residual_bootstrap, 1.0 - alpha, axis=0)
    residual_positive = np.mean(residual_bootstrap > 0, axis=0)

    graph_config = {
        **config["graph"],
        "stable_positive_probability": config["bootstrap"]["stable_positive_probability"],
        "stable_ci_lower_bound": config["bootstrap"]["stable_ci_lower_bound"],
    }
    adjacency_actual, actual_graph_audit = graph_from_correlation(
        actual_pearson, actual_lower, actual_positive, station_types, graph_config
    )
    adjacency_residual, residual_graph_audit = graph_from_correlation(
        standardized_pearson, residual_lower, residual_positive, station_types, graph_config
    )
    adjacency_tail, tail_graph_audit = graph_from_tail_lift(
        low_lift, station_types, graph_config, float(config["tail"]["minimum_lift"])
    )

    pair_rows = []
    bootstrap_rows = []
    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            label = pair_type(station_types, i, j)
            pair_rows.append({
                "channel_i": i,
                "channel_j": j,
                "station_id_i": int(stations.iloc[i].station_id),
                "station_id_j": int(stations.iloc[j].station_id),
                "station_name_i": stations.iloc[i].FARM_NAME,
                "station_name_j": stations.iloc[j].FARM_NAME,
                "pair_type": label,
                "distance_km": float(distance[i, j]),
                "geo_weight": float(geo[i, j]),
                "geo_edge": bool(geo[i, j] > 0),
                "actual_pearson": actual_pearson[i, j],
                "actual_spearman": actual_spearman[i, j],
                "actual_observations": int(actual_counts[i, j]),
                "forecast_pearson": forecast_pearson[i, j],
                "forecast_spearman": forecast_spearman[i, j],
                "forecast_observations": int(forecast_counts[i, j]),
                "residual_raw_pearson": raw_pearson[i, j],
                "residual_raw_spearman": raw_spearman[i, j],
                "residual_raw_observations": int(raw_counts[i, j]),
                "residual_std_pearson": standardized_pearson[i, j],
                "residual_std_spearman": standardized_spearman[i, j],
                "residual_std_observations": int(standardized_counts[i, j]),
                "low_tail_joint_probability": low_probability[i, j],
                "low_tail_lift": low_lift[i, j],
                "high_tail_joint_probability": high_probability[i, j],
                "high_tail_lift": high_lift[i, j],
                "optimal_lag_hours_i_to_j": int(optimal_lag[i, j]),
                "optimal_lag_correlation_i_to_j": float(optimal_lag_correlation[i, j]),
                **{f"residual_std_day{day}_pearson": lead_std[index, i, j] for index, day in enumerate(days)},
            })
            for source, point, lower, upper, positive in [
                ("actual_unique_power", actual_pearson, actual_lower, actual_upper, actual_positive),
                ("condition_standardized_residual", standardized_pearson, residual_lower, residual_upper, residual_positive),
            ]:
                bootstrap_rows.append({
                    "channel_i": i,
                    "channel_j": j,
                    "station_id_i": int(stations.iloc[i].station_id),
                    "station_id_j": int(stations.iloc[j].station_id),
                    "pair_type": label,
                    "source": source,
                    "point_pearson": point[i, j],
                    "ci_lower": lower[i, j],
                    "ci_upper": upper[i, j],
                    "positive_probability": positive[i, j],
                    "stable_positive": bool(
                        positive[i, j] >= float(config["bootstrap"]["stable_positive_probability"])
                        and lower[i, j] > float(config["bootstrap"]["stable_ci_lower_bound"])
                    ),
                    "bootstrap_repetitions": repetitions,
                })
    pair_frame = pd.DataFrame(pair_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)

    legacy = legacy_residual_correlation(
        residual,
        forecast,
        actual,
        station_types,
        float(config["reference_audit"]["legacy_solar_active_threshold_pu"]),
    )
    audit = reference_audit(
        legacy,
        raw_for_corr,
        low_lift,
        lead_raw,
        geo,
        station_types,
        config["reference_audit"],
    )

    matrices = {
        "adjacency_geo.npy": geo,
        "adjacency_actual.npy": adjacency_actual,
        "adjacency_residual_std.npy": adjacency_residual,
        "adjacency_tail_low.npy": adjacency_tail,
        "correlation_actual_pearson.npy": actual_pearson,
        "correlation_actual_spearman.npy": actual_spearman,
        "correlation_forecast_pearson.npy": forecast_pearson,
        "correlation_forecast_spearman.npy": forecast_spearman,
        "correlation_residual_raw_pearson.npy": raw_pearson,
        "correlation_residual_raw_spearman.npy": raw_spearman,
        "correlation_residual_std_pearson.npy": standardized_pearson,
        "correlation_residual_std_spearman.npy": standardized_spearman,
        "lead_day_residual_correlation.npy": lead_std,
        "lagged_residual_correlation.npy": lagged,
        "optimal_lag_hours.npy": optimal_lag,
        "optimal_lag_correlation.npy": optimal_lag_correlation,
        "tail_low_joint_probability.npy": low_probability,
        "tail_low_lift.npy": low_lift,
        "tail_high_joint_probability.npy": high_probability,
        "tail_high_lift.npy": high_lift,
    }
    for name, matrix in matrices.items():
        np.save(output_dir / name, matrix)
    pair_frame.to_csv(output_dir / "historical_spatial_pair_metrics.csv", index=False)
    bootstrap_frame.to_csv(output_dir / "historical_spatial_bootstrap_stability.csv", index=False)
    pd.DataFrame({
        "channel_index": np.arange(len(stations)),
        "station_id": stations.station_id,
        "station_type": station_types,
        "low_tail_threshold_standardized": low_threshold,
        "high_tail_threshold_standardized": high_threshold,
    }).to_csv(output_dir / "historical_tail_thresholds.csv", index=False)

    shutil.copy2(config_path, output_dir / "diagnostic_config_used.yaml")
    shutil.copy2(scale_path, output_dir / "residual_scale_used.json")
    plot_heatmaps(
        output_dir / "historical_spatial_heatmaps.png",
        [
            ("Geographic adjacency", geo),
            ("Actual power Pearson", actual_pearson),
            ("Forecast Pearson", forecast_pearson),
            ("Raw residual Pearson", raw_pearson),
            ("Standardized residual Pearson", standardized_pearson),
            ("Lead day 1 residual", lead_std[0]),
            ("Lead day 7 residual", lead_std[-1]),
            ("Optimal lag correlation", optimal_lag_correlation),
        ],
    )
    plot_geo_scatter(output_dir / "geo_vs_history_scatter.png", pair_frame)
    plot_lead_stability(output_dir / "lead_day_edge_stability.png", lead_std, station_types)

    summary = {
        "experiment": config["experiment"],
        "train_only": True,
        "validation_actual_used": False,
        "test_actual_used": False,
        "data_shapes": {
            "forecast": list(forecast.shape),
            "actual": list(actual.shape),
            "residual": list(residual.shape),
        },
        "actual_deduplication": dedup_audit,
        "residual_scale": {
            **scale_audit,
            "source": str(scale_path),
            "sha256": sha256_file(scale_path),
        },
        "correlation_summary": {
            "actual": matrix_group_summary(actual_pearson, station_types),
            "forecast": matrix_group_summary(forecast_pearson, station_types),
            "residual_raw": matrix_group_summary(raw_pearson, station_types),
            "residual_standardized": matrix_group_summary(standardized_pearson, station_types),
        },
        "graph_audit": {
            "geographic": {"source": str(data_path / "station_adjacency.npy")},
            "historical_actual": actual_graph_audit,
            "historical_residual_std": residual_graph_audit,
            "historical_tail_low": tail_graph_audit,
        },
        "reference_reproduction": audit,
        "bootstrap": {
            **config["bootstrap"],
            "repetitions": repetitions,
        },
        "loaded_train_files_sha256": loaded_files,
    }
    (output_dir / "historical_spatial_prior_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    actual_summary = summary["correlation_summary"]["actual"]
    residual_summary = summary["correlation_summary"]["residual_standardized"]
    report = [
        "# Station24历史空间先验训练集诊断",
        "",
        "## 审计结论",
        "",
        "- 本实验只读取训练集；验证集和测试集实测均未使用。",
        f"- 训练发布样本：{forecast.shape[0]}；实际功率去重后目标小时：{dedup_audit['unique_target_hours']}。",
        f"- 条件标准化残差尺度：`{scale_specification['method']}`。",
        f"- 既有统计复核：{'通过' if audit['passed'] else '未通过'}。",
        "",
        "## 分块Pearson相关",
        "",
        "| 数据源 | 风—风均值 | 光—光均值 | 风—光均值 |",
        "|---|---:|---:|---:|",
        f"| 历史实际功率 | {actual_summary['wind-wind']['mean']:.3f} | {actual_summary['solar-solar']['mean']:.3f} | {actual_summary['wind-solar']['mean']:.3f} |",
        f"| 条件标准化残差 | {residual_summary['wind-wind']['mean']:.3f} | {residual_summary['solar-solar']['mean']:.3f} | {residual_summary['wind-solar']['mean']:.3f} |",
        "",
        "## 独立候选输出",
        "",
        "- `adjacency_actual.npy`：仅供后续 `geo_history_actual_dual` 使用。",
        "- `adjacency_residual_std.npy`：仅供后续 `geo_history_residual_dual` 使用。",
        "- `adjacency_tail_low.npy`：只做后续尾部研究储备，本轮双图训练不得使用。",
        "- 原始地理图未被覆盖。",
        "",
        "图、逐对指标、bootstrap区间和全部矩阵见同目录文件。",
        "",
    ]
    (output_dir / "historical_spatial_prior_summary.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"HISTORICAL_SPATIAL_DIAGNOSTIC_COMPLETE output_dir={output_dir}")
    print(f"REFERENCE_AUDIT_PASSED={audit['passed']}")
    if bool(config["reference_audit"]["fail_on_large_mismatch"]) and not audit["passed"]:
        raise SystemExit("reference audit mismatch; inspect outputs before training")


if __name__ == "__main__":
    main()
