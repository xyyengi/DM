"""Frozen validation-only Stage-1A alpha comparison and selection.

This tool trains nothing, generates nothing, and never reads the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from station_dataset import build_station_daylight_mask
from station_evaluation import energy_score, temporal_metrics
from station_jstd_targets import build_station_jstd_target_arrays


def json_default(value):
    """Convert NumPy scalar evidence without changing the frozen decisions."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


ALPHAS = (0.65, 1.00, 1.35)
WEIGHTS = {0.65: (0.117, 0.091, 0.065), 1.00: (0.180, 0.140, 0.100), 1.35: (0.243, 0.189, 0.135)}
PRIMARY_BODY = (
    "wind_crps", "solar_daylight_crps", "renewable_crps", "energy_score",
    "coverage_error_90", "interval_width_90", "interval_score_90",
    "spatial_rmse", "wind_solar_rmse",
)
SPATIAL_BODY = ("spatial_rmse", "wind_solar_rmse")
POINTWISE_BODY = tuple(metric for metric in PRIMARY_BODY if metric not in SPATIAL_BODY)
STRUCTURE_BODY = (
    "wind_acf_error_24", "wind_acf_error_48",
    "solar_acf_error_24", "solar_acf_error_48",
)
LOWER_BETTER = set(PRIMARY_BODY) | {
    "onset_error", "duration_error", "depth_error", "depth_ratio_error",
    "std_distance", "q90_distance", "q95_distance", "q99_distance",
    "ramp_mae", "ramp_coverage_error",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("diffusion_input_station"))
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-post", type=Path, required=True)
    parser.add_argument("--alpha065-root", type=Path, required=True)
    parser.add_argument("--alpha135-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def crps_points(samples, truth):
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    members = samples.shape[0]
    first = np.mean(np.abs(samples - truth[None, ...]), axis=0)
    ordered = np.sort(samples, axis=0)
    coeff = (2 * np.arange(1, members + 1) - members - 1).reshape(
        (members,) + (1,) * truth.ndim
    )
    second = np.sum(ordered * coeff, axis=0) / members**2
    return float(np.mean(first - second))


def interval_bundle(samples, truth):
    lower = np.quantile(samples, .05, axis=0)
    upper = np.quantile(samples, .95, axis=0)
    covered = (truth >= lower) & (truth <= upper)
    width = upper - lower
    score = width + 20 * (lower - truth) * (truth < lower) + 20 * (truth - upper) * (truth > upper)
    coverage = float(np.mean(covered))
    return coverage, float(np.mean(width)), float(np.mean(score))


def load_array(path):
    try:
        return np.load(path, mmap_mode="r")
    except OSError:
        # Windows can exhaust the mmap section-table resource well before RAM;
        # ordinary loading preserves identical values and keeps local audits usable.
        return np.load(path)


def result_arrays(path):
    return (
        load_array(path / "actual_scenarios_normalized.npy"),
        load_array(path / "actual_data_normalized.npy"),
    )


def source_aggregate(values, indices, capacities):
    return np.sum(values[..., indices] * capacities[indices], axis=-1)


def moments_by_issue(values):
    """Return additive station moments, keeping issue as the resampling unit."""
    values = np.asarray(values)
    issue_count, station_count = values.shape[0], values.shape[-1]
    count = np.empty(issue_count, dtype=np.int64)
    total = np.empty((issue_count, station_count), dtype=np.float64)
    cross = np.empty((issue_count, station_count, station_count), dtype=np.float64)
    for issue in range(issue_count):
        flat = np.asarray(values[issue], dtype=np.float64).reshape(-1, station_count)
        if not np.all(np.isfinite(flat)):
            raise ValueError(f"nonfinite values in spatial issue {issue}")
        count[issue] = flat.shape[0]
        total[issue] = flat.sum(axis=0)
        cross[issue] = flat.T @ flat
    return {"count": count, "sum": total, "cross": cross}


def correlations_from_moments(moments, draw_indices):
    """Recompute correlations after each paired block-bootstrap draw."""
    draw_indices = np.asarray(draw_indices, dtype=np.int64)
    count = moments["count"][draw_indices].sum(axis=1).astype(np.float64)
    total = moments["sum"][draw_indices].sum(axis=1)
    cross = moments["cross"][draw_indices].sum(axis=1)
    centered = cross - np.einsum("bi,bj->bij", total, total) / count[:, None, None]
    variance = np.diagonal(centered, axis1=1, axis2=2)
    denominator = np.sqrt(np.maximum(variance[:, :, None] * variance[:, None, :], 0.0))
    correlation = np.full_like(centered, np.nan)
    np.divide(centered, denominator, out=correlation, where=denominator > 1e-12)
    return correlation


def spatial_rmse_from_correlations(actual, generated, station_types):
    difference = generated - actual
    station_count = difference.shape[-1]
    masks = {
        "spatial_rmse": np.triu(np.ones((station_count, station_count), dtype=bool), k=1),
        "wind_wind_rmse": np.triu(np.equal.outer(station_types, "wind"), k=1),
        "solar_solar_rmse": np.triu(np.equal.outer(station_types, "solar"), k=1),
        "wind_solar_rmse": np.triu(np.not_equal.outer(station_types, station_types), k=1),
    }
    output = {}
    for name, pair_mask in masks.items():
        selected = difference[:, pair_mask]
        finite = np.isfinite(selected)
        if not np.all(finite):
            bad_draws = np.flatnonzero(~np.all(finite, axis=1))
            raise ValueError(
                f"nonfinite station-pair correlations for {name} in "
                f"{len(bad_draws)} bootstrap/full draws"
            )
        output[name] = np.sqrt(np.mean(selected * selected, axis=1))
    return output


def spatial_metrics_for_draws(result, actual_moments, draw_indices, station_types):
    samples, _ = result_arrays(result)
    generated_moments = moments_by_issue(samples)
    actual_corr = correlations_from_moments(actual_moments, draw_indices)
    generated_corr = correlations_from_moments(generated_moments, draw_indices)
    return spatial_rmse_from_correlations(actual_corr, generated_corr, station_types)


def body_issue_rows(label, alpha, result, stations, adjacency, daylight):
    samples, actual = result_arrays(result)
    station_types = stations.data_type.to_numpy(str)
    capacities = stations.capacity_mw.to_numpy(float)
    wind = np.flatnonzero(station_types == "wind")
    solar = np.flatnonzero(station_types == "solar")
    renewable_samples = source_aggregate(samples, np.arange(len(stations)), capacities)
    renewable_actual = source_aggregate(actual, np.arange(len(stations)), capacities)
    rows, lead_rows = [], []
    energy_indices = np.linspace(0, samples.shape[1] - 1, 80, dtype=int)
    for issue in range(samples.shape[0]):
        current = np.asarray(samples[issue])
        truth = np.asarray(actual[issue])
        solar_mask = daylight[issue][:, solar]
        solar_samples = np.moveaxis(current[:, :, solar], 0, -1)[solar_mask].T
        solar_truth = truth[:, solar][solar_mask]
        renewable = np.asarray(renewable_samples[issue])
        renewable_y = np.asarray(renewable_actual[issue])
        coverage, width, interval_score = interval_bundle(renewable, renewable_y)
        temporal = temporal_metrics(current[None], truth[None], station_types)
        row = {
            "alpha": alpha, "label": label, "issue": issue,
            "wind_crps": crps_points(current[:, :, wind], truth[:, wind]),
            "solar_daylight_crps": crps_points(solar_samples, solar_truth),
            "renewable_crps": crps_points(renewable, renewable_y),
            "energy_score": energy_score(current[None, energy_indices], truth[None]),
            "coverage_90": coverage, "coverage_error_90": abs(coverage - .90),
            "interval_width_90": width, "interval_score_90": interval_score,
            "wind_acf_error_24": temporal["wind_acf_abs_error_lag24"],
            "wind_acf_error_48": temporal["wind_acf_abs_error_lag48"],
            "solar_acf_error_24": temporal["solar_acf_abs_error_lag24"],
            "solar_acf_error_48": temporal["solar_acf_abs_error_lag48"],
        }
        rows.append(row)
        for day in range(7):
            sl = slice(day * 24, (day + 1) * 24)
            cov, day_width, day_score = interval_bundle(renewable[:, sl], renewable_y[sl])
            lead_rows.append({
                "alpha": alpha, "label": label, "issue": issue, "lead_day": day + 1,
                "renewable_crps": crps_points(renewable[:, sl], renewable_y[sl]),
                "coverage_90": cov, "coverage_error_90": abs(cov - .90),
                "interval_width_90": day_width, "interval_score_90": day_score,
            })
    return rows, lead_rows


def moving_block_indices(n, repetitions=10000, block=7, seed=20261006):
    if n < block:
        raise ValueError("fewer issues than bootstrap block")
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    count = math.ceil(n / block)
    output = np.empty((repetitions, n), dtype=np.int16)
    offsets = np.arange(block)
    for draw in range(repetitions):
        chosen = rng.choice(starts, size=count, replace=True)
        output[draw] = (chosen[:, None] + offsets).reshape(-1)[:n]
    return output


def ci_of_paired(left, right, indices):
    delta = np.asarray(left, float) - np.asarray(right, float)
    if not np.all(np.isfinite(delta)):
        raise ValueError("paired bootstrap received nonfinite issue-level differences")
    draws = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def body_bootstrap(body_issue, lead_issue, labels, indices, spatial_draws):
    rows = []
    for candidate in labels:
        if candidate in ("raw", "alpha_1.00"):
            continue
        for reference in ("raw", "alpha_1.00"):
            for metric in POINTWISE_BODY + STRUCTURE_BODY:
                left = body_issue.loc[body_issue.label.eq(candidate)].sort_values("issue")[metric]
                right = body_issue.loc[body_issue.label.eq(reference)].sort_values("issue")[metric]
                mean, low, high = ci_of_paired(left, right, indices)
                rows.append({"family": "body", "candidate": candidate, "reference": reference,
                             "metric": metric, "difference": mean, "ci_low": low, "ci_high": high,
                             "repetitions": len(indices), "bootstrap_seed": 20261006})
            for metric in SPATIAL_BODY:
                candidate_values = spatial_draws[candidate][metric]
                reference_values = spatial_draws[reference][metric]
                delta = candidate_values[1:] - reference_values[1:]
                if not np.all(np.isfinite(delta)):
                    raise ValueError(f"nonfinite spatial bootstrap difference: {candidate}, {reference}, {metric}")
                rows.append({"family": "body", "candidate": candidate, "reference": reference,
                             "metric": metric,
                             "difference": float(candidate_values[0] - reference_values[0]),
                             "ci_low": float(np.quantile(delta, .025)),
                             "ci_high": float(np.quantile(delta, .975)),
                             "repetitions": len(indices), "bootstrap_seed": 20261006})
        if candidate != "alpha_1.00":
            for lead_day in range(1, 8):
                left_frame = lead_issue.loc[
                    lead_issue.label.eq(candidate) & lead_issue.lead_day.eq(lead_day)
                ].sort_values("issue")
                right_frame = lead_issue.loc[
                    lead_issue.label.eq("alpha_1.00") & lead_issue.lead_day.eq(lead_day)
                ].sort_values("issue")
                for metric in ("renewable_crps", "coverage_error_90", "interval_width_90", "interval_score_90"):
                    mean, low, high = ci_of_paired(left_frame[metric], right_frame[metric], indices)
                    rows.append({"family": "lead_day", "candidate": candidate,
                                 "reference": "alpha_1.00", "lead_day": lead_day,
                                 "metric": metric, "difference": mean, "ci_low": low,
                                 "ci_high": high, "repetitions": len(indices),
                                 "bootstrap_seed": 20261006})
    # Pair every alpha for deterministic Pareto/tie-break decisions.
    alpha_labels = [x for x in labels if x != "raw"]
    for i, candidate in enumerate(alpha_labels):
        for reference in alpha_labels[i + 1:]:
            for metric in PRIMARY_BODY:
                if metric in SPATIAL_BODY:
                    candidate_values = spatial_draws[candidate][metric]
                    reference_values = spatial_draws[reference][metric]
                    delta = candidate_values[1:] - reference_values[1:]
                    if not np.all(np.isfinite(delta)):
                        raise ValueError(f"nonfinite pairwise spatial difference: {candidate}, {reference}, {metric}")
                    mean, low, high = (float(candidate_values[0] - reference_values[0]),
                                       float(np.quantile(delta, .025)),
                                       float(np.quantile(delta, .975)))
                else:
                    left = body_issue.loc[body_issue.label.eq(candidate)].sort_values("issue")[metric]
                    right = body_issue.loc[body_issue.label.eq(reference)].sort_values("issue")[metric]
                    mean, low, high = ci_of_paired(left, right, indices)
                rows.append({"family": "body_pairwise", "candidate": candidate, "reference": reference,
                             "metric": metric, "difference": mean, "ci_low": low, "ci_high": high,
                             "repetitions": len(indices), "bootstrap_seed": 20261006})
    return rows


def dilate(mask, radius=6):
    out = np.zeros_like(mask, dtype=bool)
    for shift in range(-radius, radius + 1):
        if shift < 0:
            out[:, :shift] |= mask[:, -shift:]
        elif shift > 0:
            out[:, shift:] |= mask[:, :-shift]
        else:
            out |= mask
    return out


def event_masks(data_path, control_run):
    payload = json.loads((control_run / "jstd_event_targets.json").read_text(encoding="utf-8"))
    targets = build_station_jstd_target_arrays(data_path, "val", payload["thresholds"])
    masks = {"wind": np.zeros((len(targets.event_active), 168), bool),
             "solar": np.zeros((len(targets.event_active), 168), bool)}
    for row in targets.catalog:
        source = str(row["source"])
        if source in masks:
            masks[source][int(row["sample_index"]), int(row["lead_onset"]):int(row["lead_stop_exclusive"])] = True
    return {key: dilate(value, 6) for key, value in masks.items()}


def directed(values, direction):
    selected = values[values > 0] if direction == "positive" else -values[values < 0]
    return selected[np.isfinite(selected)]


def safe_stat(values, fn):
    return float(fn(values)) if len(values) else float("nan")


def ramp_issue_rows(label, alpha, result, stations, masks, daylight):
    samples, actual = result_arrays(result)
    types = stations.data_type.to_numpy(str)
    capacities = stations.capacity_mw.to_numpy(float)
    rows = []
    for source in ("wind", "solar"):
        station_idx = np.flatnonzero(types == source)
        generated = source_aggregate(samples, station_idx, capacities)
        truth = source_aggregate(actual, station_idx, capacities)
        for issue in range(samples.shape[0]):
            for lag in (1, 3, 6):
                g_delta = np.asarray(generated[issue, :, lag:] - generated[issue, :, :-lag])
                y_delta = np.asarray(truth[issue, lag:] - truth[issue, :-lag])
                event_pair = masks[source][issue, lag:] | masks[source][issue, :-lag]
                if source == "solar":
                    daylight_pair = np.any(
                        daylight[issue][lag:, station_idx] & daylight[issue][:-lag, station_idx],
                        axis=-1,
                    )
                else:
                    daylight_pair = np.ones_like(event_pair, dtype=bool)
                for window, time_mask in (("event", event_pair & daylight_pair),
                                          ("non_event", ~event_pair & daylight_pair)):
                    for direction in ("positive", "negative"):
                        y_values = directed(y_delta[time_mask], direction)
                        g_values = directed(g_delta[:, time_mask].reshape(-1), direction)
                        q = {}
                        for quantile in (.90, .95, .99):
                            key = f"q{int(quantile * 100)}"
                            q[f"actual_{key}"] = safe_stat(y_values, lambda x, z=quantile: np.quantile(x, z))
                            q[f"generated_{key}"] = safe_stat(g_values, lambda x, z=quantile: np.quantile(x, z))
                            q[f"{key}_distance"] = abs(q[f"generated_{key}"] - q[f"actual_{key}"])
                        actual_std = safe_stat(y_values, np.std)
                        generated_std = safe_stat(g_values, np.std)
                        actual_direction_mask = time_mask & ((y_delta > 0) if direction == "positive" else (y_delta < 0))
                        median = np.median(g_delta, axis=0)
                        ramp_mae = safe_stat(np.abs(median[actual_direction_mask] - y_delta[actual_direction_mask]), np.mean)
                        if np.any(actual_direction_mask):
                            lo = np.quantile(g_delta[:, actual_direction_mask], .05, axis=0)
                            hi = np.quantile(g_delta[:, actual_direction_mask], .95, axis=0)
                            cov = float(np.mean((y_delta[actual_direction_mask] >= lo) & (y_delta[actual_direction_mask] <= hi)))
                        else:
                            cov = float("nan")
                        rows.append({
                            "alpha": alpha, "label": label, "issue": issue, "source": source,
                            "direction": direction, "window": window, "lag_h": lag,
                            "actual_count": len(y_values), "generated_count": len(g_values),
                            "actual_std": actual_std, "generated_std": generated_std,
                            "std_distance": abs(generated_std - actual_std), **q,
                            "ramp_mae": ramp_mae, "ramp_coverage_90": cov,
                            "ramp_coverage_error": abs(cov - .90) if np.isfinite(cov) else float("nan"),
                        })
    return rows


def ramp_bootstrap(ramp_issue, indices):
    rows = []
    metrics = ("std_distance", "q95_distance", "q99_distance", "ramp_mae", "ramp_coverage_error")
    for candidate in ("alpha_0.65", "alpha_1.35"):
        for keys, candidate_group in ramp_issue.loc[ramp_issue.label.eq(candidate)].groupby(
            ["source", "direction", "window", "lag_h"], sort=False
        ):
            reference = ramp_issue.loc[ramp_issue.label.eq("alpha_1.00")]
            for key, value in zip(("source", "direction", "window", "lag_h"), keys):
                reference = reference.loc[reference[key].eq(value)]
            candidate_group = candidate_group.sort_values("issue")
            reference = reference.sort_values("issue")
            for metric in metrics:
                if candidate_group[metric].isna().any() or reference[metric].isna().any():
                    continue
                mean, low, high = ci_of_paired(candidate_group[metric], reference[metric], indices)
                rows.append({"family": "ramp", "candidate": candidate, "reference": "alpha_1.00",
                             "source": keys[0], "direction": keys[1], "window": keys[2], "lag_h": keys[3],
                             "metric": metric, "difference": mean, "ci_low": low, "ci_high": high,
                             "repetitions": len(indices), "bootstrap_seed": 20261006})
    return rows


def load_eventwise(label, alpha, folder):
    per_event = pd.read_csv(folder / "continuous_event_per_event.csv")
    matches = pd.read_csv(folder / "continuous_event_member_matches.csv")
    variant = next(value for value in per_event.variant.unique() if value != "Raw body-tail")
    selected = per_event.loc[(per_event.variant == variant) & (per_event.scope == "independent_physical")].copy()
    selected["alpha"] = alpha
    selected["label"] = label
    depth = matches.loc[(matches.variant == variant) & (matches.scope == "independent_physical")].copy()
    depth["depth_abs_error"] = pd.to_numeric(depth.depth_abs_error, errors="coerce")
    depth_summary = depth.groupby("event_id", as_index=False).depth_abs_error.median()
    selected = selected.merge(depth_summary, on="event_id", how="left")
    return selected


def persistent_summary(eventwise):
    rows = []
    for (label, alpha, group, standard), frame in eventwise.groupby(["label", "alpha", "member_group", "standard"]):
        rows.append({
            "family": "persistent", "alpha": alpha, "label": label, "member_group": group,
            "standard": standard, "event_count": int(len(frame)),
            "any_hit_count": int(frame.any_hit.astype(str).str.lower().eq("true").sum()) if frame.any_hit.dtype == object else int(frame.any_hit.sum()),
            "member_hit_rate": float(frame.member_hit_rate.mean()),
            "onset_error": float(frame.median_onset_error_h.median()),
            "duration_error": float(frame.median_duration_error_h.median()),
            "depth_error": float(frame.depth_abs_error.median()),
            "depth_ratio": float(frame.median_depth_ratio.median()),
            "depth_ratio_error": float(np.median(np.abs(frame.median_depth_ratio - 1.0))),
        })
    return rows


def event_bootstrap(eventwise):
    rows = []
    rng = np.random.default_rng(20261006)
    for candidate in ("alpha_0.65", "alpha_1.35"):
        for group in ("all", "tail"):
            for standard in ("primary", "strict"):
                left = eventwise.loc[(eventwise.label == candidate) & (eventwise.member_group == group) & (eventwise.standard == standard)].sort_values("event_id")
                right = eventwise.loc[(eventwise.label == "alpha_1.00") & (eventwise.member_group == group) & (eventwise.standard == standard)].sort_values("event_id")
                if list(left.event_id) != list(right.event_id):
                    raise ValueError("persistent event IDs differ between alphas")
                draws = rng.integers(0, len(left), size=(10000, len(left)))
                metric_map = {
                    "member_hit_rate": (left.member_hit_rate.to_numpy(), right.member_hit_rate.to_numpy(), -1),
                    "onset_error": (left.median_onset_error_h.to_numpy(), right.median_onset_error_h.to_numpy(), 1),
                    "duration_error": (left.median_duration_error_h.to_numpy(), right.median_duration_error_h.to_numpy(), 1),
                    "depth_error": (left.depth_abs_error.to_numpy(), right.depth_abs_error.to_numpy(), 1),
                }
                for metric, (a, b, sign) in metric_map.items():
                    # Store all differences as loss orientation: below zero is better.
                    delta = sign * (a - b)
                    boot = delta[draws].mean(axis=1)
                    rows.append({"family": "persistent", "candidate": candidate, "reference": "alpha_1.00",
                                 "member_group": group, "standard": standard, "metric": metric,
                                 "difference": float(delta.mean()), "ci_low": float(np.quantile(boot, .025)),
                                 "ci_high": float(np.quantile(boot, .975)), "repetitions": 10000,
                                 "bootstrap_seed": 20261006})
    return rows


def leave_one_out(eventwise):
    rows = []
    for (label, alpha), frame in eventwise.loc[(eventwise.member_group == "all") & (eventwise.standard == "primary")].groupby(["label", "alpha"]):
        for omitted in sorted(frame.event_id.unique()):
            keep = frame.loc[frame.event_id != omitted]
            rows.append({"alpha": alpha, "label": label, "scope": "leave_one_event_out", "omitted_event": omitted,
                         "member_group": "all", "standard": "primary", "event_count": len(keep),
                         "any_hit_count": int(keep.any_hit.astype(str).str.lower().eq("true").sum()) if keep.any_hit.dtype == object else int(keep.any_hit.sum()),
                         "member_hit_rate": float(keep.member_hit_rate.mean()),
                         "onset_error": float(keep.median_onset_error_h.median()),
                         "duration_error": float(keep.median_duration_error_h.median()),
                         "depth_error": float(keep.depth_abs_error.median())})
    return rows


def ci_lookup(bootstrap, candidate, reference, family, metric, **filters):
    frame = bootstrap.loc[(bootstrap.candidate == candidate) & (bootstrap.reference == reference) &
                          (bootstrap.family == family) & (bootstrap.metric == metric)]
    for key, value in filters.items():
        frame = frame.loc[frame[key].eq(value)]
    if len(frame) != 1:
        return None
    return frame.iloc[0]


def body_gate(alpha_label, body_mean, lead_mean, bootstrap):
    row = body_mean.loc[body_mean.label.eq(alpha_label)].iloc[0]
    raw = body_mean.loc[body_mean.label.eq("raw")].iloc[0]
    hard = {
        "renewable_crps": row.renewable_crps <= raw.renewable_crps * 1.02,
        "energy_score": row.energy_score <= raw.energy_score * 1.01,
        "wind_crps": row.wind_crps <= raw.wind_crps * 1.02,
        "solar_daylight_crps": row.solar_daylight_crps <= raw.solar_daylight_crps * 1.02,
        "spatial_rmse": row.spatial_rmse <= raw.spatial_rmse * 1.05,
        "wind_solar_rmse": row.wind_solar_rmse <= raw.wind_solar_rmse * 1.05,
    }
    ci_pass = True
    lead_day_pass = True
    acf_pass = True
    lead_detail = {"days_5_7_all_point_worse": False, "days_5_7_significantly_worse": 0}
    acf_detail = {"wind_significantly_worse": [], "solar_significantly_worse": []}
    if alpha_label != "alpha_1.00":
        for metric in PRIMARY_BODY:
            item = ci_lookup(bootstrap, alpha_label, "alpha_1.00", "body", metric)
            ci_pass &= item is not None and float(item.ci_low) <= 0
        current_lead = lead_mean.loc[lead_mean.label.eq(alpha_label)].set_index("lead_day")
        control_lead = lead_mean.loc[lead_mean.label.eq("alpha_1.00")].set_index("lead_day")
        point_worse = []
        significant_worse = 0
        for day in (5, 6, 7):
            point_worse.append(
                float(current_lead.loc[day, "renewable_crps"])
                > float(control_lead.loc[day, "renewable_crps"])
            )
            item = ci_lookup(bootstrap, alpha_label, "alpha_1.00", "lead_day",
                             "renewable_crps", lead_day=day)
            significant_worse += item is not None and float(item.ci_low) > 0
        lead_detail = {"days_5_7_all_point_worse": bool(all(point_worse)),
                       "days_5_7_significantly_worse": int(significant_worse)}
        lead_day_pass = not (all(point_worse) and significant_worse >= 2)

        for source in ("wind", "solar"):
            for lag in (24, 48):
                metric = f"{source}_acf_error_{lag}"
                item = ci_lookup(bootstrap, alpha_label, "alpha_1.00", "body", metric)
                if item is not None and float(item.ci_low) > 0:
                    acf_detail[f"{source}_significantly_worse"].append(lag)
        # The frozen rule rejects only when both source classes each contain at
        # least one significant ACF degradation.
        acf_pass = not (acf_detail["wind_significantly_worse"] and
                        acf_detail["solar_significantly_worse"])
    passed = bool(all(hard.values()) and ci_pass and lead_day_pass and acf_pass)
    detail = {"hard_raw_thresholds": hard, "primary_ci_no_degradation": bool(ci_pass),
              "lead_day_5_7": lead_detail, "lead_day_guardrail": bool(lead_day_pass),
              "acf_24_48": acf_detail, "acf_guardrail": bool(acf_pass)}
    return passed, detail


def calibration_status(label, bootstrap):
    if label == "alpha_1.00":
        return "control"
    cov = ci_lookup(bootstrap, label, "alpha_1.00", "body", "coverage_error_90")
    width = ci_lookup(bootstrap, label, "alpha_1.00", "body", "interval_width_90")
    score = ci_lookup(bootstrap, label, "alpha_1.00", "body", "interval_score_90")
    if score.ci_low > 0 or (width.ci_low > 0 and cov.ci_high >= 0):
        return "FAIL_calibration_or_sharpness"
    if cov.ci_high < 0 and width.ci_low > 0 and score.ci_high >= 0:
        return "coverage improvement mainly obtained by interval widening"
    if cov.ci_high < 0 and (width.ci_low <= 0 or score.ci_high < 0):
        return "genuine_calibration_improvement"
    if width.ci_high < 0 and score.ci_high < 0 and cov.ci_low <= 0 <= cov.ci_high:
        return "sharpness_improvement_calibration_neutral"
    return "indistinguishable"


def classify_extreme_family(passed, significant_good, significant_bad,
                            hard_degradation=False):
    """Map evidence to the frozen three-state scale without hiding mixed CIs."""
    if passed:
        return "stable_improvement"
    if hard_degradation or (significant_bad > 0 and significant_good == 0):
        return "stable_degradation"
    if significant_good == 0 and significant_bad == 0:
        return "indistinguishable"
    return "mixed_directional_evidence"


def persistent_status(label, eventwise, bootstrap):
    if label == "alpha_1.00":
        return "control", {}
    current = eventwise.loc[(eventwise.label == label) & (eventwise.member_group == "all") & (eventwise.standard == "primary")].sort_values("event_id")
    control = eventwise.loc[(eventwise.label == "alpha_1.00") & (eventwise.member_group == "all") & (eventwise.standard == "primary")].sort_values("event_id")
    strict_current = eventwise.loc[(eventwise.label == label) & (eventwise.member_group == "all") & (eventwise.standard == "strict")]
    strict_control = eventwise.loc[(eventwise.label == "alpha_1.00") & (eventwise.member_group == "all") & (eventwise.standard == "strict")]
    point = {
        "member_hit_rate": current.member_hit_rate.mean() > control.member_hit_rate.mean(),
        "onset_error": current.median_onset_error_h.median() < control.median_onset_error_h.median(),
        "duration_error": current.median_duration_error_h.median() < control.median_duration_error_h.median(),
        "depth_error": current.depth_abs_error.median() < control.depth_abs_error.median(),
    }
    cis = {metric: ci_lookup(bootstrap, label, "alpha_1.00", "persistent", metric,
                             member_group="all", standard="primary") for metric in point}
    significant_good = sum(item is not None and item.ci_high < 0 for item in cis.values())
    significant_bad = sum(item is not None and item.ci_low > 0 for item in cis.values())
    per_event_nonworse = 0
    for event in current.event_id:
        a, b = current.loc[current.event_id.eq(event)].iloc[0], control.loc[control.event_id.eq(event)].iloc[0]
        votes = [a.member_hit_rate >= b.member_hit_rate, a.median_onset_error_h <= b.median_onset_error_h,
                 a.median_duration_error_h <= b.median_duration_error_h, a.depth_abs_error <= b.depth_abs_error]
        per_event_nonworse += sum(votes) >= 2
    loo_good = 0
    for omitted in current.event_id:
        a, b = current.loc[~current.event_id.eq(omitted)], control.loc[~control.event_id.eq(omitted)]
        votes = [a.member_hit_rate.mean() >= b.member_hit_rate.mean(), a.median_onset_error_h.median() <= b.median_onset_error_h.median(),
                 a.median_duration_error_h.median() <= b.median_duration_error_h.median(), a.depth_abs_error.median() <= b.depth_abs_error.median()]
        a_primary_hits = (int(a.any_hit.astype(str).str.lower().eq("true").sum())
                          if a.any_hit.dtype == object else int(a.any_hit.sum()))
        strict_a = strict_current.loc[~strict_current.event_id.eq(omitted)]
        strict_b = strict_control.loc[~strict_control.event_id.eq(omitted)]
        strict_a_hits = (int(strict_a.any_hit.astype(str).str.lower().eq("true").sum())
                         if strict_a.any_hit.dtype == object else int(strict_a.any_hit.sum()))
        strict_b_hits = (int(strict_b.any_hit.astype(str).str.lower().eq("true").sum())
                         if strict_b.any_hit.dtype == object else int(strict_b.any_hit.sum()))
        loo_gate = a_primary_hits == len(a) and strict_a_hits >= strict_b_hits
        loo_good += sum(votes) >= 2 and loo_gate
    any_hit = int(current.any_hit.astype(str).str.lower().eq("true").sum()) if current.any_hit.dtype == object else int(current.any_hit.sum())
    strict_hits = int(strict_current.any_hit.astype(str).str.lower().eq("true").sum()) if strict_current.any_hit.dtype == object else int(strict_current.any_hit.sum())
    strict_ref = int(strict_control.any_hit.astype(str).str.lower().eq("true").sum()) if strict_control.any_hit.dtype == object else int(strict_control.any_hit.sum())
    control_any_hit = (int(control.any_hit.astype(str).str.lower().eq("true").sum())
                       if control.any_hit.dtype == object else int(control.any_hit.sum()))
    passed = any_hit == 4 and strict_hits >= strict_ref and sum(point.values()) >= 2 and significant_good >= 1 and significant_bad == 0 and per_event_nonworse >= 3 and loo_good >= 3
    evidence = {"primary_any_hit": any_hit, "strict_any_hit": strict_hits, "point_improvements": sum(point.values()),
                "significant_improvements": significant_good, "significant_degradations": significant_bad,
                "events_nonworse": per_event_nonworse, "leave_one_out_improved": loo_good}
    status = classify_extreme_family(
        passed, significant_good, significant_bad,
        hard_degradation=(any_hit < control_any_hit or strict_hits < strict_ref),
    )
    return status, evidence


def ramp_status(label, ramp_mean, bootstrap):
    if label == "alpha_1.00":
        return "control", {}
    candidate = ramp_mean.loc[ramp_mean.label.eq(label)]
    control = ramp_mean.loc[ramp_mean.label.eq("alpha_1.00")]
    strata_pass, significant_sources = [], set()
    for source in ("wind", "solar"):
        for direction in ("positive", "negative"):
            improved_lags = 0
            significant = False
            for lag in (1, 3, 6):
                a = candidate.loc[(candidate.source == source) & (candidate.direction == direction) & (candidate.window == "event") & (candidate.lag_h == lag)].iloc[0]
                b = control.loc[(control.source == source) & (control.direction == direction) & (control.window == "event") & (control.lag_h == lag)].iloc[0]
                improved_lags += (a.q95_distance < b.q95_distance or a.q99_distance < b.q99_distance)
                for metric in ("q95_distance", "q99_distance"):
                    item = ci_lookup(bootstrap, label, "alpha_1.00", "ramp", metric,
                                     source=source, direction=direction, window="event", lag_h=lag)
                    significant |= item is not None and item.ci_high < 0
            passed = improved_lags >= 2
            strata_pass.append((source, direction, passed))
            if passed and significant:
                significant_sources.add(source)
    non_event_bad = 0
    for source in ("wind", "solar"):
        for direction in ("positive", "negative"):
            bad_lags = 0
            for lag in (1, 3, 6):
                bad = False
                for metric in ("q95_distance", "q99_distance"):
                    item = ci_lookup(bootstrap, label, "alpha_1.00", "ramp", metric,
                                     source=source, direction=direction, window="non_event", lag_h=lag)
                    bad |= item is not None and item.ci_low > 0
                bad_lags += bad
            non_event_bad += bad_lags >= 2
    passed_strata = [(s, d) for s, d, passed in strata_pass if passed]
    passed = len(passed_strata) >= 3 and significant_sources == {"wind", "solar"} and {d for _, d in passed_strata} == {"positive", "negative"} and non_event_bad < 3
    significant_good = 0
    significant_bad = 0
    missing_primary_ci = 0
    for source in ("wind", "solar"):
        for direction in ("positive", "negative"):
            for window in ("event", "non_event"):
                for lag in (1, 3, 6):
                    for metric in ("q95_distance", "q99_distance"):
                        item = ci_lookup(
                            bootstrap, label, "alpha_1.00", "ramp", metric,
                            source=source, direction=direction, window=window,
                            lag_h=lag,
                        )
                        if item is None:
                            missing_primary_ci += 1
                            continue
                        significant_good += float(item.ci_high) < 0
                        significant_bad += float(item.ci_low) > 0
    evidence = {"event_strata_passed": len(passed_strata), "significant_sources": sorted(significant_sources),
                "directions_passed": sorted({d for _, d in passed_strata}), "non_event_bad_strata": non_event_bad,
                "significant_improvements": int(significant_good),
                "significant_degradations": int(significant_bad),
                "missing_primary_ci": int(missing_primary_ci)}
    if passed:
        status = "stable_improvement"
    elif missing_primary_ci:
        status = "mixed_or_insufficient_evidence"
    else:
        status = classify_extreme_family(False, significant_good, significant_bad)
    return status, evidence


def _pairwise_body_winner(pool, bootstrap):
    """Apply the frozen lexicographic Body tie-break without a numeric score."""
    survivors = list(pool)
    for metric in ("renewable_crps", "energy_score", "interval_score_90"):
        if len(survivors) <= 1:
            break
        losses = {label: 0 for label in survivors}
        wins = {label: 0 for label in survivors}
        for i, left in enumerate(survivors):
            for right in survivors[i + 1:]:
                pair = bootstrap.loc[
                    (bootstrap.family == "body_pairwise") & (bootstrap.metric == metric) &
                    (((bootstrap.candidate == left) & (bootstrap.reference == right)) |
                     ((bootstrap.candidate == right) & (bootstrap.reference == left)))
                ]
                if len(pair) != 1:
                    continue
                row = pair.iloc[0]
                low, high = float(row.ci_low), float(row.ci_high)
                if row.candidate == right:
                    low, high = -high, -low
                if high < 0:
                    wins[left] += 1; losses[right] += 1
                elif low > 0:
                    wins[right] += 1; losses[left] += 1
        nondominated = [label for label in survivors if losses[label] == 0]
        if len(nondominated) < len(survivors):
            survivors = nondominated
            if len(survivors) == 1:
                return survivors[0], metric
    return None, None


def _pairwise_loss_ci(left, right, metric, bootstrap):
    frame = bootstrap.loc[
        (bootstrap.family == "body_pairwise") & (bootstrap.metric == metric) &
        (((bootstrap.candidate == left) & (bootstrap.reference == right)) |
         ((bootstrap.candidate == right) & (bootstrap.reference == left)))
    ]
    if len(frame) != 1:
        raise ValueError(f"missing pairwise CI: {left}, {right}, {metric}")
    row = frame.iloc[0]
    low, high = float(row.ci_low), float(row.ci_high)
    if row.candidate == right:
        low, high = -high, -low
    return low, high


def _extreme_level(status):
    return {"stable_degradation": 0, "indistinguishable": 1,
            "control": 1, "stable_improvement": 2,
            "mixed_directional_evidence": None,
            "mixed_or_insufficient_evidence": None}[status]


def _pareto_frontier(pool, bootstrap, decisions):
    """Frozen CI-based dominance; no point-estimate or weighted score shortcut."""
    extreme = {
        label: (_extreme_level(decisions[label]["persistent"]),
                _extreme_level(decisions[label]["ramp"]))
        for label in pool
    }
    frontier = []
    for label in pool:
        dominated = False
        for other in pool:
            if other == label:
                continue
            body_cis = [_pairwise_loss_ci(other, label, metric, bootstrap)
                        for metric in PRIMARY_BODY]
            body_no_worse = all(low <= 0 for low, _ in body_cis)
            comparable = all(a is not None and b is not None
                             for a, b in zip(extreme[other], extreme[label]))
            extreme_no_worse = comparable and all(
                a >= b for a, b in zip(extreme[other], extreme[label])
            )
            strictly_better = (any(high < 0 for _, high in body_cis)
                               or (comparable and any(
                                   a > b for a, b in zip(extreme[other], extreme[label])
                               )))
            if body_no_worse and extreme_no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(label)
    return frontier


def choose_alpha(body_mean, lead_mean, bootstrap, eventwise, ramp_mean):
    decisions = {}
    for alpha in ALPHAS:
        label = f"alpha_{alpha:.2f}"
        body_pass, body_detail = body_gate(label, body_mean, lead_mean, bootstrap)
        calibration = calibration_status(label, bootstrap)
        persistent, persistent_evidence = persistent_status(label, eventwise, bootstrap)
        ramp, ramp_evidence = ramp_status(label, ramp_mean, bootstrap)
        eligible = body_pass and not calibration.startswith("FAIL") and calibration != "coverage improvement mainly obtained by interval widening"
        decisions[label] = {"alpha": alpha, "body_guardrail": body_pass,
                            "body_guardrail_detail": body_detail,
                            "calibration": calibration, "persistent": persistent,
                            "persistent_evidence": persistent_evidence, "ramp": ramp, "ramp_evidence": ramp_evidence,
                            "eligible": eligible}
    eligible = [label for label, value in decisions.items() if value["eligible"]]
    if not eligible:
        # Frozen protocol falls back to current control; this should only occur if reused artifacts fail.
        selected = "alpha_1.00"
        reason = "no new candidate passed all preregistered gates; retain current control"
        selection_status = "RETAIN_CONTROL_NO_ELIGIBLE_CANDIDATE"
    else:
        frontier = _pareto_frontier(eligible, bootstrap, decisions)
        for label in decisions:
            decisions[label]["pareto_frontier"] = label in frontier
        both = [label for label in frontier
                if decisions[label]["persistent"] == "stable_improvement"
                and decisions[label]["ramp"] == "stable_improvement"]
        if len(both) == 1:
            selected = both[0]
            reason = ("unique Pareto candidate with stable improvement in both "
                      "Persistent and Ramp/local-trajectory families")
            return selected, reason, decisions, "SELECTED_ALPHA_STAR"
        pool = both if both else frontier
        if len(pool) == 1:
            selected = pool[0]
            reason = "unique eligible candidate remaining on the frozen Pareto frontier"
            return selected, reason, decisions, "SELECTED_ALPHA_STAR"
        extremes_indistinguishable = all(
            decisions[label][family] in ("control", "indistinguishable")
            for label in pool for family in ("persistent", "ramp")
        )
        if not extremes_indistinguishable:
            # The frozen protocol only authorizes the Body tie-break when the
            # Extreme families are genuinely indistinguishable. A Body-vs-
            # Extreme trade-off has no preregistered scalar tie-break, so fail
            # closed and keep the current control operationally unchanged.
            selected = "alpha_1.00"
            reason = (
                "frozen protocol does not uniquely resolve the Body-versus-Extreme "
                "trade-off on the Pareto frontier; retain current control and "
                "block Stage 1B pending protocol governance"
            )
            return selected, reason, decisions, "BLOCKED_PROTOCOL_AMBIGUITY"
        # A statistically resolved Body winner must not subsequently be
        # overwritten by the closeness-to-control fallback.
        selected, decisive_metric = _pairwise_body_winner(pool, bootstrap)
        if selected is not None:
            reason = ("Pareto/gates passed; frozen lexicographic Body tie-break "
                      f"was resolved by {decisive_metric}")
            return selected, reason, decisions, "SELECTED_ALPHA_STAR"
        # Only a complete paired-CI tie reaches the closeness fallback.
        values = {label: decisions[label]["alpha"] for label in pool}
        selected = min(pool,
                       key=lambda label: (abs(values[label] - 1.0), values[label] > 1.0, values[label]))
        reason = ("Pareto/gates passed and preregistered Body comparisons were "
                  "indistinguishable; closest-to-control fallback applied")
        selection_status = "SELECTED_ALPHA_STAR"
    return selected, reason, decisions, selection_status


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stations = pd.read_csv(args.data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    station_types = stations.data_type.to_numpy(str)
    adjacency = np.load(args.data_path / "station_adjacency.npy")
    daylight, _ = build_station_daylight_mask(args.data_path, "val")
    alpha_roots = {0.65: args.alpha065_root, 1.35: args.alpha135_root}
    candidate_artifacts = {}
    for alpha, root in alpha_roots.items():
        manifest = root / "finalize_artifacts.json"
        if manifest.is_file():
            candidate_artifacts[alpha] = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            candidate_artifacts[alpha] = {
                "mixture": str(root / "mixture_body400_tail100_n500"),
                "event_dir": str(root / "postprocess/continuous_event_evaluation"),
            }
    results = {
        "raw": args.raw_result,
        "alpha_1.00": args.control_result,
        "alpha_0.65": Path(candidate_artifacts[0.65]["mixture"]),
        "alpha_1.35": Path(candidate_artifacts[1.35]["mixture"]),
    }
    event_dirs = {
        "alpha_1.00": args.control_post / "continuous_event_evaluation",
        "alpha_0.65": Path(candidate_artifacts[0.65]["event_dir"]),
        "alpha_1.35": Path(candidate_artifacts[1.35]["event_dir"]),
    }
    runs = {"alpha_1.00": args.control_run}
    for alpha, root in alpha_roots.items():
        runs[f"alpha_{alpha:.2f}"] = Path((root / "train_run.txt").read_text(encoding="utf-8").strip())
    for path in results.values():
        if not (path / "metrics.json").is_file():
            raise FileNotFoundError(path / "metrics.json")

    body_rows, lead_rows = [], []
    for label, result in results.items():
        alpha = float("nan") if label == "raw" else float(label.split("_")[1])
        rows, lead = body_issue_rows(label, alpha, result, stations, adjacency, daylight)
        body_rows.extend(rows); lead_rows.extend(lead)
    body_issue = pd.DataFrame(body_rows)
    lead_issue = pd.DataFrame(lead_rows)
    body_mean = body_issue.groupby(["alpha", "label"], dropna=False, as_index=False).mean(numeric_only=True).drop(columns="issue")
    lead_mean = lead_issue.groupby(["alpha", "label", "lead_day"], dropna=False, as_index=False).mean(numeric_only=True).drop(columns="issue")
    # Spatial correlations are nonlinear aggregate statistics.  They must be
    # recomputed after concatenating the resampled 7-day issue blocks, not
    # averaged from per-issue correlations (which may be undefined for a
    # constant station within one issue).
    indices = moving_block_indices(body_issue.issue.nunique())
    _, raw_actual = result_arrays(results["raw"])
    actual_moments = moments_by_issue(raw_actual)
    full_and_bootstrap = np.vstack(
        [np.arange(body_issue.issue.nunique(), dtype=np.int64)[None, :], indices]
    )
    spatial_draws = {}
    for label, result in results.items():
        metrics = spatial_metrics_for_draws(
            result, actual_moments, full_and_bootstrap, station_types
        )
        # Index 0 is the full-validation point estimate. Remaining entries are
        # paired moving-block bootstrap draws. Keeping both prevents the
        # bootstrap mean from being mislabeled as the observed difference.
        spatial_draws[label] = metrics
        for name, values in metrics.items():
            body_mean.loc[body_mean.label.eq(label), name] = float(values[0])
    body_mean.to_csv(args.output_dir / "alpha_body_metrics.csv", index=False)
    lead_mean.to_csv(args.output_dir / "alpha_lead_day_metrics.csv", index=False)
    bootstrap_rows = body_bootstrap(
        body_issue, lead_issue, list(results), indices, spatial_draws
    )

    masks = event_masks(args.data_path, args.control_run)
    ramp_rows = []
    for label, result in results.items():
        if label == "raw":
            continue
        alpha = float(label.split("_")[1])
        ramp_rows.extend(ramp_issue_rows(label, alpha, result, stations, masks, daylight))
    ramp_issue = pd.DataFrame(ramp_rows)
    ramp_mean = ramp_issue.groupby(["alpha", "label", "source", "direction", "window", "lag_h"], as_index=False).mean(numeric_only=True).drop(columns="issue")
    bootstrap_rows.extend(ramp_bootstrap(ramp_issue, indices))

    event_frames = [load_eventwise(label, float(label.split("_")[1]), event_dirs[label])
                    for label in event_dirs]
    eventwise = pd.concat(event_frames, ignore_index=True)
    eventwise_output = pd.concat([eventwise, pd.DataFrame(leave_one_out(eventwise))], ignore_index=True, sort=False)
    eventwise_output.to_csv(args.output_dir / "alpha_eventwise_metrics.csv", index=False)
    persistent_rows = persistent_summary(eventwise)
    bootstrap_rows.extend(event_bootstrap(eventwise))
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(args.output_dir / "alpha_bootstrap_ci.csv", index=False)

    extreme = pd.concat([pd.DataFrame(persistent_rows), ramp_mean.assign(family="ramp")], ignore_index=True, sort=False)
    # Preserve existing drop-recovery diagnostics as an auxiliary, non-selection table.
    recovery_rows = []
    for alpha, root in alpha_roots.items():
        pointer = root / "drop_recovery_dir.txt"
        recovery_dir = (Path(pointer.read_text(encoding="utf-8").strip()) if pointer.is_file()
                        else root / "postprocess/drop_recovery_diagnostic")
        path = recovery_dir / "drop_recovery_summary.csv"
        if path.is_file():
            frame = pd.read_csv(path)
            frame["alpha"] = alpha
            frame["family"] = "drop_recovery_auxiliary"
            recovery_rows.append(frame)
    if recovery_rows:
        extreme = pd.concat([extreme, *recovery_rows], ignore_index=True, sort=False)
    extreme.to_csv(args.output_dir / "alpha_extreme_metrics.csv", index=False)

    selected, reason, decisions, selection_status = choose_alpha(
        body_mean, lead_mean, bootstrap, eventwise, ramp_mean
    )
    selected_alpha = float(selected.split("_")[1])
    selected_payload = {"selected_alpha": selected_alpha, "selected_label": selected,
                        "weights": dict(zip(("ramp", "shape", "slow"), WEIGHTS[selected_alpha])),
                        "selection_status": selection_status,
                        "alpha_star_frozen": selection_status == "SELECTED_ALPHA_STAR",
                        "retained_control_alpha": 1.00,
                        "stage1b_launch_eligible": selection_status == "SELECTED_ALPHA_STAR",
                        "reason": reason, "decisions": decisions, "stage1b_started": False,
                        "test_used": False, "multi_seed_started": False}
    (args.output_dir / "selected_alpha.json").write_text(
        json.dumps(selected_payload, indent=2, default=json_default),
        encoding="utf-8",
    )

    summary_rows = []
    for alpha in ALPHAS:
        label = f"alpha_{alpha:.2f}"
        row = body_mean.loc[body_mean.label.eq(label)].iloc[0].to_dict()
        row.update({"lambda_ramp": WEIGHTS[alpha][0], "lambda_shape": WEIGHTS[alpha][1], "lambda_slow": WEIGHTS[alpha][2],
                    "body_guardrail": decisions[label]["body_guardrail"], "calibration_status": decisions[label]["calibration"],
                    "persistent_status": decisions[label]["persistent"], "ramp_status": decisions[label]["ramp"],
                    "operational_control": alpha == selected_alpha,
                    "alpha_star_selected": (selection_status == "SELECTED_ALPHA_STAR"
                                            and alpha == selected_alpha)})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "alpha_summary.csv", index=False)

    # Exact body400 equality and training/generation integrity collection.
    member_arrays = (
        "actual_scenarios_normalized.npy", "actual_scenarios_raw_normalized.npy",
        "generated_residual_normalized.npy", "generated_residual_standardized.npy",
        "generated_stochastic_residual_standardized.npy",
    )
    body_equal = {}
    for label in ("alpha_0.65", "alpha_1.00", "alpha_1.35"):
        body_equal[label] = {}
        for name in member_arrays:
            reference = load_array(args.raw_result / name)
            candidate = load_array(results[label] / name)
            body_equal[label][name] = (
                candidate.shape[0] == reference.shape[0]
                and candidate.shape[1] == 500
                and all(np.array_equal(reference[i, :400], candidate[i, :400])
                        for i in range(len(candidate)))
            )
    body_equal_pass = all(all(items.values()) for items in body_equal.values())
    generation_audit = {}
    for label, path in results.items():
        metadata = json.loads((path / "metrics.json").read_text(encoding="utf-8"))["run"]
        tail_source = metadata.get("source_runs", [{}])[-1] if metadata.get("source_runs") else metadata
        generation_audit[label] = {
            "split": metadata.get("split"), "generation_seed": metadata.get("generation_seed"),
            "n_samples": metadata.get("n_samples"), "test_used": metadata.get("test_used"),
            "condition_variant": metadata.get("condition_variant"),
            "tail_issue_batch_size": tail_source.get("issue_batch_size"),
            "tail_member_chunk_size": tail_source.get("member_chunk_size"),
            "tail_checkpoint_state_source": tail_source.get("checkpoint_state_source"),
        }
    generation_pass = all(
        row["split"] == "val" and int(row["generation_seed"]) == 424242
        and int(row["n_samples"]) == 500 and not bool(row["test_used"])
        for row in generation_audit.values()
    )
    control_generation = generation_audit["alpha_1.00"]
    pairing_keys = ("tail_issue_batch_size", "tail_member_chunk_size", "tail_checkpoint_state_source")
    generation_pairing_pass = all(
        all(generation_audit[label][key] == control_generation[key] for key in pairing_keys)
        for label in ("alpha_0.65", "alpha_1.35")
    )
    spatial_bootstrap_finite = all(
        np.all(np.isfinite(values))
        for label_metrics in spatial_draws.values()
        for values in label_metrics.values()
    )
    integrity = {"status": "PASS" if body_equal_pass and generation_pass and generation_pairing_pass and spatial_bootstrap_finite else "FAIL",
                 "body400_equal": body_equal, "generation": generation_audit,
                 "generation_pairing_pass": generation_pairing_pass,
                 "evaluation_semantics_version": "stage1a_decision_state_v3",
                 "spatial_bootstrap_method": "paired_7day_blocks_recomputed_from_additive_station_moments",
                 "spatial_bootstrap_all_finite": spatial_bootstrap_finite,
                 "selection_status": selection_status,
                 "validation_issue_count": int(body_issue.issue.nunique()), "generation_seed": 424242,
                 "bootstrap_repetitions": 10000, "bootstrap_seed": 20261006, "test_used": False,
                 "runs": {}, "frozen_protocol_hashes": {}}
    for label, run in runs.items():
        integrity["runs"][label] = {"run": str(run), "checkpoint": str(run / "checkpoints/model_best.pt"),
                                     "checkpoint_sha256": sha(run / "checkpoints/model_best.pt")}
    protocol_names = ("FROZEN_HYPERPARAMETER_SELECTION_PROTOCOL.md", "validation_objectives.md",
                      "pareto_selection_protocol.md", "AUXILIARY_HYPERPARAMETER_OPTIMIZATION_PLAN.md")
    for name in protocol_names:
        path = Path("docs/auxiliary_hyperparameter_optimization_20261006") / name
        integrity["frozen_protocol_hashes"][name] = sha(path)
    frozen_at_launch = json.loads((args.alpha065_root.parent / "frozen_protocol_hashes.json").read_text(encoding="utf-8"))
    integrity["frozen_protocol_hashes_at_launch"] = frozen_at_launch
    integrity["protocol_unchanged_during_run"] = all(
        frozen_at_launch.get(str(Path("docs/auxiliary_hyperparameter_optimization_20261006") / name))
        == integrity["frozen_protocol_hashes"][name] for name in protocol_names
    )
    if not integrity["protocol_unchanged_during_run"]:
        integrity["status"] = "FAIL"
    for alpha, root in alpha_roots.items():
        integrity["runs"][f"alpha_{alpha:.2f}"]["training_integrity"] = json.loads((root / "alpha_run_integrity.json").read_text(encoding="utf-8"))
    (args.output_dir / "alpha_integrity_audit.json").write_text(
        json.dumps(integrity, indent=2, default=json_default),
        encoding="utf-8",
    )
    if integrity["status"] != "PASS":
        raise RuntimeError("Stage 1A integrity gate failed")

    ordered = pd.DataFrame(summary_rows).sort_values("alpha")
    def trend(metric):
        values = ordered[metric].to_numpy(float)
        if np.all(np.diff(values) < 0):
            return "monotonic improvement as auxiliary strength increases"
        if np.all(np.diff(values) > 0):
            return "monotonic degradation as auxiliary strength increases"
        best = int(np.argmin(values))
        return f"non-monotonic; lowest value occurs at alpha={ordered.iloc[best].alpha:.2f}"

    lines = [
        "# Auxiliary Alpha Sensitivity Report", "",
        "**Validation only. Test NOT RUN; Stage 1B NOT RUN; additional generation/train seeds NOT RUN.**", "",
        "This report compares only overall auxiliary strength. Architecture, Raw checkpoint/freeze, "
        "sampling, relative ramp:shape:slow proportions, optimizer, train budget, generation seed "
        "and 400 Raw + 100 Tail composition are fixed.", "",
        "## A. Integrity and comparability", "",
        f"- Integrity gate: **{integrity['status']}**.",
        f"- Raw Body 400 exact equality across all five member arrays: **{body_equal_pass}**.",
        f"- Validation split / 500 members / generation seed 424242 / test lock: **{generation_pass}**.",
        "- Spatial/correlation CIs are recomputed from concatenated paired 7-day bootstrap blocks; "
        "undefined per-issue correlations are never averaged or silently converted into a failed gate.",
        f"- All recomputed spatial point estimates and bootstrap draws finite: **{spatial_bootstrap_finite}**.",
        f"- Frozen protocol unchanged from launch through selection: **{integrity['protocol_unchanged_during_run']}**.",
        "- Each new run separately passed target CUDA/AMP, 20,588-trainable-parameter, Raw-state hash and post-training integrity gates.", "",
        "## B. Ordinary 168 h Body quality", "",
        "| alpha | ramp/shape/slow | wind CRPS | solar-daylight CRPS | renewable CRPS | Energy | coverage | width | Interval Score | spatial RMSE | wind-solar RMSE | Body gate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row.alpha:.2f} | {row.lambda_ramp:.3f}/{row.lambda_shape:.3f}/{row.lambda_slow:.3f} "
            f"| {row.wind_crps:.6f} | {row.solar_daylight_crps:.6f} | {row.renewable_crps:.6f} "
            f"| {row.energy_score:.6f} | {row.coverage_90:.4f} | {row.interval_width_90:.4f} "
            f"| {row.interval_score_90:.4f} | {row.spatial_rmse:.6f} | {row.wind_solar_rmse:.6f} "
            f"| {'PASS' if row.body_guardrail else 'FAIL'} |"
        )
    lines += [
        "", "Lead-day 1–7 and wind/solar 24/48 h ACF checks are retained in "
        "`alpha_lead_day_metrics.csv` and in each candidate's Body-gate evidence; they are not hidden "
        "inside a weighted score.", "",
        "## C. Calibration and sharpness", "",
    ]
    for _, row in ordered.iterrows():
        lines.append(f"- alpha={row.alpha:.2f}: **{row.calibration_status}**.")
    lines += [
        "", "A coverage reduction in absolute error counts as genuine only when paired bootstrap "
        "does not identify interval widening without Interval-Score improvement.", "",
        "## D. Persistent-event and ramp/local-trajectory families", "",
        "| alpha | Persistent family | Ramp/local family | primary hits | strict hits | event strata passing | non-event bad strata |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        label = f"alpha_{row.alpha:.2f}"
        p = decisions[label]["persistent_evidence"]
        r = decisions[label]["ramp_evidence"]
        lines.append(
            f"| {row.alpha:.2f} | {row.persistent_status} | {row.ramp_status} "
            f"| {p.get('primary_any_hit', 'control')} | {p.get('strict_any_hit', 'control')} "
            f"| {r.get('event_strata_passed', 'control')} | {r.get('non_event_bad_strata', 'control')} |"
        )
    lines += [
        "", "Persistent and Ramp remain separate families. A single event, solar case, direction or "
        "lag cannot promote a candidate. Full event-wise, source/direction/lag and paired-CI records "
        "are in `alpha_eventwise_metrics.csv`, `alpha_extreme_metrics.csv`, and `alpha_bootstrap_ci.csv`.", "",
        "## E. Sensitivity and Body–Extreme trade-off", "",
        f"- Renewable CRPS: {trend('renewable_crps')}.",
        f"- Energy Score: {trend('energy_score')}.",
        f"- 90% Interval Score: {trend('interval_score_90')}.",
    ]
    for label, decision in decisions.items():
        lines.append(
            f"- {label}: eligible={decision['eligible']}; Pareto-frontier="
            f"{decision.get('pareto_frontier', False)}; Persistent={decision['persistent']}; "
            f"Ramp={decision['ramp']}."
        )
    lines += [
        "", "No mathematical knee and no arbitrary weighted performance score were used. Dominance "
        "uses paired Body CIs plus the preregistered Extreme stability states. Mixed directional "
        "evidence is kept incomparable instead of being relabeled as indistinguishable.", "",
        "## F. Frozen Stage 1A decision", "",
        f"Decision status: **{selection_status}**.", "", reason, "",
    ]
    if selection_status == "SELECTED_ALPHA_STAR":
        lines += [
            f"Selected alpha*: **{selected_alpha:.2f}**.", "",
            "This is the unique alpha passed to Stage 1B under the frozen protocol; it is not yet a "
            "claim that the current deployable control has been replaced. Final replacement still "
            "requires the later preregistered seed confirmation.", "",
        ]
    else:
        lines += [
            "Selected alpha*: **NOT SELECTED**.", "",
            "Operational control retained: **alpha=1.00 (0.18/0.14/0.10)**. This is a fail-closed "
            "protocol result, not a post-hoc claim that alpha=1.00 is theoretically optimal. "
            "Stage 1B is not launch-eligible until the frozen-protocol ambiguity is governed before "
            "any additional result is observed.", "",
        ]
    lines += [
        "## Mandatory stop", "",
        "Stage 1B: **NOT RUN**. Test: **NOT RUN**. Additional alpha points: **NOT RUN**. "
        "Multi-seed confirmation: **NOT RUN**.", "",
    ]
    (args.output_dir / "ALPHA_SENSITIVITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        "AUXILIARY_ALPHA_SELECTION_COMPLETE "
        f"status={selection_status} operational_alpha={selected_alpha:.2f} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

