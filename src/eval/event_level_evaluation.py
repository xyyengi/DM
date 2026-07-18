#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event-level re-evaluation of saved scenario ensembles (no generation)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CHANNELS = ("wind", "solar", "load")
LEAD_BINS = (
    ("0-24h", 0, 24),
    ("24-48h", 24, 48),
    ("48-72h", 48, 72),
    ("72-168h", 72, 168),
)


def lead_group(lead_hours: int) -> str | None:
    for label, lower, upper in LEAD_BINS:
        if lower <= lead_hours < upper:
            return label
    return None


def _upper_pair_mean(samples: np.ndarray, norm: bool) -> np.ndarray:
    """Mean pair distance over ensemble members, retaining observation axes."""
    members = samples.shape[0]
    if members < 2:
        shape = () if norm else samples.shape[1:]
        return np.zeros(shape, dtype=np.float64)
    left, right = np.triu_indices(members, 1)
    diff = samples[left] - samples[right]
    if norm:
        return float(np.mean(np.sqrt(np.sum(diff * diff, axis=tuple(range(1, diff.ndim))))))
    return np.mean(np.abs(diff), axis=0)


def window_probability_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    global_ranges: np.ndarray,
) -> dict:
    """Metrics for one window/event slice: samples[S,C,T], actual[C,T]."""
    if samples.ndim != 3 or actual.ndim != 2 or samples.shape[1:] != actual.shape:
        raise ValueError(f"shape mismatch samples={samples.shape}, actual={actual.shape}")
    first_abs = np.mean(np.abs(samples - actual[None, :, :]), axis=0)
    pair_abs = _upper_pair_mean(samples, norm=False)
    crps_points = first_abs - 0.5 * pair_abs

    result = {
        "total_crps_mw": float(np.mean(crps_points)),
        "total_mae_scenario_mean_mw": float(np.mean(np.abs(np.mean(samples, axis=0) - actual))),
    }
    for channel, name in enumerate(CHANNELS):
        result[f"{name}_crps_mw"] = float(np.mean(crps_points[channel]))
        for q in (0.80, 0.90, 0.95):
            lower = np.quantile(samples[:, channel], (1.0 - q) / 2.0, axis=0)
            upper = np.quantile(samples[:, channel], 1.0 - (1.0 - q) / 2.0, axis=0)
            result[f"{name}_coverage_{int(q * 100)}"] = float(
                100.0 * np.mean((actual[channel] >= lower) & (actual[channel] <= upper))
            )
            result[f"{name}_width_{int(q * 100)}_pct_range"] = float(
                100.0 * np.mean(upper - lower) / global_ranges[channel]
            )
    for q in (80, 90, 95):
        result[f"total_coverage_{q}"] = float(np.mean([
            result[f"{name}_coverage_{q}"] for name in CHANNELS
        ]))
        result[f"total_width_{q}_pct_range"] = float(np.mean([
            result[f"{name}_width_{q}_pct_range"] for name in CHANNELS
        ]))

    truth_distance = np.mean([
        np.sqrt(np.sum((member - actual) ** 2)) for member in samples
    ])
    pair_distance = float(_upper_pair_mean(samples, norm=True))
    es = float(truth_distance - 0.5 * pair_distance)
    dimension = int(actual.size)
    result["multivariate_energy_score_mw"] = es
    result["multivariate_es_per_sqrt_dimension_mw"] = float(es / np.sqrt(dimension))
    return result


def aggregate_ordinary_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    window_indices: Sequence[int],
    global_ranges: np.ndarray,
) -> dict:
    metrics = [
        window_probability_metrics(samples[idx], actual[idx], global_ranges)
        for idx in window_indices
    ]
    keys = metrics[0].keys()
    return {
        "n_windows": len(window_indices),
        **{key: float(np.mean([row[key] for row in metrics])) for key in keys},
    }


def audit_saved_run(
    run_dir: str | Path,
    expected_actual_norm: np.ndarray,
    expected_forecast_norm: np.ndarray,
    scales: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Confirm row order/units and return physical scenarios and actual arrays."""
    run_dir = Path(run_dir)
    required = [
        "actual_scenarios.npy", "actual_scenarios_normalized.npy",
        "actual_data.npy", "actual_data_normalized.npy",
        "forecast_data.npy", "forecast_data_normalized.npy",
        "denormalization_used.json", "samples/scenarios.npz",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return ({"eligible": False, "reason": f"missing files: {missing}"}, None, None)

    scenarios = np.load(run_dir / "actual_scenarios.npy")
    scenarios_norm = np.load(run_dir / "actual_scenarios_normalized.npy")
    actual = np.load(run_dir / "actual_data.npy")
    actual_norm = np.load(run_dir / "actual_data_normalized.npy")
    forecast = np.load(run_dir / "forecast_data.npy")
    forecast_norm = np.load(run_dir / "forecast_data_normalized.npy")
    denorm = json.loads((run_dir / "denormalization_used.json").read_text(encoding="utf-8"))
    scale_shape = scales.reshape(1, 3, 1)
    scenario_scale_shape = scales.reshape(1, 1, 3, 1)

    checks = {
        "actual_shape": list(actual.shape),
        "scenario_shape": list(scenarios.shape),
        "actual_normalized_matches_source": bool(np.allclose(actual_norm, expected_actual_norm, atol=1e-6, rtol=1e-6)),
        "forecast_normalized_matches_source": bool(np.allclose(forecast_norm, expected_forecast_norm, atol=1e-6, rtol=1e-6)),
        "actual_physical_matches_denormalization": bool(np.allclose(actual, actual_norm * scale_shape, atol=0.02, rtol=1e-6)),
        "forecast_physical_matches_denormalization": bool(np.allclose(forecast, forecast_norm * scale_shape, atol=0.02, rtol=1e-6)),
        "scenarios_physical_matches_denormalization": bool(np.allclose(scenarios, scenarios_norm * scenario_scale_shape, atol=0.02, rtol=1e-6)),
        "channel_order_declared": denorm.get("channel_order"),
        "scales_declared": denorm.get("scales"),
        "scales_match": bool(np.allclose(denorm.get("scales", []), scales)),
    }
    expected_shape_ok = (
        actual.shape == expected_actual_norm.shape
        and forecast.shape == expected_forecast_norm.shape
        and scenarios.ndim == 4
        and scenarios.shape[0] == expected_actual_norm.shape[0]
        and scenarios.shape[2:] == expected_actual_norm.shape[1:]
    )
    checks["shape_and_window_count_ok"] = bool(expected_shape_ok)

    with np.load(run_dir / "samples" / "scenarios.npz") as archive:
        flat = np.stack([archive["wind"], archive["pv"], archive["load"]], axis=1)
        checks["npz_keys"] = list(archive.files)
        checks["npz_flat_shape"] = list(flat.shape)
        checks["npz_matches_windowed_scenarios"] = bool(
            flat.shape == (scenarios.shape[0] * scenarios.shape[1], 3, 168)
            and np.allclose(flat, scenarios.reshape(flat.shape), atol=1e-5, rtol=1e-6)
        )
        checks["npz_probability_uniform"] = bool(
            "prob" in archive.files
            and archive["prob"].shape == (flat.shape[0],)
            and np.allclose(archive["prob"], 1.0 / flat.shape[0])
        )

    required_checks = [
        "actual_normalized_matches_source", "forecast_normalized_matches_source",
        "actual_physical_matches_denormalization", "forecast_physical_matches_denormalization",
        "scenarios_physical_matches_denormalization", "scales_match",
        "shape_and_window_count_ok", "npz_matches_windowed_scenarios",
        "npz_probability_uniform",
    ]
    checks["n_test_windows"] = int(scenarios.shape[0]) if scenarios.ndim == 4 else None
    checks["n_scenarios_per_window"] = int(scenarios.shape[1]) if scenarios.ndim == 4 else None
    checks["window_id_rule"] = "array row i == test_window_{i:05d}; proven by actual/forecast equality to source window order"
    checks["eligible"] = bool(all(checks[key] for key in required_checks))
    checks["reason"] = "all correspondence checks passed" if checks["eligible"] else "one or more correspondence checks failed"
    return checks, scenarios, actual


def event_peak_values(samples: np.ndarray, event_type: str, scales: Mapping[str, float]) -> np.ndarray:
    """One peak/severity value per ensemble member for a fully contained event."""
    wind, solar, load = samples[:, 0], samples[:, 1], samples[:, 2]
    if event_type == "low_wind":
        return np.min(wind, axis=1)
    if event_type == "low_renewable":
        return np.min(wind + solar, axis=1)
    if event_type == "low_solar_daily_energy":
        return np.sum(solar, axis=1) / (float(scales["solar_capacity_mw"]) * samples.shape[-1])
    if event_type == "high_load":
        return np.max(load, axis=1)
    net_load = load - wind - solar
    if event_type == "high_net_load":
        return np.max(net_load, axis=1)
    if event_type.startswith("high_ramp_"):
        horizon = int(event_type.split("_")[-1][:-1])
        if net_load.shape[1] <= horizon:
            return np.full(samples.shape[0], np.nan)
        return np.max(np.abs(net_load[:, horizon:] - net_load[:, :-horizon]), axis=1)
    return np.full(samples.shape[0], np.nan)


def duration_values(
    samples: np.ndarray,
    event_type: str,
    thresholds: Mapping[str, object],
    scales: Mapping[str, float],
    low_level: str,
    high_level: str,
) -> np.ndarray:
    wind, solar, load = samples[:, 0], samples[:, 1], samples[:, 2]
    if event_type == "low_wind":
        mask = wind / float(scales["wind_capacity_mw"]) <= thresholds["low_wind_capacity_ratio"][low_level]
    elif event_type == "low_renewable":
        mask = (wind + solar) / (float(scales["wind_capacity_mw"]) + float(scales["solar_capacity_mw"])) <= thresholds["low_renewable_capacity_ratio"][low_level]
    elif event_type == "high_load":
        mask = load >= thresholds["high_load_mw"][high_level]
    elif event_type == "high_net_load":
        mask = load - wind - solar >= thresholds["high_net_load_mw"][high_level]
    elif event_type == "low_solar_daily_energy":
        ratio = np.sum(solar, axis=1) / (float(scales["solar_capacity_mw"]) * samples.shape[-1])
        return np.where(ratio <= thresholds["low_solar_daily_daylight_energy_ratio"][low_level], samples.shape[-1], 0)
    else:
        return np.full(samples.shape[0], np.nan)
    return np.sum(mask, axis=1).astype(float)


def evaluate_event_mappings(
    samples: np.ndarray,
    actual: np.ndarray,
    mappings: Sequence[Mapping[str, object]],
    events: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, object],
    scales: Mapping[str, float],
    global_ranges: np.ndarray,
    event_types: Iterable[str],
    low_level: str,
    high_level: str,
) -> tuple[list[dict], list[dict]]:
    """Two-level aggregation: windows -> event/lead group -> event-type/lead."""
    event_by_id = {row["event_id"]: row for row in events}
    requested = set(event_types)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    full_grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for mapping in mappings:
        if mapping["event_type"] not in requested:
            continue
        lead = int(mapping["lead_hours"])
        group = lead_group(lead)
        if group is None or not bool(mapping["contains_event_start"]) or bool(mapping["post_onset"]):
            continue
        window_idx = int(str(mapping["window_id"]).rsplit("_", 1)[1])
        start = max(0, int(mapping["event_start_index"]))
        end = min(167, int(mapping["event_end_index"]))
        if end < start:
            continue
        metrics = window_probability_metrics(
            samples[window_idx, :, :, start : end + 1],
            actual[window_idx, :, start : end + 1],
            global_ranges,
        )
        key = (str(mapping["event_type"]), str(mapping["event_id"]), group)
        grouped[key].append(metrics)

        if bool(mapping["fully_contains_event"]):
            sample_slice = samples[window_idx, :, :, start : end + 1]
            actual_slice = actual[window_idx, :, start : end + 1]
            full_metrics = dict(metrics)
            generated_peaks = event_peak_values(sample_slice, str(mapping["event_type"]), scales)
            actual_peak = float(event_peak_values(actual_slice[None, :, :], str(mapping["event_type"]), scales)[0])
            full_metrics["peak_mae"] = float(np.nanmean(np.abs(generated_peaks - actual_peak)))
            durations = duration_values(sample_slice, str(mapping["event_type"]), thresholds, scales, low_level, high_level)
            actual_duration = float(event_by_id[str(mapping["event_id"])]["duration_hours"])
            full_metrics["duration_mae_hours"] = float(np.nanmean(np.abs(durations - actual_duration))) if np.isfinite(durations).any() else np.nan
            full_grouped[key].append(full_metrics)

    event_rows: list[dict] = []
    keys = sorted(set(grouped) | set(full_grouped))
    for event_type, event_id, group in keys:
        onset = grouped.get((event_type, event_id, group), [])
        full = full_grouped.get((event_type, event_id, group), [])
        row = {
            "event_type": event_type,
            "event_id": event_id,
            "lead_group": group,
            "n_windows": len(onset),
            "n_full_windows": len(full),
        }
        if onset:
            for metric in onset[0]:
                row[f"onset_{metric}"] = float(np.mean([x[metric] for x in onset]))
        if full:
            for metric in full[0]:
                values = np.asarray([x[metric] for x in full], dtype=float)
                row[f"full_{metric}"] = float(np.mean(values[np.isfinite(values)])) if np.isfinite(values).any() else np.nan
        event_rows.append(row)

    summary_rows: list[dict] = []
    by_type_lead: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in event_rows:
        by_type_lead[(row["event_type"], row["lead_group"])].append(row)
    for event_type in sorted(requested):
        for lead_label, _, _ in LEAD_BINS:
            rows = by_type_lead.get((event_type, lead_label), [])
            window_counts = [row["n_windows"] for row in rows]
            summary = {
                "event_type": event_type,
                "lead_group": lead_label,
                "n_events": len(rows),
                "n_windows": int(sum(window_counts)),
                "windows_per_event_mean": float(np.mean(window_counts)) if rows else 0.0,
                "windows_per_event_min": int(min(window_counts)) if rows else 0,
                "windows_per_event_max": int(max(window_counts)) if rows else 0,
                "case_only": len(rows) < 3,
            }
            metric_keys = sorted({key for row in rows for key in row if key.startswith(("onset_", "full_"))})
            for metric in metric_keys:
                values = [row[metric] for row in rows if metric in row and np.isfinite(row[metric])]
                summary[metric] = float(np.mean(values)) if values else np.nan
            summary_rows.append(summary)
    return event_rows, summary_rows


def coupling_metrics(
    values: np.ndarray,
    thresholds: Mapping[str, object],
    high_level: str = "p90",
) -> dict:
    """Coupling attributes for values[S,C,T], using fixed train tail thresholds."""
    wind_drop = np.maximum(values[:, 0, :-1] - values[:, 0, 1:], 0.0)
    solar_drop = np.maximum(values[:, 1, :-1] - values[:, 1, 1:], 0.0)
    load_rise = np.maximum(values[:, 2, 1:] - values[:, 2, :-1], 0.0)
    wind_sig = wind_drop >= thresholds["wind_drop_1h_mw_positive"][high_level]
    solar_sig = solar_drop >= thresholds["solar_drop_1h_mw_positive"][high_level]
    source_sig = wind_sig | solar_sig
    load_sig = load_rise >= thresholds["load_rise_1h_mw_positive"][high_level]
    same = np.sum(source_sig & load_sig, axis=1).astype(float)
    lag_ratios = []
    for member in range(values.shape[0]):
        source_indices = np.flatnonzero(source_sig[member])
        if source_indices.size == 0:
            lag_ratios.append(np.nan)
            continue
        matches = 0
        for idx in source_indices:
            if np.any(load_sig[member, idx : min(idx + 4, load_sig.shape[1])]):
                matches += 1
        lag_ratios.append(matches / source_indices.size)
    return {
        "wind_max_drop_mw": float(np.mean(np.max(wind_drop, axis=1))) if wind_drop.shape[1] else 0.0,
        "solar_max_drop_mw": float(np.mean(np.max(solar_drop, axis=1))) if solar_drop.shape[1] else 0.0,
        "load_max_rise_mw": float(np.mean(np.max(load_rise, axis=1))) if load_rise.shape[1] else 0.0,
        "same_hour_source_drop_load_rise_hours": float(np.mean(same)),
        "lag_0_3h_match_ratio": float(np.nanmean(lag_ratios)) if np.isfinite(lag_ratios).any() else np.nan,
    }


def evaluate_coupling(
    samples: np.ndarray,
    actual: np.ndarray,
    mappings: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, object],
) -> list[dict]:
    grouped: dict[tuple[str, str], list[tuple[dict, dict]]] = defaultdict(list)
    for mapping in mappings:
        if mapping["event_type"] not in {"high_net_load", "high_ramp_6h"}:
            continue
        if not (mapping["fully_contains_event"] and int(mapping["lead_hours"]) >= 0):
            continue
        window_idx = int(str(mapping["window_id"]).rsplit("_", 1)[1])
        start, end = int(mapping["event_start_index"]), int(mapping["event_end_index"])
        generated = coupling_metrics(samples[window_idx, :, :, start : end + 1], thresholds)
        truth = coupling_metrics(actual[window_idx, None, :, start : end + 1], thresholds)
        grouped[(str(mapping["event_type"]), str(mapping["event_id"]))].append((generated, truth))
    rows = []
    for (event_type, event_id), pairs in sorted(grouped.items()):
        row = {"event_type": event_type, "event_id": event_id, "n_full_windows": len(pairs)}
        for key in pairs[0][0]:
            generated_values = [pair[0][key] for pair in pairs if np.isfinite(pair[0][key])]
            actual_value = pairs[0][1][key]
            generated_mean = float(np.mean(generated_values)) if generated_values else np.nan
            row[f"actual_{key}"] = actual_value
            row[f"generated_{key}"] = generated_mean
            row[f"difference_generated_minus_actual_{key}"] = generated_mean - actual_value if np.isfinite(generated_mean) and np.isfinite(actual_value) else np.nan
        rows.append(row)
    return rows
