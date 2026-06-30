#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Advanced evaluation helpers for scenario generation runs.

All thresholds are fitted on the training split and then applied to test
windows. Arrays use the project convention:

    scenario samples: [N, S, C, L]
    actual/forecast: [N, C, L]

Channels are ordered as wind, pv, load.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


CHANNELS = ("wind", "pv", "load")


@dataclass(frozen=True)
class ScenarioArrays:
    samples: np.ndarray
    actual: np.ndarray
    forecast: Optional[np.ndarray] = None


def load_actual_from_split(data_path: str, split: str) -> np.ndarray:
    """Load reconstructed actual [N, C, L] for one split."""
    pred = np.load(os.path.join(data_path, f"{split}_pred.npy"))[:, :, :3]
    res = np.load(os.path.join(data_path, f"{split}_res.npy"))[:, :, :3]
    actual = pred - res
    return actual.transpose(0, 2, 1).astype(np.float64)


def _net_load(actual: np.ndarray) -> np.ndarray:
    return actual[:, 2, :] - actual[:, 0, :] - actual[:, 1, :]


def _ramp_6h(series: np.ndarray) -> np.ndarray:
    if series.shape[1] <= 6:
        return np.max(np.abs(np.diff(series, axis=1)), axis=1)
    return np.max(np.abs(series[:, 6:] - series[:, :-6]), axis=1)


def _fluctuation(series: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(series, axis=1)), axis=1)


def fit_extreme_thresholds(train_actual: np.ndarray) -> Dict[str, float]:
    """Fit extreme-window thresholds from train actual only."""
    renewable_mean = np.mean(train_actual[:, 0, :] + train_actual[:, 1, :], axis=1)
    load_mean = np.mean(train_actual[:, 2, :], axis=1)
    load_max = np.max(train_actual[:, 2, :], axis=1)
    net_load = _net_load(train_actual)
    netload_max = np.max(net_load, axis=1)
    netload_ramp_6h = _ramp_6h(net_load)
    netload_fluctuation = _fluctuation(net_load)

    thresholds = {
        "low_renewable_mean_p10": float(np.percentile(renewable_mean, 10)),
        "high_load_max_p90": float(np.percentile(load_max, 90)),
        "high_load_mean_p90": float(np.percentile(load_mean, 90)),
        "high_netload_max_p90": float(np.percentile(netload_max, 90)),
        "high_ramp_6h_p90": float(np.percentile(netload_ramp_6h, 90)),
        "high_ramp_fluctuation_p90": float(np.percentile(netload_fluctuation, 90)),
    }
    for idx, name in enumerate(CHANNELS):
        values = train_actual[:, idx, :].reshape(-1)
        thresholds[f"{name}_actual_p05"] = float(np.percentile(values, 5))
        thresholds[f"{name}_actual_p95"] = float(np.percentile(values, 95))
    return thresholds


def apply_extreme_flags(actual: np.ndarray, thresholds: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """Apply train-fitted thresholds to a split."""
    renewable_mean = np.mean(actual[:, 0, :] + actual[:, 1, :], axis=1)
    load_mean = np.mean(actual[:, 2, :], axis=1)
    load_max = np.max(actual[:, 2, :], axis=1)
    net_load = _net_load(actual)
    netload_max = np.max(net_load, axis=1)
    netload_ramp_6h = _ramp_6h(net_load)
    netload_fluctuation = _fluctuation(net_load)

    low_renewable = renewable_mean <= thresholds["low_renewable_mean_p10"]
    high_load = (
        (load_max >= thresholds["high_load_max_p90"])
        | (load_mean >= thresholds["high_load_mean_p90"])
    )
    high_netload = netload_max >= thresholds["high_netload_max_p90"]
    high_ramp = (
        (netload_ramp_6h >= thresholds["high_ramp_6h_p90"])
        | (netload_fluctuation >= thresholds["high_ramp_fluctuation_p90"])
    )
    compound_extreme = low_renewable & (high_load | high_netload)

    return {
        "low_renewable": low_renewable,
        "high_load": high_load,
        "high_netload": high_netload,
        "high_ramp": high_ramp,
        "compound_extreme": compound_extreme,
    }


def save_thresholds(thresholds: Mapping[str, float], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(thresholds), f, indent=2, ensure_ascii=False)


def save_flags(flags: Mapping[str, np.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = list(flags.keys())
    n = len(next(iter(flags.values()))) if flags else 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_index", *names])
        writer.writeheader()
        for i in range(n):
            row = {"sample_index": i}
            row.update({name: int(flags[name][i]) for name in names})
            writer.writerow(row)


def coverage(samples: np.ndarray, actual: np.ndarray, quantile: float = 1.0) -> float:
    lower_q = (1.0 - quantile) / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(samples, lower_q, axis=1)
    upper = np.quantile(samples, upper_q, axis=1)
    return float(np.mean((actual >= lower) & (actual <= upper)) * 100.0)


def interval_width(samples: np.ndarray, actual: np.ndarray, quantile: float = 1.0) -> float:
    lower_q = (1.0 - quantile) / 2.0
    upper_q = 1.0 - lower_q
    lower = np.quantile(samples, lower_q, axis=1)
    upper = np.quantile(samples, upper_q, axis=1)
    value_range = float(np.max(actual) - np.min(actual))
    denom = value_range if value_range > 1e-12 else 1.0
    return float(np.mean(upper - lower) / denom * 100.0)


def crps(samples: np.ndarray, actual: np.ndarray) -> float:
    """Vectorized empirical CRPS for [N, S, L] samples."""
    term1 = np.mean(np.abs(samples - actual[:, None, :]))
    sorted_samples = np.sort(samples, axis=1)
    s = samples.shape[1]
    if s <= 1:
        term2 = 0.0
    else:
        weights = (2 * np.arange(1, s + 1) - s - 1).reshape(1, s, 1)
        pair_mean = 2.0 * np.sum(weights * sorted_samples) / (s * s * samples.shape[0] * samples.shape[2])
        term2 = 0.5 * pair_mean
    return float(term1 - term2)


def acf_mae(samples: np.ndarray, actual: np.ndarray, max_lag: int = 24) -> float:
    def _acf_mean(data: np.ndarray) -> np.ndarray:
        out = np.zeros(max_lag, dtype=np.float64)
        out[0] = 1.0
        for lag in range(1, max_lag):
            vals = []
            for series in data:
                centered = series - np.mean(series)
                denom = np.sum(centered ** 2)
                if denom > 1e-12:
                    vals.append(float(np.sum(centered[:-lag] * centered[lag:]) / denom))
            out[lag] = float(np.mean(vals)) if vals else 0.0
        return out

    actual_acf = _acf_mean(actual)
    sample_acfs = [_acf_mean(samples[:, s, :]) for s in range(samples.shape[1])]
    return float(np.mean(np.abs(actual_acf - np.mean(sample_acfs, axis=0))))


def ramp_error(samples: np.ndarray, actual: np.ndarray) -> float:
    sample_mean = np.mean(samples, axis=1)
    sample_ramp = np.diff(sample_mean, axis=1)
    actual_ramp = np.diff(actual, axis=1)
    return float(np.mean(np.abs(sample_ramp - actual_ramp)))


def extreme_ramp_error(samples: np.ndarray, actual: np.ndarray) -> float:
    sample_mean = np.mean(samples, axis=1)
    sample_ramp = np.max(np.abs(np.diff(sample_mean, axis=1)), axis=1)
    actual_ramp = np.max(np.abs(np.diff(actual, axis=1)), axis=1)
    return float(np.mean(np.abs(sample_ramp - actual_ramp)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    if x_flat.size < 2 or np.std(x_flat) < 1e-12 or np.std(y_flat) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_flat, y_flat)[0, 1])


def channel_metrics(
    samples: np.ndarray,
    actual: np.ndarray,
    channel: str,
    run_id: str,
    subset: str,
) -> Dict[str, float | str | int]:
    sample_mean = np.mean(samples, axis=1)
    err = sample_mean - actual
    abs_err = np.abs(err)
    row: Dict[str, float | str | int] = {
        "run_id": run_id,
        "subset": subset,
        "channel": channel,
        "n_windows": int(samples.shape[0]),
        "n_points": int(samples.shape[0] * samples.shape[2]),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(math.sqrt(np.mean(err ** 2))),
        "CRPS": crps(samples, actual),
        "coverage_100": coverage(samples, actual, 1.0),
        "coverage_90": coverage(samples, actual, 0.9),
        "coverage_80": coverage(samples, actual, 0.8),
        "interval_width_100": interval_width(samples, actual, 1.0),
        "interval_width_90": interval_width(samples, actual, 0.9),
        "ACF_MAE": acf_mae(samples, actual),
        "corr_mean_actual": pearson_corr(sample_mean, actual),
        "ramp_error": ramp_error(samples, actual),
        "extreme_ramp_error": extreme_ramp_error(samples, actual),
        "P95_MAE": float(np.percentile(abs_err, 95)),
        "P99_MAE": float(np.percentile(abs_err, 99)),
    }
    return row


def aggregate_total(rows: List[Dict[str, float | str | int]], run_id: str, subset: str) -> Dict[str, float | str | int]:
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key not in {"n_windows", "n_points"}
    ]
    out: Dict[str, float | str | int] = {
        "run_id": run_id,
        "subset": subset,
        "channel": "total",
        "n_windows": rows[0]["n_windows"],
        "n_points": int(sum(int(row["n_points"]) for row in rows)),
    }
    for key in numeric_keys:
        out[key] = float(np.nanmean([float(row[key]) for row in rows]))
    return out


def evaluate_rows(arrays: ScenarioArrays, run_id: str, subset: str = "overall", mask: Optional[np.ndarray] = None) -> List[Dict[str, float | str | int]]:
    samples = arrays.samples
    actual = arrays.actual
    if mask is not None:
        samples = samples[mask]
        actual = actual[mask]
    if samples.shape[0] == 0:
        return []

    rows = []
    for idx, name in enumerate(CHANNELS):
        rows.append(channel_metrics(samples[:, :, idx, :], actual[:, idx, :], name, run_id, subset))
    rows.append(aggregate_total(rows, run_id, subset))
    return rows


def tail_rows(
    arrays: ScenarioArrays,
    thresholds: Mapping[str, float],
    run_id: str,
    reference_actual: Optional[np.ndarray] = None,
) -> List[Dict[str, float | str | int]]:
    rows: List[Dict[str, float | str | int]] = []
    tail_actual = reference_actual if reference_actual is not None else arrays.actual
    if tail_actual.shape != arrays.actual.shape:
        raise ValueError(
            f"reference_actual shape {tail_actual.shape} must match actual shape {arrays.actual.shape}"
        )
    for idx, name in enumerate(CHANNELS):
        samples = arrays.samples[:, :, idx, :]
        actual = arrays.actual[:, idx, :]
        actual_for_tail = tail_actual[:, idx, :]
        sample_mean = np.mean(samples, axis=1)
        abs_err = np.abs(sample_mean - actual)
        tail_mask = (
            (actual_for_tail <= thresholds[f"{name}_actual_p05"])
            | (actual_for_tail >= thresholds[f"{name}_actual_p95"])
        )
        tail_count = int(np.sum(tail_mask))
        if tail_count == 0:
            tail_cov_90 = float("nan")
        else:
            lower = np.quantile(samples, 0.05, axis=1)
            upper = np.quantile(samples, 0.95, axis=1)
            tail_cov_90 = float(np.mean(((actual >= lower) & (actual <= upper))[tail_mask]) * 100.0)
        rows.append({
            "run_id": run_id,
            "channel": name,
            "n_tail_points": tail_count,
            "P95_MAE": float(np.percentile(abs_err, 95)),
            "P99_MAE": float(np.percentile(abs_err, 99)),
            "tail_MAE": float(np.mean(abs_err[tail_mask])) if tail_count else float("nan"),
            "tail_coverage_90": tail_cov_90,
            "extreme_ramp_error": extreme_ramp_error(samples, actual),
        })
    total = {
        "run_id": run_id,
        "channel": "total",
        "n_tail_points": int(sum(int(row["n_tail_points"]) for row in rows)),
        "P95_MAE": float(np.nanmean([float(row["P95_MAE"]) for row in rows])),
        "P99_MAE": float(np.nanmean([float(row["P99_MAE"]) for row in rows])),
        "tail_MAE": float(np.nanmean([float(row["tail_MAE"]) for row in rows])),
        "tail_coverage_90": float(np.nanmean([float(row["tail_coverage_90"]) for row in rows])),
        "extreme_ramp_error": float(np.nanmean([float(row["extreme_ramp_error"]) for row in rows])),
    }
    rows.append(total)
    return rows

def rank_histogram_rows(arrays: ScenarioArrays, run_id: str) -> List[Dict[str, int | str]]:
    rows: List[Dict[str, int | str]] = []
    for idx, name in enumerate(CHANNELS):
        samples = arrays.samples[:, :, idx, :]
        actual = arrays.actual[:, idx, :]
        ranks = np.sum(samples < actual[:, None, :], axis=1).reshape(-1)
        counts = np.bincount(ranks, minlength=samples.shape[1] + 1)
        for rank, count in enumerate(counts):
            rows.append({"run_id": run_id, "channel": name, "rank": rank, "count": int(count)})
    return rows


def write_csv(path: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
