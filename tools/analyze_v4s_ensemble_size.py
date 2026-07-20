#!/usr/bin/env python
"""Compare V4-s (50 members) with V4 at an equal 20-member ensemble size."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "outputs_shandong/20260718_145118_v4_residual_forecast_time_no_guidance_168h"
V4S = ROOT / "outputs_shandong/20260718_232509_v4s_residual_event_sampler_no_guidance_168h"
OUT = ROOT / "outputs_shandong/event_evaluation/v4s_analysis/ensemble_size_audit.json"
CHANNELS = ("wind", "solar", "load")


def interval_metrics(samples: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ranges = np.ptp(actual, axis=(0, 2))
    result: dict[str, float] = {}
    for channel, name in enumerate(CHANNELS):
        channel_samples = np.asarray(samples[:, :, channel, :])
        members = channel_samples.shape[1]
        sorted_samples = np.sort(channel_samples, axis=1)
        coefficients = (2 * np.arange(members) - members + 1).reshape(1, members, 1)
        mean_pair_distance = (
            2.0 * np.sum(sorted_samples * coefficients, axis=1) / (members * (members - 1))
        )
        result[f"{name}_crps_mw"] = float(np.mean(
            np.mean(np.abs(channel_samples - actual[:, None, channel, :]), axis=1)
            - 0.5 * mean_pair_distance
        ))
        for nominal in (0.80, 0.90, 0.95):
            alpha = (1.0 - nominal) / 2.0
            lower = np.quantile(samples[:, :, channel, :], alpha, axis=1)
            upper = np.quantile(samples[:, :, channel, :], 1.0 - alpha, axis=1)
            q = int(nominal * 100)
            result[f"{name}_coverage_{q}"] = float(
                100.0 * np.mean((actual[:, channel, :] >= lower) & (actual[:, channel, :] <= upper))
            )
            result[f"{name}_width_{q}_pct_range"] = float(
                100.0 * np.mean(upper - lower) / ranges[channel]
            )
    for q in (80, 90, 95):
        result[f"total_coverage_{q}"] = float(np.mean([result[f"{name}_coverage_{q}"] for name in CHANNELS]))
        result[f"total_width_{q}_pct_range"] = float(np.mean([result[f"{name}_width_{q}_pct_range"] for name in CHANNELS]))
    result["total_crps_mw"] = float(np.mean([result[f"{name}_crps_mw"] for name in CHANNELS]))
    result["mean_mae_mw"] = float(np.mean(np.abs(np.mean(samples, axis=1) - actual)))
    return result


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "sd": float(np.std([row[key] for row in rows], ddof=1)),
            "p05": float(np.quantile([row[key] for row in rows], 0.05)),
            "p95": float(np.quantile([row[key] for row in rows], 0.95)),
        }
        for key in rows[0]
    }


def diagnostic_test_fitted_shrinkage(samples: np.ndarray, actual: np.ndarray) -> dict[str, dict[str, float]]:
    """Quantify overdispersion severity only; these test-fitted factors are not deployable."""
    mean = np.mean(samples, axis=1, keepdims=True)
    deviations = samples - mean
    result = {}
    for channel, name in enumerate(CHANNELS):
        lower_dev = np.quantile(deviations[:, :, channel, :], 0.05, axis=1)
        upper_dev = np.quantile(deviations[:, :, channel, :], 0.95, axis=1)
        lo, hi = 0.0, 1.5
        for _ in range(40):
            factor = (lo + hi) / 2.0
            lower = mean[:, 0, channel, :] + factor * lower_dev
            upper = mean[:, 0, channel, :] + factor * upper_dev
            coverage = 100.0 * np.mean((actual[:, channel, :] >= lower) & (actual[:, channel, :] <= upper))
            if coverage < 90.0:
                lo = factor
            else:
                hi = factor
        factor = (lo + hi) / 2.0
        lower = mean[:, 0, channel, :] + factor * lower_dev
        upper = mean[:, 0, channel, :] + factor * upper_dev
        result[name] = {
            "factor": float(factor),
            "coverage_90": float(100.0 * np.mean((actual[:, channel, :] >= lower) & (actual[:, channel, :] <= upper))),
            "width_90_pct_range": float(100.0 * np.mean(upper - lower) / np.ptp(actual[:, channel, :])),
        }
    return result


def residual_scale_diagnostic(run_dir: Path) -> dict[str, dict[str, float]]:
    samples = np.load(run_dir / "actual_scenarios.npy", mmap_mode="r")
    actual = np.load(run_dir / "actual_data.npy", mmap_mode="r")
    forecast = np.load(run_dir / "forecast_data.npy", mmap_mode="r")
    actual_residual = actual - forecast
    generated_residual = samples - forecast[:, None, :, :]
    result = {}
    for channel, name in enumerate(CHANNELS):
        actual_std = float(np.std(actual_residual[:, channel, :]))
        generated_std = float(np.std(generated_residual[:, :, channel, :]))
        result[name] = {
            "actual_residual_std_mw": actual_std,
            "generated_residual_std_mw": generated_std,
            "generated_to_actual_std_ratio": generated_std / actual_std,
            "actual_residual_bias_mw": float(np.mean(actual_residual[:, channel, :])),
            "generated_residual_bias_mw": float(np.mean(generated_residual[:, :, channel, :])),
            "mean_pointwise_ensemble_spread_mw": float(np.mean(np.std(samples[:, :, channel, :], axis=1))),
            "ensemble_mean_mae_mw": float(np.mean(np.abs(np.mean(samples[:, :, channel, :], axis=1) - actual[:, channel, :]))),
        }
    return result


def main() -> None:
    v4_samples = np.load(V4 / "actual_scenarios.npy", mmap_mode="r")
    v4_actual = np.load(V4 / "actual_data.npy", mmap_mode="r")
    v4s_samples = np.load(V4S / "actual_scenarios.npy", mmap_mode="r")
    v4s_actual = np.load(V4S / "actual_data.npy", mmap_mode="r")
    if not np.allclose(v4_actual, v4s_actual, atol=0.02, rtol=1e-6):
        raise RuntimeError("V4 and V4-s actual arrays are not aligned")

    rng = np.random.default_rng(20260718)
    subsets = [np.sort(rng.choice(v4s_samples.shape[1], size=20, replace=False)) for _ in range(40)]
    subset_rows = [interval_metrics(v4s_samples[:, subset], v4s_actual) for subset in subsets]
    record = {
        "comparison_purpose": "separate event-sampler effects from the change n_samples=20 to 50",
        "actual_arrays_aligned": True,
        "v4_n_samples": int(v4_samples.shape[1]),
        "v4s_n_samples": int(v4s_samples.shape[1]),
        "v4_20_members": interval_metrics(v4_samples, v4_actual),
        "v4s_all_50_members": interval_metrics(v4s_samples, v4s_actual),
        "v4s_first_20_members": interval_metrics(v4s_samples[:, :20], v4s_actual),
        "v4s_random_20_member_subsets": {
            "n_repeats": len(subset_rows),
            "seed": 20260718,
            "summary": summarize(subset_rows),
        },
        "v4s_test_fitted_shrinkage_diagnostic_only": diagnostic_test_fitted_shrinkage(v4s_samples, v4s_actual),
        "residual_scale_diagnostic": {
            "V4": residual_scale_diagnostic(V4),
            "V4s": residual_scale_diagnostic(V4S),
        },
        "calibration_warning": "Do not use these factors as formal results; fit channel factors on validation scenarios only.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
