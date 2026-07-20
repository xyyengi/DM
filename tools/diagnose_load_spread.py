#!/usr/bin/env python
"""Decompose the unusually large load spread in saved Shandong runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "diffusion_npy_normalized"
OUT = ROOT / "outputs_shandong/event_evaluation/v4s_analysis/load_spread_diagnostic.json"
RUNS = {
    "V4": ROOT / "outputs_shandong/20260718_145118_v4_residual_forecast_time_no_guidance_168h",
    "V4s": ROOT / "outputs_shandong/20260718_232509_v4s_residual_event_sampler_no_guidance_168h",
}
CHANNELS = ("wind", "solar", "load")


def unique_hourly(windows: np.ndarray) -> np.ndarray:
    """Recover stride-one unique hours from [window,168,channel]."""
    return np.concatenate([windows[0], windows[1:, -1, :]], axis=0)


def split_stats(split: str, scales: np.ndarray) -> dict:
    actual = unique_hourly(np.load(DATA / f"{split}_actual.npy", mmap_mode="r"))
    forecast = unique_hourly(np.load(DATA / f"{split}_forecast.npy", mmap_mode="r"))
    residual = actual - forecast
    result = {"unique_hours": int(actual.shape[0]), "channels": {}}
    for channel, name in enumerate(CHANNELS):
        physical = residual[:, channel] * scales[channel]
        result["channels"][name] = {
            "residual_mean_mw": float(np.mean(physical)),
            "residual_std_mw": float(np.std(physical)),
            "residual_mae_mw": float(np.mean(np.abs(physical))),
            "residual_p01_mw": float(np.quantile(physical, 0.01)),
            "residual_p99_mw": float(np.quantile(physical, 0.99)),
            "normalized_residual_std": float(np.std(residual[:, channel])),
            "actual_min_mw": float(np.min(actual[:, channel]) * scales[channel]),
            "actual_max_mw": float(np.max(actual[:, channel]) * scales[channel]),
            "forecast_min_mw": float(np.min(forecast[:, channel]) * scales[channel]),
            "forecast_max_mw": float(np.max(forecast[:, channel]) * scales[channel]),
        }
    return result


def run_stats(run_dir: Path, load_scale: float) -> dict:
    samples = np.load(run_dir / "actual_scenarios.npy", mmap_mode="r")[:, :, 2, :]
    actual = np.load(run_dir / "actual_data.npy", mmap_mode="r")[:, 2, :]
    forecast = np.load(run_dir / "forecast_data.npy", mmap_mode="r")[:, 2, :]
    generated_residual = samples - forecast[:, None, :]
    actual_residual = actual - forecast
    conditional_mean = np.mean(generated_residual, axis=1)
    within_variance = np.var(generated_residual, axis=1)
    total_variance = float(np.var(generated_residual))
    mean_within_variance = float(np.mean(within_variance))
    between_variance = float(np.var(conditional_mean))
    flat_samples = np.asarray(samples).reshape(-1)
    return {
        "saved_members": int(samples.shape[1]),
        "actual_residual_std_mw": float(np.std(actual_residual)),
        "generated_residual_total_std_mw": float(np.sqrt(total_variance)),
        "generated_conditional_mean_residual_std_mw": float(np.sqrt(between_variance)),
        "generated_within_ensemble_rms_spread_mw": float(np.sqrt(mean_within_variance)),
        "variance_decomposition_relative_error": float(
            abs(total_variance - mean_within_variance - between_variance) / total_variance
        ),
        "conditional_mean_residual_mae_vs_actual_mw": float(np.mean(np.abs(conditional_mean - actual_residual))),
        "conditional_mean_residual_correlation_with_actual": float(
            np.corrcoef(conditional_mean.reshape(-1), actual_residual.reshape(-1))[0, 1]
        ),
        "scenario_load_min_mw": float(np.min(flat_samples)),
        "scenario_load_p001_mw": float(np.quantile(flat_samples, 0.001)),
        "scenario_load_p01_mw": float(np.quantile(flat_samples, 0.01)),
        "scenario_load_p99_mw": float(np.quantile(flat_samples, 0.99)),
        "scenario_load_p999_mw": float(np.quantile(flat_samples, 0.999)),
        "scenario_load_max_mw": float(np.max(flat_samples)),
        "scenario_fraction_below_zero": float(np.mean(flat_samples < 0.0)),
        "scenario_fraction_above_train_load_scale": float(np.mean(flat_samples > load_scale)),
    }


def main() -> None:
    params = json.loads((DATA / "normalization_params.json").read_text(encoding="utf-8"))
    scales = np.asarray([
        params["wind_total_capacity"], params["solar_total_capacity"], params["load_denominator"]
    ], dtype=float)
    record = {
        "purpose": "separate split shift, denormalization errors, between-time variation, and within-ensemble spread",
        "scales_mw": dict(zip(CHANNELS, scales.tolist())),
        "unique_hourly_split_stats": {split: split_stats(split, scales) for split in ("train", "val", "test")},
        "saved_run_load_decomposition": {
            model: run_stats(run_dir, scales[2]) for model, run_dir in RUNS.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
