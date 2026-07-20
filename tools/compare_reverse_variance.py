#!/usr/bin/env python
"""Compare beta and posterior generation outputs from the same checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = ("wind", "solar", "load")


def summarize(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    scenarios = np.load(run_dir / "actual_scenarios.npy", mmap_mode="r")
    actual = np.load(run_dir / "actual_data.npy", mmap_mode="r")
    forecast = np.load(run_dir / "forecast_data.npy", mmap_mode="r")
    generated_residual = scenarios - forecast[:, None, :, :]
    actual_residual = actual - forecast
    channels = {}
    for channel, name in enumerate(CHANNELS):
        within_rms = float(np.sqrt(np.mean(np.var(generated_residual[:, :, channel, :], axis=1))))
        channels[name] = {
            "actual_residual_std_mw": float(np.std(actual_residual[:, channel, :])),
            "generated_residual_std_mw": float(np.std(generated_residual[:, :, channel, :])),
            "within_ensemble_rms_spread_mw": within_rms,
            "crps_mw": float(metrics[f"{name}_crps"]),
            "coverage_90": float(metrics[f"{name}_coverage_90%"]),
            "width_90_pct_range": float(metrics[f"{name}_width_90%"]),
        }
    return {
        "run_dir": str(run_dir),
        "reverse_variance_type": metrics.get("reverse_variance_type", "unknown"),
        "n_windows": int(scenarios.shape[0]),
        "n_members": int(scenarios.shape[1]),
        "total_crps_mw": float(metrics["total_crps"]),
        "total_coverage_90": float(metrics["total_coverage_90%"]),
        "total_width_90_pct_range": float(metrics["total_width_90%"]),
        "channels": channels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("beta_run", type=Path)
    parser.add_argument("posterior_run", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    record = {"beta": summarize(args.beta_run), "posterior": summarize(args.posterior_run)}
    text = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
