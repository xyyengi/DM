#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate forecast quality against actual wind/pv/load curves.

The current processed dataset stores forecast in *_pred.npy and residual in
*_res.npy. The residual definition is:

    residual = forecast - actual
    actual = forecast - residual

Expected array shape is [N, 168, C], where C >= 3 and channels [0:3] are
wind, pv, load.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


CHANNELS = ("wind", "pv", "load")


def maybe_compare_true(data_path, split, reconstructed_actual):
    """Compare reconstructed actual with true.npy when shapes can be aligned."""
    true_path = Path(data_path) / "true.npy"
    if not true_path.exists() or split == "legacy":
        return None

    true = np.load(true_path)
    if true.ndim != 3 or true.shape[1] != reconstructed_actual.shape[1] or true.shape[2] < 3:
        return {
            "status": "not_aligned",
            "reason": f"true.npy shape {true.shape} is incompatible with {reconstructed_actual.shape}",
        }

    true_3ch = true[:, :, :3].astype(np.float64)
    n = min(true_3ch.shape[0], reconstructed_actual.shape[0])
    if n == 0:
        return {"status": "not_aligned", "reason": "empty true/reconstructed arrays"}

    diff = reconstructed_actual[:n] - true_3ch[:n]
    return {
        "status": "aligned_by_prefix",
        "n_aligned": int(n),
        "true_shape": list(true_3ch.shape),
        "reconstructed_shape": list(reconstructed_actual.shape),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs_error": float(np.max(np.abs(diff))),
    }


def load_split(data_path, split):
    """Load forecast and actual arrays for one split."""
    prefix = "" if split == "legacy" else f"{split}_"
    pred_path = Path(data_path) / f"{prefix}pred.npy"
    res_path = Path(data_path) / f"{prefix}res.npy"

    if split == "legacy":
        pred_path = Path(data_path) / "pred.npy"
        true_path = Path(data_path) / "true.npy"
        if not pred_path.exists() or not true_path.exists():
            raise FileNotFoundError(f"Expected {pred_path} and {true_path}")
        forecast = np.load(pred_path)
        actual = np.load(true_path)
    else:
        if not pred_path.exists() or not res_path.exists():
            raise FileNotFoundError(f"Expected {pred_path} and {res_path}")
        forecast = np.load(pred_path)
        residual = np.load(res_path)
        actual = forecast - residual

    if forecast.ndim != 3 or actual.ndim != 3:
        raise ValueError(f"Expected [N, 168, C], got forecast={forecast.shape}, actual={actual.shape}")
    if forecast.shape != actual.shape:
        raise ValueError(f"Shape mismatch: forecast={forecast.shape}, actual={actual.shape}")
    if forecast.shape[1] != 168:
        raise ValueError(f"Expected length 168, got {forecast.shape[1]}")
    if forecast.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels, got {forecast.shape[2]}")

    forecast_3ch = forecast[:, :, :3].astype(np.float64)
    actual_3ch = actual[:, :, :3].astype(np.float64)
    true_check = maybe_compare_true(data_path, split, actual_3ch)
    return forecast_3ch, actual_3ch, true_check


def pearson_corr(x, y):
    """Compute Pearson correlation with a finite fallback."""
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(forecast, actual, eps=1e-6):
    """Compute per-channel metrics for [N, 168, 3] arrays."""
    rows = []
    for idx, name in enumerate(CHANNELS):
        f = forecast[:, :, idx]
        a = actual[:, :, idx]
        err = f - a
        abs_err = np.abs(err)
        denom = np.maximum(np.abs(a), eps)
        smape_denom = np.maximum((np.abs(f) + np.abs(a)) / 2.0, eps)

        rows.append({
            "variable": name,
            "mae": float(np.mean(abs_err)),
            "rmse": float(math.sqrt(np.mean(err ** 2))),
            "mape": float(np.mean(abs_err / denom)),
            "smape": float(np.mean(abs_err / smape_denom)),
            "pearson_corr": pearson_corr(f, a),
            "bias": float(np.mean(err)),
            "max_error": float(np.max(abs_err)),
            "p90_abs_error": float(np.percentile(abs_err, 90)),
        })
    return rows


def save_outputs(rows, output_dir):
    """Save metrics as CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = Path(output_dir) / "forecast_quality.csv"
    json_path = Path(output_dir) / "forecast_quality.json"

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({row["variable"]: row for row in rows}, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate wind/pv/load forecast quality.")
    parser.add_argument("--data_path", default="./input_4.27/", help="Processed data directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "legacy"])
    parser.add_argument("--output_dir", default="./outputs/forecast_quality/")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    forecast, actual, true_check = load_split(args.data_path, args.split)
    print(
        "Reconstructed actual: "
        f"shape={actual.shape}, min={actual.min():.6f}, "
        f"max={actual.max():.6f}, mean={actual.mean():.6f}"
    )
    if true_check is not None:
        if true_check["status"] == "aligned_by_prefix":
            print(
                "true.npy alignment check: "
                f"n={true_check['n_aligned']}, MAE={true_check['mae']:.10f}, "
                f"MaxAE={true_check['max_abs_error']:.10f}"
            )
        else:
            print(f"true.npy alignment check: {true_check['status']} ({true_check['reason']})")
    rows = compute_metrics(forecast, actual, eps=args.eps)
    csv_path, json_path = save_outputs(rows, args.output_dir)

    print("Forecast Quality:")
    for row in rows:
        print(
            f"{row['variable']}: "
            f"MAE={row['mae']:.6f}, RMSE={row['rmse']:.6f}, "
            f"sMAPE={row['smape']:.6f}, Corr={row['pearson_corr']:.6f}, "
            f"Bias={row['bias']:.6f}, P90AE={row['p90_abs_error']:.6f}"
        )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
