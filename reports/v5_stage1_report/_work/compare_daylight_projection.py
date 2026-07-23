"""Read-only comparison of forecast-threshold and astronomical solar masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\DM_local")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.physical_projection import (
    daylight_mask_from_export_metadata,
    daylight_mask_from_train_support,
    project_power_scenarios,
)


RUNS = {
    "V4-RS": ROOT / "outputs_shandong/v5_stage1/20260722_143437_v4rs_repro_stage1_seed2026_20260722_143431_val_rank1_epoch11_posterior_n20_seed424242",
    "V5-T": ROOT / "outputs_shandong/v5_stage1/20260722_151755_v5_t_stage1_seed2026_20260722_151749_val_rank1_epoch29_posterior_n20_seed424242",
    "V5-TF": ROOT / "outputs_shandong/v5_stage1/20260722_155013_v5_tf_stage1_seed2026_20260722_155007_val_rank1_epoch8_posterior_n20_seed424242",
}


def solar_coverage(actual: np.ndarray, scenarios: np.ndarray, mask: np.ndarray) -> float:
    lower, upper = np.quantile(scenarios[:, :, 1, :], [0.05, 0.95], axis=1)
    covered = (actual[:, 1, :] >= lower) & (actual[:, 1, :] <= upper)
    return float(np.mean(covered[mask]) * 100.0)


def compact_metrics(actual: np.ndarray, scenarios: np.ndarray) -> dict[str, float]:
    members = scenarios.shape[1]
    lower, upper = np.quantile(scenarios, [0.05, 0.95], axis=1)
    coverage = np.mean((actual >= lower) & (actual <= upper), axis=(0, 2)) * 100.0
    ranges = np.ptp(actual, axis=(0, 2))
    width = (
        np.mean(upper - lower, axis=(0, 2)) / np.maximum(ranges, 1e-12) * 100.0
    )
    term1 = np.mean(np.abs(scenarios - actual[:, None, :, :]), axis=1)
    ordered = np.sort(scenarios, axis=1)
    weights = (
        2 * np.arange(1, members + 1) - members - 1
    ).reshape(1, members, 1, 1)
    term2 = np.sum(ordered * weights, axis=1) / (members * (members - 1))
    crps = np.mean(term1 - term2, axis=(0, 2))
    return {
        "total_crps": float(np.mean(crps)),
        "total_coverage_90_pct": float(np.mean(coverage)),
        "total_width_90_pct": float(np.mean(width)),
    }


def main() -> None:
    first = next(iter(RUNS.values()))
    shape = np.load(first / "actual_scenarios.npy", mmap_mode="r").shape
    astronomical_midpoint, midpoint_audit = daylight_mask_from_export_metadata(
        ROOT / "diffusion_npy_normalized/export_metadata.json",
        "val",
        window_count=shape[0],
        sequence_length=shape[3],
    )
    astronomical_start, _ = daylight_mask_from_export_metadata(
        ROOT / "diffusion_npy_normalized/export_metadata.json",
        "val",
        window_count=shape[0],
        sequence_length=shape[3],
        timestamp_offset_minutes=0.0,
    )
    astronomical_end, _ = daylight_mask_from_export_metadata(
        ROOT / "diffusion_npy_normalized/export_metadata.json",
        "val",
        window_count=shape[0],
        sequence_length=shape[3],
        timestamp_offset_minutes=60.0,
    )
    astronomical = astronomical_start | astronomical_midpoint | astronomical_end
    daylight, daylight_audit = daylight_mask_from_train_support(
        ROOT / "diffusion_npy_normalized",
        "val",
        window_count=shape[0],
        sequence_length=shape[3],
    )
    output = {
        "daylight_audit": daylight_audit,
        "astronomical_full_hour_daylight_pct": float(
            np.mean(astronomical) * 100.0
        ),
        "train_support_vs_astronomical_disagreement_pct": float(
            np.mean(daylight != astronomical) * 100.0
        ),
        "models": {},
    }
    for name, path in RUNS.items():
        raw = np.load(path / "actual_scenarios.npy")
        actual = np.load(path / "actual_data.npy")
        forecast = np.load(path / "forecast_data.npy")
        with (path / "denormalization_used.json").open("r", encoding="utf-8") as handle:
            scales = np.asarray(json.load(handle)["scales"], dtype=np.float64)
        old, old_report = project_power_scenarios(raw, forecast, scales)
        new, new_report = project_power_scenarios(
            raw,
            forecast,
            scales,
            solar_daylight_mask=daylight,
            solar_daylight_metadata=daylight_audit,
        )
        old_metrics = compact_metrics(actual, old)
        new_metrics = compact_metrics(actual, new)
        old_night = forecast[:, 1, :] <= 1.0
        output["models"][name] = {
            "forecast_threshold_night_pct": float(np.mean(old_night) * 100.0),
            "train_support_night_pct": float(np.mean(~daylight) * 100.0),
            "mask_disagreement_pct": float(np.mean(old_night != ~daylight) * 100.0),
            "actual_positive_outside_train_support_pct": float(
                np.mean(actual[:, 1, :][~daylight] > 1.0) * 100.0
            ),
            "actual_max_outside_train_support_mw": float(
                np.max(actual[:, 1, :][~daylight])
            ),
            "old_total_crps": float(old_metrics["total_crps"]),
            "new_total_crps": float(new_metrics["total_crps"]),
            "old_total_coverage_90_pct": old_metrics["total_coverage_90_pct"],
            "new_total_coverage_90_pct": new_metrics["total_coverage_90_pct"],
            "old_total_width_90_pct": old_metrics["total_width_90_pct"],
            "new_total_width_90_pct": new_metrics["total_width_90_pct"],
            "old_solar_daylight_coverage_90_pct": solar_coverage(
                actual, old, daylight
            ),
            "new_solar_daylight_coverage_90_pct": solar_coverage(
                actual, new, daylight
            ),
            "old_changed_scalar_pct": old_report["changed_scalar_pct"],
            "new_changed_scalar_pct": new_report["changed_scalar_pct"],
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
