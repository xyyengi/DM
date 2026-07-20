#!/usr/bin/env python
"""Evaluate saved V4-s scenarios against the frozen Shandong event map."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.event_level_evaluation import evaluate_coupling, evaluate_event_mappings  # noqa: E402


RUN = ROOT / "outputs_shandong/20260718_232509_v4s_residual_event_sampler_no_guidance_168h"
BASE = ROOT / "outputs_shandong/event_evaluation/saved_runs_event_reevaluation"
OUT = ROOT / "outputs_shandong/event_evaluation/v4s_analysis"
EVENT_TYPES = (
    "low_wind", "low_renewable", "low_solar_daily_energy", "high_load",
    "high_net_load", "high_ramp_6h", "high_ramp_12h",
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coerce_mapping(row: dict) -> dict:
    result = dict(row)
    for key in ("lead_hours", "event_start_index", "event_end_index", "overlap_hours"):
        result[key] = int(float(result[key]))
    for key in ("contains_event_start", "fully_contains_event", "post_onset"):
        result[key] = str(result[key]).lower() == "true"
    return result


def coerce_event(row: dict) -> dict:
    result = dict(row)
    result["duration_hours"] = float(result["duration_hours"])
    return result


def main() -> None:
    samples = np.load(RUN / "actual_scenarios.npy", mmap_mode="r")
    actual = np.load(RUN / "actual_data.npy", mmap_mode="r")
    rng = np.random.default_rng(20260718)
    member_indices = np.sort(rng.choice(samples.shape[1], size=20, replace=False))
    selected = np.asarray(samples[:, member_indices])

    mappings = [coerce_mapping(row) for row in read_csv(BASE / "window_event_map_main_p10_p90_gap1.csv")]
    events = [coerce_event(row) for row in read_csv(BASE / "events_main_p10_p90_gap1.csv")]
    audit = json.loads((RUN / "logs/event_sampler_audit.json").read_text(encoding="utf-8"))
    thresholds = audit["thresholds"]
    denorm = json.loads((RUN / "denormalization_used.json").read_text(encoding="utf-8"))
    scales = {
        "wind_capacity_mw": float(denorm["scales"][0]),
        "solar_capacity_mw": float(denorm["scales"][1]),
        "load_scale_mw": float(denorm["scales"][2]),
    }
    global_ranges = np.ptp(actual, axis=(0, 2))

    event_rows, summary_rows = evaluate_event_mappings(
        selected, actual, mappings, events, thresholds, scales, global_ranges,
        EVENT_TYPES, "p10", "p90",
    )
    for row in event_rows:
        row.update({"model": "V4-s", "member_count": 20, "member_selection": "seeded_random_subset_of_saved_50"})
    for row in summary_rows:
        row.update({"model": "V4-s", "spec": "main_p10_p90_gap1", "member_count": 20})

    coupling_rows = evaluate_coupling(selected, actual, mappings, thresholds)
    for row in coupling_rows:
        row.update({"model": "V4-s", "member_count": 20})

    baseline = [
        row for row in read_csv(BASE / "event_summary_all_specs.csv")
        if row.get("model") == "V4" and row.get("spec") == "main_p10_p90_gap1"
    ]
    baseline_by_key = {(row["event_type"], row["lead_group"]): row for row in baseline}
    comparison = []
    for row in summary_rows:
        old = baseline_by_key.get((row["event_type"], row["lead_group"]), {})
        comparison.append({
            "event_type": row["event_type"],
            "lead_group": row["lead_group"],
            "n_events": row["n_events"],
            "n_windows": row["n_windows"],
            "v4_total_crps_mw": old.get("onset_total_crps_mw", ""),
            "v4s_total_crps_mw": row.get("onset_total_crps_mw", ""),
            "v4_total_coverage_90": old.get("onset_total_coverage_90", ""),
            "v4s_total_coverage_90": row.get("onset_total_coverage_90", ""),
            "v4_total_width_90": old.get("onset_total_width_90_pct_range", ""),
            "v4s_total_width_90": row.get("onset_total_width_90_pct_range", ""),
            "v4_peak_mae": old.get("full_peak_mae", ""),
            "v4s_peak_mae": row.get("full_peak_mae", ""),
            "v4_duration_mae_hours": old.get("full_duration_mae_hours", ""),
            "v4s_duration_mae_hours": row.get("full_duration_mae_hours", ""),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "v4s_event_by_id_main_20members.csv", event_rows)
    write_csv(OUT / "v4s_event_summary_main_20members.csv", summary_rows)
    write_csv(OUT / "v4s_vs_v4_event_comparison_20members.csv", comparison)
    write_csv(OUT / "v4s_coupling_by_event_20members.csv", coupling_rows)
    (OUT / "v4s_event_evaluation_metadata.json").write_text(json.dumps({
        "run": RUN.name,
        "saved_members": int(samples.shape[1]),
        "evaluated_members": 20,
        "member_indices": member_indices.tolist(),
        "event_definition": "main_p10_p90_gap1 frozen map",
        "no_training_or_generation": True,
        "n_event_summary_rows": len(summary_rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"member_indices": member_indices.tolist(), "summary_rows": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
