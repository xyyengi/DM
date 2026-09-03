#!/usr/bin/env python3
"""Audit sustained-event hit sensitivity without generation or training.

The input is the per-member match table produced by
``diagnose_station24_sustained_drop_tail_sweep.py``.  The audit keeps each
member's selected candidate event fixed and varies only the three acceptance
thresholds, so changes are attributable to the hit definition rather than to
candidate re-ranking.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import pandas as pd


ONSET_TOLERANCES = (3, 6, 12)
INTERVAL_RECALLS = (0.25, 0.50, 0.75)
DEPTH_RATIOS = (0.50, 0.75, 1.00)
PRIMARY_STANDARD = (6, 0.50, 0.75)
LEGACY_STANDARD = (12, 0.50, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "variant",
        "event_id",
        "member_index",
        "member_group",
        "has_candidate",
        "coverage_hit",
        "onset_abs_error_h",
        "true_interval_recall",
        "depth_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"match table lacks columns: {sorted(missing)}")
    output = frame.copy()
    for column in ("has_candidate", "coverage_hit"):
        output[column] = output[column].astype(str).str.lower().eq("true")
    for column in (
        "member_index",
        "onset_abs_error_h",
        "true_interval_recall",
        "depth_ratio",
    ):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def add_all_group(frame: pd.DataFrame) -> pd.DataFrame:
    all_members = frame.copy()
    all_members["member_group"] = "all"
    return pd.concat([frame, all_members], ignore_index=True)


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_event_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    grouped = add_all_group(frame).groupby(
        ["variant", "member_group", "event_id"], sort=False
    )
    for onset_h, interval_recall, depth_ratio in product(
        ONSET_TOLERANCES, INTERVAL_RECALLS, DEPTH_RATIOS
    ):
        event_rows: list[dict[str, object]] = []
        for (variant, member_group, event_id), event in grouped:
            hit = (
                event["has_candidate"]
                & event["onset_abs_error_h"].le(onset_h)
                & event["true_interval_recall"].ge(interval_recall)
                & event["depth_ratio"].ge(depth_ratio)
            )
            row = {
                "variant": variant,
                "member_group": member_group,
                "event_id": event_id,
                "onset_tolerance_h": onset_h,
                "interval_recall_required": interval_recall,
                "depth_ratio_required": depth_ratio,
                "member_count": int(len(event)),
                "hit_count": int(hit.sum()),
                "hit_rate": float(hit.mean()),
                "any_hit": bool(hit.any()),
            }
            event_rows.append(row)
            per_event_rows.append(row)
        event_frame = pd.DataFrame(event_rows)
        for (variant, member_group), group in event_frame.groupby(
            ["variant", "member_group"], sort=False
        ):
            summary_rows.append(
                {
                    "variant": variant,
                    "member_group": member_group,
                    "onset_tolerance_h": onset_h,
                    "interval_recall_required": interval_recall,
                    "depth_ratio_required": depth_ratio,
                    "event_count": int(len(group)),
                    "events_with_any_hit": int(group["any_hit"].sum()),
                    "event_any_hit_rate": float(group["any_hit"].mean()),
                    "total_hit_members": int(group["hit_count"].sum()),
                    "mean_member_hit_rate": float(group["hit_rate"].mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(per_event_rows)


def select_standard(summary: pd.DataFrame, standard: tuple[int, float, float]) -> pd.DataFrame:
    onset_h, interval_recall, depth_ratio = standard
    return summary[
        summary["onset_tolerance_h"].eq(onset_h)
        & summary["interval_recall_required"].eq(interval_recall)
        & summary["depth_ratio_required"].eq(depth_ratio)
    ].copy()


def legacy_assertion(frame: pd.DataFrame) -> None:
    expected = (
        frame["has_candidate"]
        & frame["onset_abs_error_h"].le(LEGACY_STANDARD[0])
        & frame["true_interval_recall"].ge(LEGACY_STANDARD[1])
        & frame["depth_ratio"].ge(LEGACY_STANDARD[2])
    )
    if not expected.equals(frame["coverage_hit"]):
        mismatches = int((expected != frame["coverage_hit"]).sum())
        raise ValueError(f"legacy hit reconstruction differs on {mismatches} rows")


def main() -> None:
    args = parse_args()
    matches = normalize(pd.read_csv(args.matches))
    legacy_assertion(matches)
    summary, per_event = evaluate(matches)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "threshold_sensitivity_summary.csv", index=False)
    per_event.to_csv(output / "threshold_sensitivity_per_event.csv", index=False)
    select_standard(summary, PRIMARY_STANDARD).to_csv(
        output / "recommended_primary_standard.csv", index=False
    )
    select_standard(summary, LEGACY_STANDARD).to_csv(
        output / "legacy_standard.csv", index=False
    )

    baseline = summary[summary["variant"].eq("baseline")]
    ranges = (
        baseline.groupby("member_group", sort=False)
        .agg(
            min_member_hit_rate=("mean_member_hit_rate", "min"),
            max_member_hit_rate=("mean_member_hit_rate", "max"),
            min_events_with_any_hit=("events_with_any_hit", "min"),
            max_events_with_any_hit=("events_with_any_hit", "max"),
        )
        .reset_index()
    )
    ranges.to_csv(output / "baseline_sensitivity_ranges.csv", index=False)

    metadata = {
        "method": "fixed_candidate_threshold_sensitivity_v1",
        "generation_or_training_performed": False,
        "onset_tolerances_h": list(ONSET_TOLERANCES),
        "interval_recall_thresholds": list(INTERVAL_RECALLS),
        "depth_ratio_thresholds": list(DEPTH_RATIOS),
        "primary_standard": {
            "onset_tolerance_h": PRIMARY_STANDARD[0],
            "interval_recall_required": PRIMARY_STANDARD[1],
            "depth_ratio_required": PRIMARY_STANDARD[2],
        },
        "legacy_standard": {
            "onset_tolerance_h": LEGACY_STANDARD[0],
            "interval_recall_required": LEGACY_STANDARD[1],
            "depth_ratio_required": LEGACY_STANDARD[2],
        },
        "legacy_definition_reproduced_exactly": True,
        "candidate_selection_note": (
            "Each member's candidate segment is held fixed from the original audit; "
            "only acceptance thresholds vary."
        ),
    }
    (output / "audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"EVENT_HIT_SENSITIVITY_COMPLETE output={output}")


if __name__ == "__main__":
    main()
