#!/usr/bin/env python3
"""Attribute Full-V2 event hits before and after the frozen Original-S5 localization.

This is an evaluation-only audit.  It never changes generated members and it
never uses event information to select a localization window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY = {"onset": 6.0, "recall": 0.50, "depth": 0.75}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-mixture", required=True, type=Path)
    parser.add_argument("--tail100", required=True, type=Path)
    parser.add_argument("--full-event-evaluation", required=True, type=Path)
    parser.add_argument("--localized", required=True, type=Path)
    parser.add_argument("--localized-event-evaluation", required=True, type=Path)
    parser.add_argument("--event-targets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def primary_hit(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["has_candidate"].astype(bool)
        & frame["onset_abs_error_h"].le(PRIMARY["onset"])
        & frame["interval_recall"].ge(PRIMARY["recall"])
        & frame["depth_ratio"].ge(PRIMARY["depth"])
    )


def candidate_rows(path: Path, expected_label: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "continuous_event_member_matches.csv")
    frame = frame[(frame.scope == "independent_physical") & (frame.variant == expected_label)].copy()
    if len(frame) != 4 * 500:
        raise ValueError(f"expected 2,000 independent candidate rows at {path}, got {len(frame)}")
    frame["primary_hit"] = primary_hit(frame)
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full = candidate_rows(args.full_event_evaluation, "Independent joint tail V2")
    localized = candidate_rows(args.localized_event_evaluation, "V2 self-localized")
    keys = ["event_id", "issue", "source", "direction", "member"]
    paired = full.merge(localized, on=keys, suffixes=("_full", "_localized"), validate="one_to_one")
    paired = paired[paired.member.ge(400)].copy()
    paired["tail_member"] = paired.member - 400
    paired["transition"] = np.select(
        [
            paired.primary_hit_full & paired.primary_hit_localized,
            paired.primary_hit_full & ~paired.primary_hit_localized,
            ~paired.primary_hit_full & paired.primary_hit_localized,
        ],
        ["preserved", "lost", "gained"],
        default="neither",
    )

    targets = json.loads(args.event_targets.read_text(encoding="utf-8"))
    catalog = targets["val_catalog"]
    expected_ids = [f"{row['source']}_{row['direction']}_{index:03d}" for index, row in enumerate(catalog)]
    if set(expected_ids) != set(paired.event_id.unique()):
        raise ValueError("event catalog and event-evaluation identifiers disagree")
    event_map = {event_id: row for event_id, row in zip(expected_ids, catalog)}

    masks = np.load(args.localized / "localized_tail_mask.npy", mmap_mode="r")
    starts = np.load(args.localized / "localized_tail_start.npy", mmap_mode="r")
    stops = np.load(args.localized / "localized_tail_stop.npy", mmap_mode="r")
    if masks.shape != (23, 100, 168):
        raise ValueError(f"unexpected Original-S5 mask shape {masks.shape}")

    overlap = []
    for row in paired.itertuples():
        event = event_map[row.event_id]
        onset, stop = int(event["lead_onset"]), int(event["lead_stop_exclusive"])
        mask = np.asarray(masks[int(row.issue), int(row.tail_member)])
        truth = mask[onset:stop]
        overlap.append(
            {
                "mask_event_overlap_fraction": float(np.mean(truth > 0)) if truth.size else 0.0,
                "mask_event_mean_amplitude": float(np.mean(truth)) if truth.size else 0.0,
                "selected_start": int(starts[int(row.issue), int(row.tail_member)]),
                "selected_stop": int(stops[int(row.issue), int(row.tail_member)]),
                "covers_post_event_3h": bool(stops[int(row.issue), int(row.tail_member)] >= min(168, stop + 3)),
                "covers_post_event_6h": bool(stops[int(row.issue), int(row.tail_member)] >= min(168, stop + 6)),
            }
        )
    paired = pd.concat([paired.reset_index(drop=True), pd.DataFrame(overlap)], axis=1)
    paired["fails_onset_after_localization"] = paired.onset_abs_error_h_localized.gt(PRIMARY["onset"])
    paired["fails_interval_after_localization"] = paired.interval_recall_localized.lt(PRIMARY["recall"])
    paired["fails_depth_after_localization"] = paired.depth_ratio_localized.lt(PRIMARY["depth"])
    paired["window_misses_most_truth"] = paired.mask_event_overlap_fraction.lt(PRIMARY["recall"])
    paired["amplitude_attenuated_below_primary"] = (
        paired.primary_hit_full & paired.fails_depth_after_localization
    )
    paired["possible_recovery_segment_loss"] = (
        paired.direction.eq("negative") & ~paired.covers_post_event_6h
    )

    transition_columns = [
        "event_id", "issue", "source", "direction", "member", "tail_member", "transition",
        "primary_hit_full", "primary_hit_localized", "onset_abs_error_h_full",
        "onset_abs_error_h_localized", "interval_recall_full", "interval_recall_localized",
        "depth_ratio_full", "depth_ratio_localized", "selected_start", "selected_stop",
        "mask_event_overlap_fraction", "mask_event_mean_amplitude", "covers_post_event_3h",
        "covers_post_event_6h", "fails_onset_after_localization",
        "fails_interval_after_localization", "fails_depth_after_localization",
        "window_misses_most_truth", "amplitude_attenuated_below_primary",
        "possible_recovery_segment_loss",
    ]
    paired[transition_columns].to_csv(args.output_dir / "s5_member_transitions.csv", index=False)

    attribution_rows = []
    for event_id, group in paired.groupby("event_id", sort=False):
        full_hits = int(group.primary_hit_full.sum())
        local_hits = int(group.primary_hit_localized.sum())
        preserved = int((group.transition == "preserved").sum())
        lost = int((group.transition == "lost").sum())
        gained = int((group.transition == "gained").sum())
        lost_rows = group[group.transition == "lost"]
        attribution_rows.append(
            {
                "event_id": event_id,
                "source": group.source.iloc[0],
                "direction": group.direction.iloc[0],
                "full_tail_hits": full_hits,
                "localized_tail_hits": local_hits,
                "preserved_hits": preserved,
                "lost_hits": lost,
                "gained_hits": gained,
                "retention_rate_of_full_hits": preserved / full_hits if full_hits else np.nan,
                "loss_rate_of_full_hits": lost / full_hits if full_hits else np.nan,
                "lost_with_window_miss": int(lost_rows.window_misses_most_truth.sum()),
                "lost_with_onset_failure": int(lost_rows.fails_onset_after_localization.sum()),
                "lost_with_interval_failure": int(lost_rows.fails_interval_after_localization.sum()),
                "lost_with_depth_failure": int(lost_rows.fails_depth_after_localization.sum()),
                "lost_with_possible_recovery_loss": int(lost_rows.possible_recovery_segment_loss.sum()),
            }
        )
    attribution = pd.DataFrame(attribution_rows)
    attribution.to_csv(args.output_dir / "s5_event_member_attribution.csv", index=False)

    # Separate fixed 20% Tail quota dilution from localization loss.
    quota_rows = []
    for event_id in expected_ids:
        full_event = full[full.event_id == event_id]
        local_event = localized[localized.event_id == event_id]
        raw400_hits = int(full_event[full_event.member.lt(400)].primary_hit.sum())
        full_tail_hits = int(full_event[full_event.member.ge(400)].primary_hit.sum())
        local_tail_hits = int(local_event[local_event.member.ge(400)].primary_hit.sum())
        quota_rows.append(
            {
                "event_id": event_id,
                "raw400_hits": raw400_hits,
                "full_tail100_hits": full_tail_hits,
                "localized_tail100_hits": local_tail_hits,
                "full_tail_only_hit_rate": full_tail_hits / 100,
                "localized_tail_only_hit_rate": local_tail_hits / 100,
                "full_tail_contribution_to_500": full_tail_hits / 500,
                "localized_tail_contribution_to_500": local_tail_hits / 500,
                "quota_dilution_from_tail_only_rate": 0.8 * full_tail_hits / 100,
                "localization_loss_in_tail_rate": (full_tail_hits - local_tail_hits) / 100,
                "localization_loss_in_500_mixture_rate": (full_tail_hits - local_tail_hits) / 500,
                "full_mixture_hit_rate": (raw400_hits + full_tail_hits) / 500,
                "localized_mixture_hit_rate": (raw400_hits + local_tail_hits) / 500,
            }
        )
    quota = pd.DataFrame(quota_rows)
    quota.to_csv(args.output_dir / "s5_quota_vs_localization_attribution.csv", index=False)

    full_mixture = np.load(args.full_mixture / "actual_scenarios_normalized.npy", mmap_mode="r")
    tail100 = np.load(args.tail100 / "actual_scenarios_normalized.npy", mmap_mode="r")
    local_scenarios = np.load(args.localized / "actual_scenarios_normalized.npy", mmap_mode="r")
    identity = {
        "full_last100_equals_tail100": bool(np.array_equal(full_mixture[:, 400:500], tail100[:, :100])),
        "full_and_localized_first400_equal": bool(np.array_equal(full_mixture[:, :400], local_scenarios[:, :400])),
        "member_identity_traceable": True,
        "primary_standard": PRIMARY,
        "notes": (
            "Tail member j is Full/localized ensemble member 400+j. Mechanism flags are "
            "post-hoc descriptions; event labels were not used to choose Original-S5 windows."
        ),
    }
    identity["member_identity_traceable"] = bool(
        identity["full_last100_equals_tail100"] and identity["full_and_localized_first400_equal"]
    )
    (args.output_dir / "s5_member_identity_audit.json").write_text(
        json.dumps(identity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not identity["member_identity_traceable"]:
        raise ValueError("member identity audit failed; refusing to claim member-level attribution")

    summary = {
        "status": "PASS",
        "tail_hits_full_total_over_four_events": int(paired.primary_hit_full.sum()),
        "tail_hits_localized_total_over_four_events": int(paired.primary_hit_localized.sum()),
        "preserved": int((paired.transition == "preserved").sum()),
        "lost": int((paired.transition == "lost").sum()),
        "gained": int((paired.transition == "gained").sum()),
        "mean_event_retention_rate": float(attribution.retention_rate_of_full_hits.mean()),
        "mean_full_tail_hit_rate": float(quota.full_tail_only_hit_rate.mean()),
        "mean_localized_tail_hit_rate": float(quota.localized_tail_only_hit_rate.mean()),
        "mean_quota_dilution": float(quota.quota_dilution_from_tail_only_rate.mean()),
        "mean_localization_loss_in_500": float(quota.localization_loss_in_500_mixture_rate.mean()),
    }
    (args.output_dir / "s5_member_attribution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"S5_MEMBER_ATTRIBUTION_COMPLETE output={args.output_dir}")


if __name__ == "__main__":
    main()
