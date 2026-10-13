#!/usr/bin/env python3
"""Read-only audit of train strong-drop/recovery support for Station-24."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--spacing-hours", default=6, type=int)
    return parser.parse_args()


def recovery(series: np.ndarray, end: int, drop: float) -> dict[str, float | bool]:
    future = series[end : end + 7]
    if future.size < 7:
        return {"usable_6h": False}
    gain = future[1:] - future[0]
    positive_steps = np.diff(future) > 0
    half = np.flatnonzero(gain >= 0.5 * drop)
    maximum = max(float(gain.max()), 0.0)
    return {
        "usable_6h": True,
        "any_positive_step_1h": bool(positive_steps[:1].any()),
        "any_positive_step_3h": bool(positive_steps[:3].any()),
        "any_positive_step_6h": bool(positive_steps.any()),
        "recovery_1h_mw": max(float(gain[0]), 0.0),
        "recovery_3h_mw": max(float(gain[:3].max()), 0.0),
        "recovery_6h_mw": maximum,
        "recovery_ratio_1h": max(float(gain[0]), 0.0) / drop,
        "recovery_ratio_3h": max(float(gain[:3].max()), 0.0) / drop,
        "recovery_ratio_6h": maximum / drop,
        "half_recovered_3h": bool((gain[:3] >= 0.5 * drop).any()),
        "half_recovered_6h": bool(half.size),
        "time_to_half_recovery_h": float(half[0] + 1) if half.size else np.nan,
    }


def greedy_nms(rows: list[dict[str, object]], spacing: int, physical: bool) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: float(item["drop_mw"]), reverse=True):
        if physical:
            time = pd.Timestamp(row["timestamp"])
            conflict = any(abs((time - pd.Timestamp(old["timestamp"])).total_seconds()) <= spacing * 3600 for old in selected)
        else:
            conflict = any(
                int(old["sample_index"]) == int(row["sample_index"])
                and abs(int(old["lead_end"]) - int(row["lead_end"])) <= spacing
                for old in selected
            )
        if not conflict:
            selected.append(row)
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual = np.load(args.data_path / "train_actual.npy", mmap_mode="r")
    valid = np.asarray(np.load(args.data_path / "train_fill_mask.npy", mmap_mode="r")) == 0
    issues = pd.read_csv(args.data_path / "train_issue_dates.csv")
    stations = pd.read_csv(args.data_path / "station_order.csv").sort_values("channel_index")
    wind = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacity = stations.capacity_mw.to_numpy(float)
    weights = capacity[wind]
    aggregate = np.einsum("nts,s->nt", np.asarray(actual[..., wind], dtype=np.float64), weights)
    hour_valid = valid[..., wind].all(axis=-1)
    delta = aggregate[:, 1:] - aggregate[:, :-1]
    pair_valid = hour_valid[:, 1:] & hour_valid[:, :-1]
    negative = -delta[pair_valid & (delta < 0)]
    threshold = float(np.quantile(negative, 0.90))

    candidates: list[dict[str, object]] = []
    for sample, offset in zip(*np.where(pair_valid & (delta <= -threshold))):
        lead_end = int(offset + 1)
        if lead_end + 6 >= aggregate.shape[1] or not hour_valid[sample, lead_end : lead_end + 7].all():
            continue
        start = pd.Timestamp(issues.loc[int(sample), "target_start"])
        timestamp = start + pd.Timedelta(hours=lead_end)
        drop = float(-delta[sample, offset])
        row: dict[str, object] = {
            "sample_index": int(sample), "lead_end": lead_end, "timestamp": timestamp,
            "month": int(timestamp.month), "hour": int(timestamp.hour),
            "lead_day": int(lead_end // 24 + 1), "drop_mw": drop,
        }
        row.update(recovery(aggregate[sample], lead_end, drop))
        candidates.append(row)

    window_nms = greedy_nms(candidates, args.spacing_hours, physical=False)
    # The same physical hour appears in overlapping 168 h windows.  Keep one
    # representation before physical-time NMS so unique-event evidence is not inflated.
    by_time: dict[pd.Timestamp, dict[str, object]] = {}
    for row in candidates:
        time = pd.Timestamp(row["timestamp"])
        if time not in by_time or float(row["drop_mw"]) > float(by_time[time]["drop_mw"]):
            by_time[time] = row
    physical_nms = greedy_nms(list(by_time.values()), args.spacing_hours, physical=True)

    candidate_frame = pd.DataFrame(candidates)
    window_frame = pd.DataFrame(window_nms)
    physical_frame = pd.DataFrame(physical_nms).sort_values("timestamp")
    candidate_frame.to_csv(args.output_dir / "train_strong_drop_candidates.csv", index=False)
    window_frame.to_csv(args.output_dir / "train_window_nms_anchors.csv", index=False)
    physical_frame.to_csv(args.output_dir / "train_unique_physical_anchors.csv", index=False)

    distributions = []
    for label, frame in (("window_exposures", window_frame), ("unique_physical", physical_frame)):
        for field in ("month", "hour", "lead_day"):
            for value, group in frame.groupby(field):
                distributions.append({"population": label, "dimension": field, "value": value, "count": len(group)})
    pd.DataFrame(distributions).to_csv(args.output_dir / "train_anchor_time_distribution.csv", index=False)

    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "count": int(len(frame)),
            "median_drop_mw": float(frame.drop_mw.median()),
            "q90_drop_mw": float(frame.drop_mw.quantile(0.90)),
            "half_recovery_probability_3h": float(frame.half_recovered_3h.mean()),
            "half_recovery_probability_6h": float(frame.half_recovered_6h.mean()),
            "any_positive_step_probability_1h": float(frame.any_positive_step_1h.mean()),
            "any_positive_step_probability_3h": float(frame.any_positive_step_3h.mean()),
            "any_positive_step_probability_6h": float(frame.any_positive_step_6h.mean()),
            "median_recovery_ratio_3h": float(frame.recovery_ratio_3h.median()),
            "median_recovery_ratio_6h": float(frame.recovery_ratio_6h.median()),
            "median_time_to_half_recovery_h_among_hits": float(frame.time_to_half_recovery_h.dropna().median()),
        }

    summary = {
        "status": "PASS",
        "test_used": False,
        "threshold_definition": "train aggregate-wind valid negative 1h ramp q90",
        "threshold_mw": threshold,
        "valid_negative_1h_count": int(negative.size),
        "raw_window_candidate_count": int(len(candidate_frame)),
        "window_nms": summarize(window_frame),
        "unique_physical_nms": summarize(physical_frame),
        "warning": (
            "window_exposures count repeated appearances of the same physical trajectory in overlapping 168 h windows; "
            "unique_physical is the independent-time support and must be used for evidence strength."
        ),
    }
    (args.output_dir / "drop_recovery_training_support_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"DROP_RECOVERY_TRAIN_SUPPORT_AUDIT_COMPLETE output={args.output_dir}")


if __name__ == "__main__":
    main()
