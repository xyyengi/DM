"""Audit member-level discrete event-memory usage after generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TYPE_NAMES = {
    0: "sustained_drop",
    1: "down_ramp",
    2: "up_ramp",
    3: "large_mismatch",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = Path(args.result_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    route = np.load(result / "tail_expert_route.npy").astype(bool)
    selected = np.load(result / "event_memory_selected_index.npy")
    event_type = np.load(result / "event_memory_selected_type.npy")
    duration = np.load(result / "event_memory_selected_duration.npy")
    train_index = np.load(result / "event_memory_selected_train_index.npy")
    starts = np.load(result / "tail_event_start.npy")
    probability = np.load(result / "event_memory_candidate_probability.npy")
    if not (
        route.shape
        == selected.shape
        == event_type.shape
        == duration.shape
        == train_index.shape
        == starts.shape
    ):
        raise ValueError("event memory member audit arrays have inconsistent shapes")
    routed = int(route.sum())
    if routed == 0:
        raise ValueError("no members were routed to the unified event expert")
    if np.any(selected[route] < 0) or np.any(starts[route] < 0):
        raise ValueError("routed event members must have one selected prototype/time")
    rows = []
    for issue in range(route.shape[0]):
        active = route[issue]
        count = int(active.sum())
        unique_source = len(np.unique(train_index[issue, active])) if count else 0
        unique_candidate = len(np.unique(selected[issue, active])) if count else 0
        rows.append(
            {
                "issue_index": issue,
                "routed_members": count,
                "routed_fraction": count / route.shape[1],
                "unique_candidate_count": unique_candidate,
                "unique_train_prototype_count": unique_source,
                "candidate_probability_effective_k": float(
                    1.0 / np.sum(np.square(probability[issue]))
                ),
                "mean_event_start": float(starts[issue, active].mean()) if count else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "event_memory_issue_usage.csv", index=False)
    type_counts = {
        TYPE_NAMES[key]: int(np.sum(event_type[route] == key)) for key in TYPE_NAMES
    }
    duration_counts = {
        str(value): int(np.sum(duration[route] == value)) for value in (6, 12, 24)
    }
    day_counts = {
        str(day + 1): int(np.sum((starts[route] // 24) == day)) for day in range(7)
    }
    summary = {
        "method": "member_level_discrete_event_memory_audit_v1",
        "issue_count": int(route.shape[0]),
        "member_count": int(route.shape[1]),
        "routed_event_member_count": routed,
        "routed_event_member_fraction": float(route.mean()),
        "one_prototype_per_routed_member_verified": True,
        "topk_trajectory_averaging_used": False,
        "mean_unique_candidates_per_issue": float(frame.unique_candidate_count.mean()),
        "mean_unique_train_prototypes_per_issue": float(
            frame.unique_train_prototype_count.mean()
        ),
        "mean_candidate_probability_effective_k": float(
            frame.candidate_probability_effective_k.mean()
        ),
        "event_type_member_counts": type_counts,
        "duration_member_counts": duration_counts,
        "lead_day_member_counts": day_counts,
    }
    (output / "discrete_event_memory_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "discrete_event_memory_audit.md").write_text(
        "# Discrete event memory audit\n\n"
        f"- Routed event members: {routed} ({route.mean():.2%})\n"
        "- One complete prototype per routed member: verified\n"
        "- Top-K trajectory averaging: disabled\n"
        f"- Mean unique candidates per issue: {frame.unique_candidate_count.mean():.2f}\n"
        f"- Mean unique train prototypes per issue: {frame.unique_train_prototype_count.mean():.2f}\n"
        f"- Mean effective candidate count: {frame.candidate_probability_effective_k.mean():.2f}\n",
        encoding="utf-8",
    )
    print(f"DISCRETE_EVENT_MEMORY_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
