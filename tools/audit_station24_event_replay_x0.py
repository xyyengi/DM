#!/usr/bin/env python3
"""Audit B1 train-only event replay and prove generation-time isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    replay = json.loads(
        (run_dir / "event_replay.json").read_text(encoding="utf-8")
    )
    generation = json.loads(
        (result_dir / "generation_metadata.json").read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        run_dir / "checkpoints" / "model_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    model = config["model"]

    checks = {
        "train_only_fit": replay.get("fit_split") == "train",
        "future_actual_not_condition": not bool(
            replay.get("future_actual_used_as_condition", True)
        ),
        "not_applied_to_validation_or_generation": not bool(
            replay.get("applied_to_validation_or_generation", True)
        ),
        "generation_metadata_disables_replay": not bool(
            generation.get("event_replay_applied_during_generation", True)
        ),
        "ordinary_epsilon_loss_not_reweighted": not bool(
            replay.get("ordinary_epsilon_loss_reweighted", True)
        ),
        "forecast_correction_disabled": str(
            model.get("forecast_correction_mode", "none")
        )
        == "none",
        "legacy_hour_weighting_disabled": not bool(
            model.get("use_extreme_event_weighting", False)
        ),
        "test_locked": generation.get("split") == "val"
        and not bool(generation.get("test_used", True)),
        "independent_representatives_only": int(
            replay.get("representative_issue_count", -1)
        )
        == int(replay.get("independent_event_count", -2)),
        "overlap_not_amplified": int(replay.get("representative_issue_count", 0))
        <= int(replay.get("overlapping_issue_count", -1)),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"event replay audit failed: {failed}")

    catalog = pd.DataFrame(replay["catalog"])
    catalog.to_csv(output_dir / "independent_event_catalog.csv", index=False)
    tier_counts = {
        str(int(tier)): int(count)
        for tier, count in catalog.groupby("tier").size().items()
    }
    summary = {
        "method": replay["method"],
        "checks": checks,
        "independent_event_count": int(replay["independent_event_count"]),
        "tier_counts": tier_counts,
        "overlapping_issue_count": int(replay["overlapping_issue_count"]),
        "representative_issue_count": int(replay["representative_issue_count"]),
        "expected_event_draws_per_epoch": float(
            replay["expected_event_draws_per_epoch"]
        ),
        "train_issue_count": len(replay["sample_replay_weights"]),
        "event_window_hours": int(replay["event_window_hours"]),
        "merge_gap_hours": int(replay["merge_gap_hours"]),
        "replay_weights": replay["replay_weights"],
        "x0_loss_weights": {
            "magnitude": float(model["event_x0_magnitude_loss_weight"]),
            "timing": float(model["event_x0_timing_loss_weight"]),
            "synchrony": float(model["event_x0_sync_loss_weight"]),
        },
        "generation": {
            "split": generation["split"],
            "n_samples": int(generation["n_samples"]),
            "generation_seed": int(generation["generation_seed"]),
            "event_replay_applied": False,
        },
    }
    (output_dir / "event_replay_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# B1 event replay audit",
        "",
        f"- Independent physical events: {summary['independent_event_count']}",
        f"- Tier counts: {tier_counts}",
        f"- Overlapping issue windows represented: {summary['overlapping_issue_count']}",
        f"- Representative issues replayed: {summary['representative_issue_count']}",
        f"- Expected event draws per epoch: {summary['expected_event_draws_per_epoch']:.2f}/{summary['train_issue_count']}",
        f"- Generation: validation, {summary['generation']['n_samples']} members, replay labels disabled",
        "",
        "All target-derived labels are restricted to the train sampler and x0 loss. "
        "They are not denoiser conditions and are not used by validation or generation.",
    ]
    (output_dir / "event_replay_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("EVENT_REPLAY_AUDIT_PASSED")


if __name__ == "__main__":
    main()
