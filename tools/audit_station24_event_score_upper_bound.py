"""Audit the two-expert localized event-score upper-bound experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def serialized_aliases(names: set[str]) -> set[str]:
    result = set(names)
    result.update(
        "diffusion." + name for name in names if name.startswith("denoiser.")
    )
    return result


def main() -> None:
    args = parse_args()
    source_run = Path(args.source_run)
    candidate_run = Path(args.candidate_run)
    source_result = Path(args.source_result)
    candidate_result = Path(args.candidate_result)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    source = torch.load(
        source_run / "checkpoints" / "model_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    candidate = torch.load(
        candidate_run / "checkpoints" / "model_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected_variant = "geo_history_actual_body_tail_event_score_upper_bound"
    if source.get("condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("source is not the Raw body-tail checkpoint")
    if candidate.get("condition_variant") != expected_variant:
        raise ValueError("candidate variant mismatch")
    if not bool(candidate.get("sampler_event_localized", False)):
        raise ValueError("localized event scoring is disabled")
    if not bool(candidate.get("sampler_temporal_body_finetune", False)):
        raise ValueError("temporal body fine-tuning is disabled")
    if int(candidate.get("sampler_body_members", 0)) != 2 or int(
        candidate.get("sampler_tail_members", 0)
    ) != 4:
        raise ValueError("formal stratified quota must be 2 body + 4 tail")

    initialization = load_json(candidate_run / "body_tail_initialization.json")
    if initialization.get("checkpoint_state_source") != "raw":
        raise ValueError("candidate was not initialized from Raw parameters")
    if bool(initialization.get("body_frozen", True)):
        raise ValueError("temporal body was unexpectedly declared frozen")
    if not bool(initialization.get("spatial_and_state_frozen", False)):
        raise ValueError("spatial/state freeze was not declared")
    if bool(initialization.get("third_expert_used", True)):
        raise ValueError("candidate introduced a third expert")

    tail_names = set(candidate.get("body_tail_trainable_parameter_names", []))
    temporal_names = set(
        candidate.get("temporal_body_trainable_parameter_names", [])
    )
    allowed = serialized_aliases(tail_names | temporal_names)
    source_state = source["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    if set(source_state) != set(candidate_state):
        raise ValueError("source and candidate structures differ")
    changed = {
        name
        for name, value in source_state.items()
        if not torch.equal(value, candidate_state[name])
    }
    outside = changed - allowed
    if outside:
        raise ValueError(f"frozen spatial/state tensors changed: {sorted(outside)[:8]}")
    changed_tail = changed & serialized_aliases(tail_names)
    changed_temporal = changed & serialized_aliases(temporal_names)
    if not changed_tail or not changed_temporal:
        raise ValueError("tail and temporal body must both receive updates")

    history = load_json(candidate_run / "logs" / "training_history.json")
    scored = [
        row
        for row in history
        if float(row.get("train_sampler_issue_count", 0.0)) > 0.0
    ]
    if not scored:
        raise ValueError("no event member set was scored")
    expected_route = 4.0 / 6.0
    if not all(
        abs(float(row.get("train_sampler_tail_route_rate", -1.0)) - expected_route)
        < 1e-5
        and float(row.get("train_sampler_temporal_variogram", -1.0)) >= 0.0
        and float(row.get("train_sampler_body_anchor", -1.0)) >= 0.0
        for row in scored
    ):
        raise ValueError("stratified route, Variogram, or body anchor audit failed")

    condition_audit = load_json(candidate_run / "condition_feature_audit.json")
    anchor_audit = condition_audit.get("sampler_score_train")
    if not isinstance(anchor_audit, dict) or bool(
        anchor_audit.get("event_replay_enabled", True)
    ):
        raise ValueError("natural issuance body-anchor loader was not isolated")

    source_meta = load_json(source_result / "generation_metadata.json")
    candidate_meta = load_json(candidate_result / "generation_metadata.json")
    for key, value in {
        "split": "val",
        "n_samples": 500,
        "generation_seed": 424242,
    }.items():
        if source_meta.get(key) != value or candidate_meta.get(key) != value:
            raise ValueError(f"generation protocol mismatch for {key}")
    if source_meta.get("test_used", False) or candidate_meta.get("test_used", False):
        raise ValueError("test split was used")
    if candidate_meta.get("checkpoint_state_source") != "raw":
        raise ValueError("formal generation must use Raw candidate parameters")

    report = {
        "method": "two_expert_local_event_score_temporal_upper_bound",
        "source_variant": source.get("condition_variant"),
        "candidate_variant": candidate.get("condition_variant"),
        "third_expert_used": False,
        "body_member_quota": 2,
        "tail_member_quota": 4,
        "changed_tail_tensor_count": len(changed_tail),
        "changed_temporal_tensor_count": len(changed_temporal),
        "changed_outside_allowed_count": 0,
        "spatial_and_state_frozen_verified": True,
        "event_context_hours": candidate.get("sampler_event_context_hours"),
        "temporal_variogram_lags": candidate.get(
            "sampler_temporal_variogram_lags"
        ),
        "natural_body_anchor_verified": True,
        "scored_epoch_count": len(scored),
        "formal_generation_members": 500,
        "validation_only": True,
        "test_used": False,
    }
    (output / "event_score_upper_bound_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "event_score_upper_bound_audit.md").write_text(
        "# Event-score upper-bound audit\n\n"
        "- Raw body-tail initialization: passed\n"
        "- Exactly two experts: passed\n"
        "- Fixed training quota (2 body + 4 tail): passed\n"
        "- Multi-scale local tail Energy Score: passed\n"
        "- Aggregate 1/3/6 h temporal Variogram: passed\n"
        "- Natural-issuance body epsilon anchor: passed\n"
        "- Temporal body updated; spatial graph/state encoder frozen: passed\n"
        "- Formal validation generation: 500 members, Raw parameters\n"
        "- Test split used: no\n",
        encoding="utf-8",
    )
    print(f"EVENT_SCORE_UPPER_BOUND_AUDIT_PASSED output={output}")


if __name__ == "__main__":
    main()
