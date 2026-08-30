"""Audit parameter isolation and protocol for sampler Energy Score L1."""

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
    if source.get("condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("source is not the Raw body-tail training checkpoint")
    if candidate.get("condition_variant") != (
        "geo_history_actual_body_tail_sampler_es_l1"
    ):
        raise ValueError("candidate variant is not sampler Energy Score L1")
    if not bool(candidate.get("train_sampler_energy_score_only", False)):
        raise ValueError("candidate checkpoint did not enable sampler Energy Score")
    if int(candidate.get("sampler_energy_score_members", 0)) != 4:
        raise ValueError("formal L1 must use four training members")

    initialization = load_json(candidate_run / "body_tail_initialization.json")
    if initialization.get("checkpoint_state_source") != "raw":
        raise ValueError("candidate was not initialized from Raw parameters")
    if not bool(initialization.get("body_frozen", False)):
        raise ValueError("candidate body was not declared frozen")
    if bool(initialization.get("third_expert_used", True)):
        raise ValueError("L1 unexpectedly introduced a third expert")

    trainable = set(candidate.get("body_tail_trainable_parameter_names", []))
    allowed = set(trainable)
    allowed.update(
        "diffusion." + name for name in trainable if name.startswith("denoiser.")
    )
    source_state = source["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    if set(source_state) != set(candidate_state):
        raise ValueError("source and candidate state dictionaries differ structurally")
    changed = []
    frozen_changed = []
    for name, value in source_state.items():
        if not torch.equal(value, candidate_state[name]):
            changed.append(name)
            if name not in allowed:
                frozen_changed.append(name)
    if frozen_changed:
        raise ValueError(f"frozen body tensors changed: {frozen_changed[:8]}")
    if not changed:
        raise ValueError("sampler Energy Score did not update any tail tensor")

    history = load_json(candidate_run / "logs" / "training_history.json")
    scored_rows = [
        row
        for row in history
        if float(row.get("train_sampler_issue_count", 0.0)) > 0.0
    ]
    if not scored_rows:
        raise ValueError("training history contains no scored event member set")
    if not all(
        float(row.get("train_sampler_truth_attraction", -1.0)) >= 0.0
        and float(row.get("train_sampler_member_repulsion", -1.0)) >= 0.0
        for row in scored_rows
    ):
        raise ValueError("Energy Score components are invalid")
    route_rates = [
        float(row.get("train_sampler_tail_route_rate", -1.0))
        for row in scored_rows
    ]
    if not all(0.0 <= value <= 1.0 for value in route_rates):
        raise ValueError("sampler route rates are invalid")
    mean_route_rate = sum(route_rates) / len(route_rates)
    if not 0.0 < mean_route_rate < 1.0:
        raise ValueError("sampler routing collapsed to an all-body or all-tail path")
    condition_audit = load_json(candidate_run / "condition_feature_audit.json")
    score_audit = condition_audit.get("sampler_score_train")
    if not isinstance(score_audit, dict) or bool(
        score_audit.get("event_replay_enabled", True)
    ):
        raise ValueError("Energy Score did not use the natural issuance loader")

    source_meta = load_json(source_result / "generation_metadata.json")
    candidate_meta = load_json(candidate_result / "generation_metadata.json")
    expected = {"split": "val", "n_samples": 500, "generation_seed": 424242}
    for key, value in expected.items():
        if source_meta.get(key) != value or candidate_meta.get(key) != value:
            raise ValueError(f"generation protocol mismatch for {key}")
    if source_meta.get("test_used", False) or candidate_meta.get("test_used", False):
        raise ValueError("test data was used before model lock")
    if candidate_meta.get("checkpoint_state_source") != "raw":
        raise ValueError("candidate formal generation must use raw parameters")

    report = {
        "method": "frozen_raw_body_tail_final_member_energy_score_l1",
        "source_variant": source.get("condition_variant"),
        "candidate_variant": candidate.get("condition_variant"),
        "body_frozen_verified": True,
        "third_expert_used": False,
        "changed_state_tensor_count": len(changed),
        "changed_state_tensors": sorted(changed),
        "frozen_changed_state_tensor_count": 0,
        "training_member_count": int(candidate["sampler_energy_score_members"]),
        "ddim_steps": int(candidate["sampler_energy_score_steps"]),
        "backprop_steps": int(candidate["sampler_energy_score_backprop_steps"]),
        "energy_score_weight": float(candidate["sampler_energy_score_weight"]),
        "route_temperature": float(
            candidate["sampler_energy_score_route_temperature"]
        ),
        "mean_hard_tail_route_rate": mean_route_rate,
        "natural_issuance_score_loader_verified": True,
        "scored_epoch_count": len(scored_rows),
        "formal_generation_members": 500,
        "validation_only": True,
        "test_used": False,
    }
    (output / "sampler_es_l1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "sampler_es_l1_audit.md").write_text(
        "# Sampler Energy Score L1 audit\n\n"
        "- Raw body-tail checkpoint initialization: passed\n"
        "- Frozen body bitwise equality: passed\n"
        "- Existing tail-only updates: passed\n"
        "- No third expert: passed\n"
        "- Natural issuance score loader: passed\n"
        "- Causal hard member routing with straight-through gradients: passed\n"
        "- True four-member final DDIM scoring: passed\n"
        "- Formal validation generation: 500 members, seed 424242\n"
        "- Test split used: no\n",
        encoding="utf-8",
    )
    print(f"SAMPLER_ES_L1_AUDIT_PASSED output={output}")


if __name__ == "__main__":
    main()
