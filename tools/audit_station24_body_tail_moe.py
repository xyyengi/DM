"""Audit parameter isolation and causal member routing for body-tail MoE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-run", required=True)
    parser.add_argument("--body-result", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_checkpoint(run: Path) -> dict[str, object]:
    path = run / "checkpoints" / "model_best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    body_run = Path(args.body_run)
    body_result = Path(args.body_result)
    candidate_run = Path(args.candidate_run)
    candidate_result = Path(args.candidate_result)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    body = load_checkpoint(body_run)
    candidate = load_checkpoint(candidate_run)
    body_meta = json.loads(
        (body_result / "generation_metadata.json").read_text(encoding="utf-8")
    )
    candidate_meta = json.loads(
        (candidate_result / "generation_metadata.json").read_text(encoding="utf-8")
    )
    if body.get("condition_variant") != "geo_history_actual_dual":
        raise ValueError("body checkpoint is not the historical-spatial baseline")
    if candidate.get("condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("candidate checkpoint variant mismatch")
    if not bool(candidate.get("use_body_tail_experts", False)):
        raise ValueError("candidate checkpoint did not enable body-tail experts")
    if candidate.get("event_replay") is None:
        raise ValueError("candidate lacks train-only independent event replay")
    if bool(candidate_meta.get("event_replay_applied_during_generation", True)):
        raise ValueError("event replay leaked into generation")
    if bool(candidate_meta.get("test_used", True)):
        raise ValueError("test split was used")
    for metadata in (body_meta, candidate_meta):
        if metadata.get("split") != "val" or int(metadata.get("n_samples", 0)) != 500:
            raise ValueError("formal audit requires validation split with 500 members")

    tail_names = set(candidate.get("body_tail_trainable_parameter_names", []))
    if not tail_names or not all(name.startswith("denoiser.tail_") for name in tail_names):
        raise ValueError("candidate trainable parameter isolation is invalid")
    body_state = body.get("ema_model_state_dict", body["model_state_dict"])
    candidate_state = candidate.get(
        "ema_model_state_dict", candidate["model_state_dict"]
    )
    changed_body_keys: list[str] = []
    missing_body_keys: list[str] = []
    for key, value in body_state.items():
        if key not in candidate_state:
            missing_body_keys.append(key)
        else:
            observed = candidate_state[key]
            equal = (
                torch.allclose(value, observed, rtol=0.0, atol=1e-7)
                if value.is_floating_point()
                else torch.equal(value, observed)
            )
            if not equal:
                changed_body_keys.append(key)
    if missing_body_keys or changed_body_keys:
        raise ValueError(
            "frozen body differs from source checkpoint: "
            f"missing={missing_body_keys[:5]} changed={changed_body_keys[:5]}"
        )

    probability = np.load(candidate_result / "tail_expert_probability.npy")
    route = np.load(candidate_result / "tail_expert_route.npy")
    attention = np.load(candidate_result / "tail_condition_attention.npy")
    if probability.shape != (23,) or route.shape != (23, 500):
        raise ValueError(
            f"unexpected routing arrays probability={probability.shape} route={route.shape}"
        )
    if np.any(~np.isfinite(probability)) or np.any((probability <= 0) | (probability >= 1)):
        raise ValueError("tail probabilities must be finite and strictly within (0,1)")
    if not np.all(np.isin(route, [0, 1])):
        raise ValueError("tail member routes must be binary")
    if int(route.sum()) == 0 or int((route == 0).sum()) == 0:
        raise ValueError("formal body-tail audit requires both routed expert groups")
    if attention.shape != (23, 6) or not np.allclose(
        attention.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("tail condition attention must be [23,6] and sum to one")

    audit = {
        "status": "passed",
        "body_condition_variant": body["condition_variant"],
        "candidate_condition_variant": candidate["condition_variant"],
        "body_parameter_keys_checked": len(body_state),
        "changed_body_parameter_keys": changed_body_keys,
        "tail_trainable_parameter_count": len(tail_names),
        "tail_probability_mean": float(probability.mean()),
        "tail_probability_min": float(probability.min()),
        "tail_probability_max": float(probability.max()),
        "tail_member_fraction": float(route.mean()),
        "tail_member_count": int(route.sum()),
        "tail_condition_attention_mean": [
            float(value) for value in attention.mean(axis=0)
        ],
        "validation_issue_count": int(route.shape[0]),
        "members_per_issue": int(route.shape[1]),
        "future_actual_used_for_routing": False,
        "test_used": False,
    }
    (output / "body_tail_moe_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "body_tail_moe_audit.md").write_text(
        "# Body-tail MoE audit\n\n"
        "- Status: passed\n"
        f"- Frozen body keys checked: {audit['body_parameter_keys_checked']}\n"
        f"- Trainable tail parameter tensors: {audit['tail_trainable_parameter_count']}\n"
        f"- Mean causal tail probability: {audit['tail_probability_mean']:.4f}\n"
        f"- Realized tail-member fraction: {audit['tail_member_fraction']:.4f}\n"
        "- Future actual used for routing: false\n"
        "- Test split used: false\n",
        encoding="utf-8",
    )
    print(f"BODY_TAIL_MOE_AUDIT_PASSED output={output}")


if __name__ == "__main__":
    main()
