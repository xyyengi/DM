#!/usr/bin/env python3
"""Post-training isolation and non-collapse audit for JSTD-Tail V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    candidate_path = Path(args.candidate_run) / "checkpoints" / "model_best.pt"
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=False)
    if candidate.get("condition_variant") != "geo_history_actual_jstd_tail_v1":
        raise ValueError("candidate checkpoint is not JSTD-Tail V1")
    if not bool(candidate.get("use_jstd_tail", False)):
        raise ValueError("candidate checkpoint did not record use_jstd_tail")
    source_state = source["model_state_dict"]
    candidate_state = candidate["model_state_dict"]
    statistical_buffers = (
        "_forecast_condition_sample_count",
        "_forecast_condition_drop_count",
    )
    changed_raw = []
    for name, value in source_state.items():
        if name not in candidate_state:
            raise ValueError(f"candidate lost Raw state key {name}")
        if name.endswith(statistical_buffers):
            continue
        if not torch.equal(value, candidate_state[name]):
            changed_raw.append(name)
    if changed_raw:
        raise ValueError(f"Raw body/tail state changed despite freeze: {changed_raw[:20]}")

    trainable = candidate.get("jstd_trainable_parameter_names", [])
    if not trainable or any(not name.startswith("denoiser.jstd_tail.") for name in trainable):
        raise ValueError("checkpoint trainable manifest is not JSTD-isolated")
    learned_keys = [
        name
        for name in candidate_state
        if name.startswith("denoiser.jstd_tail.")
        and (name.endswith("weight") or name.endswith("bias"))
    ]
    learned_norm = float(
        sum(candidate_state[name].float().square().sum() for name in learned_keys).sqrt()
    )
    correction_keys = [
        name
        for name in learned_keys
        if any(token in name for token in ("slow_raw", "fast_raw", "slow_modes", "fast_modes"))
    ]
    correction_norm = float(
        sum(candidate_state[name].float().square().sum() for name in correction_keys).sqrt()
    )
    if correction_norm <= 1e-6:
        raise ValueError("JSTD correction heads remained at zero")

    result = Path(args.candidate_result)
    metadata = json.loads((result / "generation_metadata.json").read_text(encoding="utf-8"))
    route = np.load(result / "tail_expert_route.npy")
    route_fraction = float(route.mean())
    if not 0.005 < route_fraction < 0.80:
        raise ValueError(f"JSTD routing collapsed: fraction={route_fraction}")
    audit = {
        "method": "jstd_tail_v1_post_training_isolation_audit",
        "raw_state_changed_count": len(changed_raw),
        "raw_body_and_condition_modulation_frozen": True,
        "jstd_trainable_parameter_count": len(trainable),
        "jstd_learned_state_norm": learned_norm,
        "jstd_correction_head_norm": correction_norm,
        "tail_member_fraction": route_fraction,
        "generated_members": int(metadata["n_samples"]),
        "checkpoint_state": metadata["checkpoint_state_source"],
        "passed": True,
    }
    (output / "jstd_result_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSTD_RESULT_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
