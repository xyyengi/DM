"""Post-training alpha-run integrity audit; performs no optimization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml

from station_lightweight_tail import FROZEN_AUXILIARY_ALPHAS, validate_config


PREFIX = "denoiser.joint_multiresidual_tail."


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state, names) -> str:
    h = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, choices=(0.65, 1.35))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.run_dir / "checkpoints/model_best.pt"
    config_path = args.run_dir / "config_used.yaml"
    summary_path = args.run_dir / "training_summary.json"
    for path in (checkpoint, config_path, summary_path, args.raw_checkpoint, args.preflight):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config, resolved=True)
    if float(config["model"].get("auxiliary_strength_alpha")) != args.alpha:
        raise ValueError("saved config alpha mismatch")
    weights = FROZEN_AUXILIARY_ALPHAS[args.alpha]
    actual = tuple(float(config["model"][key]) for key in (
        "event_balanced_ramp_loss_weight",
        "event_balanced_shape_loss_weight",
        "event_balanced_slow_loss_weight",
    ))
    if actual != weights:
        raise ValueError(f"saved loss weights mismatch: {actual} != {weights}")

    raw = torch.load(args.raw_checkpoint, map_location="cpu", weights_only=False)
    trained = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_state = raw["model_state_dict"]
    trained_state = trained["model_state_dict"]
    missing = sorted(set(raw_state) - set(trained_state))
    changed = [name for name in raw_state if not torch.equal(raw_state[name], trained_state[name])]
    if missing or changed:
        raise ValueError(f"Raw state changed or missing: missing={missing[:3]} changed={changed[:3]}")
    tail_names = sorted(name for name in trained_state if name.startswith(PREFIX))
    if not tail_names:
        raise ValueError("trained checkpoint contains no lightweight Tail state")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    trainable_names = sorted(summary.get("joint_multiresidual_trainable_parameter_names", []))
    if not trainable_names or not set(trainable_names).issubset(set(tail_names)):
        raise ValueError("training summary trainable set is not contained in Tail state")
    parameter_count = sum(int(trained_state[name].numel()) for name in trainable_names)
    if parameter_count != 20588:
        raise ValueError(f"unexpected trainable count: {parameter_count}")

    report = {
        "status": "PASS",
        "alpha": args.alpha,
        "actual_lambda_ramp": weights[0],
        "actual_lambda_shape": weights[1],
        "actual_lambda_slow": weights[2],
        "sampling_ratio": float(config["model"]["independent_tail_event_sampling_fraction"]),
        "training_seed": int(config["train"]["seed"]),
        "validation_seed": int(config["train"]["validation_seed"]),
        "generation_seed": int(config["evaluation"]["generation_seed"]),
        "tail_initialization_source": str(args.raw_checkpoint),
        "raw_checkpoint_sha256": file_sha(args.raw_checkpoint),
        "raw_state_sha256_before": state_sha(raw_state, raw_state),
        "raw_state_sha256_after": state_sha(trained_state, raw_state),
        "raw_parameters_and_buffers_unchanged": True,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": parameter_count,
        "frozen_parameter_count": int(preflight.get("frozen_parameter_count", -1)),
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_objective": float(summary["best_validation_objective"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha(checkpoint),
        "test_used": bool(summary.get("test_used", False)),
    }
    if report["test_used"]:
        raise ValueError("test split was used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ALPHA_RUN_INTEGRITY_PASS alpha={args.alpha} output={args.output}")


if __name__ == "__main__":
    main()

