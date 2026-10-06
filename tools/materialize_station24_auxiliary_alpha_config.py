"""Materialize one frozen Stage-1A config and prove alpha-only parity."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from station_lightweight_tail import FROZEN_AUXILIARY_ALPHAS, expected_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/station24_lightweight_joint_tail_v2_fair_168h.yaml"
ALLOWED = {
    "experiment.name",
    "experiment.variant",
    "experiment.description",
    "model.auxiliary_strength_alpha",
    "model.event_balanced_ramp_loss_weight",
    "model.event_balanced_shape_loss_weight",
    "model.event_balanced_slow_loss_weight",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value, prefix=""):
    rows = {}
    if isinstance(value, dict):
        for key, item in value.items():
            rows.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        rows[prefix] = value
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, choices=sorted(FROZEN_AUXILIARY_ALPHAS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.alpha == 1.0:
        raise ValueError("alpha=1.00 must reuse the existing control and may not be retrained")

    control = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    candidate = expected_config(auxiliary_alpha=args.alpha)
    validate_config(candidate)
    left, right = flatten(control), flatten(candidate)
    changed = sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )
    unexpected = sorted(set(changed) - ALLOWED)
    if unexpected:
        raise ValueError(f"non-alpha config changes: {unexpected}")
    weights = FROZEN_AUXILIARY_ALPHAS[args.alpha]
    expected_changed = {
        "experiment.name", "experiment.variant", "experiment.description",
        "model.auxiliary_strength_alpha",
        "model.event_balanced_ramp_loss_weight",
        "model.event_balanced_shape_loss_weight",
        "model.event_balanced_slow_loss_weight",
    }
    if set(changed) != expected_changed:
        raise ValueError(f"missing or extra declared changes: {changed}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    audit = {
        "status": "PASS",
        "alpha": args.alpha,
        "weights": {"ramp": weights[0], "shape": weights[1], "slow": weights[2]},
        "control_config": str(CONTROL.relative_to(ROOT)),
        "control_sha256": sha(CONTROL),
        "candidate_config": str(args.output),
        "candidate_sha256": sha(args.output),
        "changed_keys": changed,
        "unexpected_changes": unexpected,
        "stage1b_authorized": False,
        "test_locked": True,
    }
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"ALPHA_CONFIG_MATERIALIZATION_PASS alpha={args.alpha} output={args.output}")


if __name__ == "__main__":
    main()

