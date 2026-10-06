"""Read-only integrity gate for every immutable Stage-1A artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from station_lightweight_tail import validate_config


MEMBER_ARRAYS = (
    "actual_scenarios_normalized.npy", "actual_scenarios_raw_normalized.npy",
    "generated_residual_normalized.npy", "generated_residual_standardized.npy",
    "generated_stochastic_residual_standardized.npy",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--control-post", type=Path, required=True)
    parser.add_argument("--raw-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-mixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks = {}
    hashes = {}
    errors = {}
    for name in MEMBER_ARRAYS:
        raw_path = args.raw_result / name
        control_path = args.control_result / name
        raw = None
        control = None
        try:
            raw = np.load(raw_path, mmap_mode="r")
            checks[f"raw_shape_{name}"] = list(raw.shape) == [23, 500, 168, 24]
            hashes[f"raw_{name}"] = sha(raw_path)
        except Exception as exc:
            checks[f"raw_shape_{name}"] = False
            errors[f"raw/{name}"] = repr(exc)
        try:
            control = np.load(control_path, mmap_mode="r")
            checks[f"control_shape_{name}"] = list(control.shape) == [23, 500, 168, 24]
            hashes[f"control_{name}"] = sha(control_path)
        except Exception as exc:
            checks[f"control_shape_{name}"] = False
            errors[f"control/{name}"] = repr(exc)
        checks[f"body400_exact_{name}"] = bool(
            raw is not None and control is not None
            and checks[f"raw_shape_{name}"] and checks[f"control_shape_{name}"]
            and all(np.array_equal(raw[i, :400], control[i, :400]) for i in range(23))
        )

    try:
        metrics = json.loads((args.control_result / "metrics.json").read_text(encoding="utf-8"))["run"]
        checks.update({
            "control_validation_split": metrics.get("split") == "val",
            "control_generation_seed": int(metrics.get("generation_seed", -1)) == 424242,
            "control_members": int(metrics.get("n_samples", -1)) == 500,
            "control_test_locked": not bool(metrics.get("test_used", True)),
            "control_body400_tail100": (int(metrics.get("body_members", -1)) == 400
                                        and int(metrics.get("tail_members", -1)) == 100),
        })
    except Exception as exc:
        for name in ("control_validation_split", "control_generation_seed", "control_members",
                     "control_test_locked", "control_body400_tail100"):
            checks[name] = False
        errors["control_metrics"] = repr(exc)
    try:
        run_config = yaml.safe_load((args.control_run / "config_used.yaml").read_text(encoding="utf-8"))
        validate_config(run_config, resolved=True)
        checks["control_exact_alpha_1_recipe"] = "auxiliary_strength_alpha" not in run_config["model"]
    except Exception as exc:
        checks["control_exact_alpha_1_recipe"] = False
        errors["control_config"] = repr(exc)
    event_files = (
        "continuous_event_per_event.csv", "continuous_event_member_matches.csv",
        "continuous_event_three_standard_summary.csv",
    )
    for name in event_files:
        checks[f"control_event_{name}"] = (args.control_post / "continuous_event_evaluation" / name).is_file()
    try:
        checks["raw_checkpoint_loadable"] = "model_state_dict" in torch.load(
            args.raw_checkpoint, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        checks["raw_checkpoint_loadable"] = False
        errors["raw_checkpoint"] = repr(exc)
    checks["v2_reference_metrics"] = (args.v2_mixture / "metrics.json").is_file()
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {"status": "PASS" if not failed else "FAIL", "checks": checks,
              "failed": failed, "errors": errors, "hashes": hashes, "test_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        raise RuntimeError(f"immutable Stage-1A inputs failed: {failed}")
    print(f"AUXILIARY_ALPHA_INPUT_AUDIT_PASS output={args.output}")


if __name__ == "__main__":
    main()
