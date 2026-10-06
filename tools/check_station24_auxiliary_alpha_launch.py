"""Promote a passed target-CUDA preflight into an auditable formal launch gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--config-audit", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, choices=(0.65, 1.35))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    config = json.loads(args.config_audit.read_text(encoding="utf-8"))
    nonpass = {name: row.get("status") for name, row in preflight.get("checks", {}).items()
               if row.get("status") != "PASS"}
    reasons = []
    if preflight.get("preflight_execution") != "PASS":
        reasons.append("preflight_execution is not PASS")
    if preflight.get("device") != "cuda":
        reasons.append("target preflight device is not cuda")
    if nonpass:
        reasons.append(f"non-PASS mandatory checks: {nonpass}")
    if config.get("status") != "PASS" or float(config.get("alpha", -1)) != args.alpha:
        reasons.append("frozen config materialization audit mismatch")
    report = {
        "status": "PASS" if not reasons else "FAIL",
        "launch_eligible": not reasons,
        "alpha": args.alpha,
        "target_device": preflight.get("device"),
        "preflight_execution": preflight.get("preflight_execution"),
        "mandatory_check_count": len(preflight.get("checks", {})),
        "nonpass_checks": nonpass,
        "unresolved_issues": reasons,
        "formal_user_authorization": "Stage 1A explicitly requested in the parent formal launcher",
        "test_locked": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if reasons:
        raise RuntimeError("; ".join(reasons))
    print(f"AUXILIARY_ALPHA_LAUNCH_GATE_PASS alpha={args.alpha}")


if __name__ == "__main__":
    main()
