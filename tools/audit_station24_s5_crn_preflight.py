#!/usr/bin/env python3
"""Fail-fast preflight for the zero-training Original-S5 CRN experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import keyed_ancestral_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--tail-run", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("diffusion_input_station"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict[str, object]], name: str, passed: bool, detail: object) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is mandatory for the formal CRN preflight")

    checks: list[dict[str, object]] = []
    inputs = {}
    configs = {}
    for label, run in (("raw", args.raw_run), ("tail", args.tail_run)):
        config_path = run / "config_used.yaml"
        checkpoint_path = run / "checkpoints" / "model_best.pt"
        add(checks, f"{label}_config_exists", config_path.is_file(), str(config_path))
        add(checks, f"{label}_checkpoint_exists", checkpoint_path.is_file(), str(checkpoint_path))
        if not config_path.is_file() or not checkpoint_path.is_file():
            continue
        configs[label] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        inputs[label] = {
            "config": str(config_path), "config_sha256": sha256(config_path),
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        }
    if len(configs) != 2:
        raise SystemExit("missing frozen CRN inputs")
    raw_model, tail_model = configs["raw"]["model"], configs["tail"]["model"]
    frozen_keys = (
        "architecture", "spatial_mode", "spatial_mix_levels",
        "parallel_spatial_fusion_levels", "parallel_spatial_adjacency_mode",
        "station_count", "sequence_length", "num_steps", "beta_start", "beta_end",
        "schedule", "reverse_variance_type", "use_recent_error", "recent_error_hours",
        "use_state_encoder", "state_feature_names", "state_ramp_lags",
    )
    differences = {
        key: {"raw": raw_model.get(key), "tail": tail_model.get(key)}
        for key in frozen_keys if raw_model.get(key) != tail_model.get(key)
    }
    add(checks, "shared_sampling_and_condition_contract", not differences, differences)
    add(checks, "formal_ancestral_steps_500", raw_model.get("num_steps") == tail_model.get("num_steps") == 500,
        {"raw": raw_model.get("num_steps"), "tail": tail_model.get("num_steps")})
    add(checks, "formal_linear_schedule", raw_model.get("schedule") == tail_model.get("schedule") == "linear",
        {"raw": raw_model.get("schedule"), "tail": tail_model.get("schedule")})
    add(checks, "validation_data_exists", args.data_path.is_dir(), str(args.data_path))

    issue = torch.tensor([0, 7], device=device, dtype=torch.long)
    common = dict(
        base_seed=int(args.seed), issue_indices=issue, member_block_start=400,
        members=3, stations=24, length=168, device=device, dtype=torch.float32,
    )
    digest_a = hashlib.sha256()
    digest_b = hashlib.sha256()
    stages_equal = True
    for stage in range(500, -1, -1):
        first = keyed_ancestral_noise(stage=stage, **common)
        _ = torch.randn((17,), device=device)
        second = keyed_ancestral_noise(stage=stage, **common)
        stages_equal &= bool(torch.equal(first, second))
        digest_a.update(first.cpu().numpy().tobytes())
        digest_b.update(second.cpu().numpy().tobytes())
    add(checks, "all_501_noise_stages_bitwise_replay", stages_equal,
        {"first_sha256": digest_a.hexdigest(), "replay_sha256": digest_b.hexdigest()})
    add(checks, "raw400_tail400_noise_plan_identity", digest_a.digest() == digest_b.digest(),
        "same issue/member-block/stage keys")

    changed_stage = keyed_ancestral_noise(stage=499, **common)
    changed_block = keyed_ancestral_noise(stage=500, **{**common, "member_block_start": 300})
    reference = keyed_ancestral_noise(stage=500, **common)
    add(checks, "different_stage_changes_noise", not torch.equal(reference, changed_stage), "500 vs 499")
    add(checks, "different_member_block_changes_noise", not torch.equal(reference, changed_block), "400 vs 300")

    samples = []
    stats_common = dict(
        base_seed=int(args.seed), issue_indices=torch.arange(8, device=device),
        member_block_start=0, members=32, stations=24, length=168,
        device=device, dtype=torch.float32,
    )
    for stage in (500, 499, 375, 250, 125, 1):
        samples.append(keyed_ancestral_noise(stage=stage, **stats_common).cpu())
    values = torch.cat([value.reshape(-1) for value in samples])
    mean, std = float(values.mean()), float(values.std())
    add(checks, "noise_marginal_mean", abs(mean) <= 0.02, mean)
    add(checks, "noise_marginal_std", abs(std - 1.0) <= 0.02, std)
    checks.append({
        "check": "target_device_gate",
        "status": "PASS" if device.type == "cuda" else "NOT RUN",
        "detail": {"device": str(device), "formal_cuda_pass": device.type == "cuda"},
    })
    add(checks, "training_not_applicable", True, "no optimizer/model mutation/checkpoint write")
    add(checks, "test_locked", True, "preflight has no split selector and reads no test arrays")
    for label, run in (("raw", args.raw_run), ("tail", args.tail_run)):
        checkpoint_path = run / "checkpoints" / "model_best.pt"
        config_path = run / "config_used.yaml"
        add(
            checks,
            f"{label}_checkpoint_hash_unchanged",
            sha256(checkpoint_path) == inputs[label]["checkpoint_sha256"],
            inputs[label]["checkpoint_sha256"],
        )
        add(
            checks,
            f"{label}_config_hash_unchanged",
            sha256(config_path) == inputs[label]["config_sha256"],
            inputs[label]["config_sha256"],
        )

    failed = [row["check"] for row in checks if row["status"] == "FAIL"]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "device": str(device),
        "formal_cuda_pass": device.type == "cuda" and not failed,
        "launch_eligible": device.type == "cuda" and not failed,
        "training": "NOT APPLICABLE",
        "generation": "NOT RUN",
        "seed": int(args.seed),
        "inputs": inputs,
        "checks": checks,
        "failed": failed,
    }
    (args.output_dir / "preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Station-24 Original S5 CRN preflight", "",
        f"Device: `{device}`; formal CUDA pass: `{report['formal_cuda_pass']}`; "
        f"launch eligible: `{report['launch_eligible']}`", "",
        "| Check | Status | Detail |", "|---|---|---|",
    ]
    for row in checks:
        detail = json.dumps(row["detail"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {row['check']} | {row['status']} | {detail} |")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("CRN preflight failed: " + ", ".join(failed))
    print(f"S5_CRN_PREFLIGHT_COMPLETE output={args.output_dir} device={device}")


if __name__ == "__main__":
    main()
