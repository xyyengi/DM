#!/usr/bin/env python3
"""Validate and query the frozen single-seed Station-24 CRN stage-2 config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--field")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nested(payload: dict, dotted: str):
    value = payload
    for key in dotted.split("."):
        value = value[key]
    return value


def validate(config: dict) -> None:
    experiment = config["experiment"]
    paths = config["paths"]
    runtime = config["runtime"]
    expected = {
        "split": "val",
        "body_members": 400,
        "background_members": 100,
        "tail_members": 100,
    }
    for key, value in expected.items():
        if experiment.get(key) != value:
            raise ValueError(f"{key} must remain {value!r}")
    seeds = experiment.get("permutation_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct permutation seeds are required")
    if int(experiment["generation_seed"]) in {int(seed) for seed in seeds}:
        raise ValueError("generation seed and permutation seeds must be distinct")
    if not bool(runtime.get("cuda_required")) or not bool(runtime.get("gate_amp")):
        raise ValueError("CUDA and AMP must remain mandatory for the server gate")
    if bool(runtime.get("formal_generation_amp")):
        raise ValueError(
            "formal generation AMP is frozen false to preserve historical inference semantics"
        )
    if int(runtime["issue_batch_size"]) <= 0 or int(runtime["member_chunk_size"]) <= 0:
        raise ValueError("batch/chunk sizes must be positive")
    if int(runtime["canonical_member_block_size"]) != 100:
        raise ValueError("the verified canonical CRN block size is 100")
    if runtime.get("checkpoint_state") != "raw":
        raise ValueError("formal Full V2 comparison uses checkpoint_state=raw")
    for key in ("data_path", "raw_run", "tail_run"):
        if not str(paths.get(key, "")).strip():
            raise ValueError(f"missing paths.{key}")


def validate_inputs(config: dict) -> list[str]:
    paths = config["paths"]
    required = [
        Path(paths["data_path"]) / "val_forecast.npy",
        Path(paths["data_path"]) / "val_actual.npy",
        Path(paths["data_path"]) / "station_order.csv",
        Path(paths["raw_run"]) / "config_used.yaml",
        Path(paths["raw_run"]) / "checkpoints" / "model_best.pt",
        Path(paths["tail_run"]) / "config_used.yaml",
        Path(paths["tail_run"]) / "checkpoints" / "model_best.pt",
        Path(paths["tail_run"]) / "jstd_event_targets.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return missing


def validate_gate(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("launch_eligible") is not True:
        raise ValueError("GPU gate is not launch-eligible")
    if payload.get("device") != "cuda" or payload.get("amp") is not True:
        raise ValueError("GPU gate did not run CUDA+AMP")
    checks = {row["check"]: row["status"] for row in payload.get("checks", [])}
    if checks.get("target_server_CUDA_AMP") != "PASS":
        raise ValueError("target_server_CUDA_AMP is not PASS")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate(config)
    if args.field:
        value = nested(config, args.field)
        if isinstance(value, (list, dict)):
            print(json.dumps(value, separators=(",", ":")))
        elif isinstance(value, bool):
            print("true" if value else "false")
        else:
            print(value)
        return
    missing = validate_inputs(config)
    if missing:
        raise FileNotFoundError("missing immutable inputs: " + ", ".join(missing))
    if args.output_root is not None:
        root = args.output_root.resolve()
        if root == Path.cwd().resolve():
            raise ValueError("output root cannot be the repository root")
        if root.exists() and not (root / "EXPERIMENT_MANIFEST.json").is_file():
            raise ValueError("existing output root lacks the CRN manifest")
    if args.gate_report is not None:
        validate_gate(args.gate_report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "config": str(args.config),
                "inputs": "PASS",
                "output_root": str(args.output_root) if args.output_root else None,
                "gate": str(args.gate_report) if args.gate_report else "NOT CHECKED",
                "formal_generation": "NOT RUN" if args.dry_run else "NOT REQUESTED",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
