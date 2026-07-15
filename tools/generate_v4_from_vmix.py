#!/usr/bin/env python
"""Generate V4 (residual target, no guidance) from an existing Vmix checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def resolve_from_repo(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a separate V4 run using a trained Vmix checkpoint with guidance=0."
    )
    parser.add_argument("--vmix_run", required=True, help="Existing Vmix run directory or run id")
    parser.add_argument("--outputs_dir", default="outputs_shandong")
    parser.add_argument("--data_path", default="diffusion_npy_normalized")
    parser.add_argument(
        "--config", default="configs/v4_residual_forecast_time_no_guidance_168h.yaml"
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = resolve_from_repo(repo_root, args.outputs_dir)
    data_path = resolve_from_repo(repo_root, args.data_path)
    config_path = resolve_from_repo(repo_root, args.config)

    vmix_arg = Path(args.vmix_run)
    if vmix_arg.is_absolute() or vmix_arg.parent != Path("."):
        vmix_run = resolve_from_repo(repo_root, args.vmix_run)
    else:
        vmix_run = outputs_dir / args.vmix_run

    source_checkpoint = vmix_run / "checkpoints" / "model_best.pt"
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Vmix checkpoint not found: {source_checkpoint}")
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config["experiment"]["name"]
    run_id = f"{timestamp}_{experiment_name}"
    run_dir = outputs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4 run: {run_dir}")

    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "samples").mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "logs").mkdir()
    shutil.copy2(source_checkpoint, run_dir / "checkpoints" / "model_best.pt")

    config["experiment"].update(
        {
            "config_path": str(config_path.relative_to(repo_root)),
            "output_dir": str(run_dir.relative_to(repo_root)),
            "run_id": run_id,
            "timestamp": timestamp,
            "source_vmix_run": str(vmix_run.relative_to(repo_root)),
        }
    )
    config["data"]["data_path"] = str(data_path.relative_to(repo_root))
    config["evaluation"]["n_samples"] = args.n_samples
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    provenance = {
        "run_id": run_id,
        "source_vmix_run": str(vmix_run.relative_to(repo_root)),
        "source_checkpoint": str(source_checkpoint.relative_to(repo_root)),
        "guidance_scale": 0.0,
        "checkpoint_reused_without_retraining": True,
        "reason": "Guidance is applied only during reverse-diffusion sampling.",
    }
    with (run_dir / "v4_provenance.json").open("w", encoding="utf-8") as file:
        json.dump(provenance, file, ensure_ascii=False, indent=2)

    command = [
        sys.executable,
        "generate.py",
        "--save_path",
        str(outputs_dir.relative_to(repo_root)),
        "--exp_name",
        run_id,
        "--data_path",
        str(data_path.relative_to(repo_root)),
        "--n_samples",
        str(args.n_samples),
        "--batch_size",
        str(args.batch_size),
        "--guidance_scale",
        "0",
    ]
    print(f"Prepared V4 run: {run_dir}")
    print(f"Reusing checkpoint: {source_checkpoint}")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
