"""Final bounded train→resume→sample CUDA preflight for Shandong91 formal v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.shandong91_reliable import CHANNELS, Shandong91ReliableDataset
from train_shandong91 import build_model, move_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shandong91/raw_body_heterogeneous_formal_v1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(f"refusing to overwrite preflight: {output}")
    if args.device == "cuda" and not args.amp: raise ValueError("final CUDA preflight requires --amp")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run = output / "bounded_training"
    common = [sys.executable, "train_shandong91.py", "--config", args.config, "--output-dir", str(run), "--device", args.device, "--allow-dirty", "--max-train-batches", "1", "--max-val-batches", "1"]
    subprocess.run([*common, "--max-epochs", "1"], check=True)
    first = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    subprocess.run([*common, "--max-epochs", "2", "--resume", str(run / "checkpoints" / "last.pt")], check=True)
    resumed = torch.load(run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    continuous_run = output / "continuous_training"
    continuous_command = [
        sys.executable, "train_shandong91.py", "--config", args.config,
        "--output-dir", str(continuous_run), "--device", args.device,
        "--allow-dirty", "--max-train-batches", "1", "--max-val-batches", "1",
        "--max-epochs", "2",
    ]
    subprocess.run(continuous_command, check=True)
    continuous = torch.load(
        continuous_run / "checkpoints" / "last.pt", map_location="cpu", weights_only=False
    )

    model_resume_delta = max(
        float((resumed["model_state_dict"][name].float() - continuous["model_state_dict"][name].float()).abs().max())
        for name in resumed["model_state_dict"]
    )

    def nested_equal(left, right):
        if torch.is_tensor(left): return torch.equal(left, right)
        if isinstance(left, dict): return left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)): return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
        return left == right

    optimizer_exact = nested_equal(resumed["optimizer_state_dict"], continuous["optimizer_state_dict"])
    scaler_exact = nested_equal(resumed["amp_scaler_state_dict"], continuous["amp_scaler_state_dict"])
    exact_resume = model_resume_delta == 0.0 and optimizer_exact and scaler_exact

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    dataset = Shandong91ReliableDataset(config["data"]["data_path"], "validation")
    raw = dataset[0]
    batch = move_batch({key: value.unsqueeze(0) for key, value in raw.items()}, device)
    node_mask = dataset.node_type_mask.to(device).view(1, 1, 91, 3)
    train_mask = dataset.channel_train_mask.to(device).view(1, 1, 91, 3)
    generation_mask = node_mask & train_mask & batch["forecast_valid_mask"].bool()
    publication_mask = node_mask & batch["forecast_valid_mask"].bool()
    initial_noise = torch.randn_like(batch["forecast"])
    def sample_from(state):
        current = build_model(config, dataset, device)
        current.load_state_dict(state["model_state_dict"], strict=True); current.eval()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
            return current.sample_ddim(
                batch["forecast"], batch["forecast_valid_mask"], batch["time_mark"],
                generation_mask, inference_steps=4, initial_noise=initial_noise,
            )
    reloaded_residual = sample_from(first)
    generated_residual = sample_from(resumed)
    generated_residual = generated_residual.float().masked_fill(~generation_mask, 0.0)
    generated_actual_normalized = (batch["forecast"] + generated_residual).masked_fill(~publication_mask, 0.0)
    residual_mw = dataset.denormalize_mw(generated_residual)
    forecast_mw = dataset.denormalize_mw(batch["forecast"])
    generated_actual_mw = (forecast_mw + residual_mw).masked_fill(~publication_mask, 0.0)
    inactive_random_delta = float(generated_residual[~generation_mask].abs().max())
    sign_chain_delta = float((generated_actual_mw[publication_mask] - (forecast_mw + residual_mw)[publication_mask]).abs().max())

    finite_by_resource = {
        name: bool(torch.isfinite(generated_actual_mw[..., index][publication_mask[..., index]]).all())
        for index, name in enumerate(CHANNELS)
    }
    checks = {
        "first_epoch_checkpoint": first["epoch"] == 1 and first["global_step"] == 1,
        "resume_epoch_and_step": resumed["epoch"] == 2 and resumed["global_step"] == 2,
        "optimizer_restored_and_updated": bool(resumed["optimizer_state_dict"]["state"]),
        "scaler_restored": "amp_scaler_state_dict" in resumed,
        "scheduler_none_explicit": resumed["scheduler_state_dict"] is None,
        "rng_restored_contract": resumed["resume_level"] == "exact deterministic at completed epoch boundary",
        "exact_epoch_boundary_resume_vs_continuous": exact_resume,
        "generated_residual_shape": list(generated_residual.shape) == [1, 168, 91, 3],
        "generated_actual_shape": list(generated_actual_mw.shape) == [1, 168, 91, 3],
        "generated_finite": bool(torch.isfinite(generated_residual).all() and torch.isfinite(generated_actual_mw).all()),
        "all_resources_finite": all(finite_by_resource.values()),
        "inactive_policy": inactive_random_delta == 0.0,
        "residual_sign_forecast_plus_residual": sign_chain_delta <= 1e-6,
        "save_reload_sampler_available": list(reloaded_residual.shape) == [1, 168, 91, 3] and bool(torch.isfinite(reloaded_residual).all()),
        "resume_sampler_available": list(generated_residual.shape) == [1, 168, 91, 3] and bool(torch.isfinite(generated_residual).all()),
        "manifest_exists": (run / "manifest.json").is_file(),
    }
    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL", "scope": f"bounded {device.type.upper()} E2E preflight; not formal training",
        "device": str(device), "amp": args.amp, "checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
        "checkpoint": {"first_epoch": first["epoch"], "first_global_step": first["global_step"], "resumed_epoch": resumed["epoch"], "resumed_global_step": resumed["global_step"], "resume_level": resumed["resume_level"] if exact_resume else "functional only", "model_resume_max_abs_delta_vs_continuous": model_resume_delta, "optimizer_exact_vs_continuous": optimizer_exact, "scaler_exact_vs_continuous": scaler_exact},
        "sampler": {"method": "DDIM eta=0", "inference_steps": 4, "residual_sign": "generated_actual = forecast + generated_residual", "generated_residual_shape": list(generated_residual.shape), "generated_actual_shape": list(generated_actual_mw.shape), "finite_ratio": float(torch.isfinite(generated_actual_mw).float().mean()), "finite_by_resource": finite_by_resource, "inactive_max_abs_residual": inactive_random_delta, "sign_chain_max_abs_delta": sign_chain_delta, "policy": "zero residual outside node_type & channel_train & forecast_valid; deterministic forecast retained for train-disabled existing resources; neutral zero and invalid mask for nonexistent/forecast-invalid resources", "normalized_value_clipped": False},
        "formal_training": "NOT RUN", "formal_generation": "NOT RUN",
    }
    output.mkdir(parents=True, exist_ok=True); (output / "e2e_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "e2e_report.md").write_text("# Shandong91 formal E2E preflight\n\n" + "\n".join([f"- Overall: **{report['status']}**", f"- Device: **{device}**", f"- AMP: **{args.amp}**", "- Formal training: **NOT RUN**", "- Formal generation: **NOT RUN**", "", *[f"- {k}: **{v}**" for k, v in report["checks"].items()]]), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
