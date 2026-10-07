"""Bounded Gate 1 overfit probe for the Shandong91 heterogeneous Raw Body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.shandong91_reliable import CHANNELS, Shandong91ReliableDataset
from src.models.shandong91_conditioned_diffusion import (
    Shandong91HeterogeneousRawBody,
    Shandong91MaskedDiffusion,
)


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _batch(dataset: Shandong91ReliableDataset, indices: list[int], device: torch.device) -> dict[str, torch.Tensor]:
    samples = [dataset[index] for index in indices]
    keys = (
        "actual", "forecast", "residual", "actual_valid_mask",
        "forecast_valid_mask", "residual_valid_mask", "time_mark", "effective_mask",
    )
    result = {key: torch.stack([sample[key] for sample in samples]).to(device) for key in keys}
    for key in ("actual", "forecast", "residual", "time_mark"):
        result[key] = result[key].float()
    return result


def _resource_losses(element_loss: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    result = {}
    for index, name in enumerate(CHANNELS):
        current_mask = mask[..., index]
        numerator = (element_loss[..., index] * current_mask).sum()
        result[name] = float((numerator / current_mask.sum().clamp_min(1)).detach())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shandong91/raw_body_heterogeneous_gate1_smoke.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Gate 1 output: {output}")
    output.mkdir(parents=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gate = config["gate1"]
    if gate.get("formal_training") is not False or gate.get("generation") is not False:
        raise ValueError("Gate 1 must not enable formal training or generation")
    if args.amp and args.device != "cuda":
        raise ValueError("--amp requires --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(gate["seed"]))

    dataset = Shandong91ReliableDataset(config["data"]["data_path"], "train")
    batch = _batch(dataset, [int(v) for v in gate["batch_indices"]], device)
    node_features = dataset.node_features.to(device=device, dtype=torch.float32)
    adjacency = dataset.adjacency_with_self.to(device=device)
    denoiser = Shandong91HeterogeneousRawBody(config["model"], node_features, adjacency).to(device)
    model = Shandong91MaskedDiffusion(denoiser, config["diffusion"]).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(gate["learning_rate"]))
    scaler = torch.amp.GradScaler(device.type, enabled=bool(args.amp))
    timestep = torch.full((len(gate["batch_indices"]),), int(gate["fixed_timestep"]), device=device, dtype=torch.long)
    noise = torch.randn_like(batch["residual"])
    mask = batch["effective_mask"].bool()
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    frozen_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}

    history = []
    started = time.perf_counter()
    for step in range(int(gate["steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(args.amp)):
            prediction, element_loss = model.prediction_and_error(batch, timestep, noise)
            loss = model.masked_loss(element_loss, mask)
        row = {"step": step, "overall": float(loss.detach()), **_resource_losses(element_loss, mask)}
        history.append(row)
        if step == int(gate["steps"]):
            break
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(_finite(gradient) for gradient in gradients):
            raise RuntimeError(f"non-finite or missing gradients at step {step}")
        scaler.step(optimizer)
        scaler.update()
    elapsed = time.perf_counter() - started

    changed = {
        name: float((value.detach() - initial_parameters[name]).abs().max())
        for name, value in model.named_parameters()
    }
    changed_nonzero = {name: delta for name, delta in changed.items() if delta > 0}
    buffer_changes = {}
    for name, value in model.named_buffers():
        current = value.detach()
        before = frozen_buffers[name]
        if torch.equal(current, before):
            buffer_changes[name] = 0.0
        else:
            finite = torch.isfinite(current) & torch.isfinite(before)
            buffer_changes[name] = (
                float((current[finite].float() - before[finite].float()).abs().max())
                if bool(finite.any()) else float("inf")
            )
    initial, final = history[0], history[-1]
    loss_checks = {"overall": final["overall"] < initial["overall"]}
    loss_checks.update({name: final[name] < initial[name] for name in CHANNELS})
    tensor_finite = _finite(prediction) and _finite(element_loss) and all(
        torch.isfinite(torch.tensor(list(row.values()), dtype=torch.float64)).all() for row in history
    )
    core_name = "denoiser.encoder_blocks.0.conv1.weight"
    checks = {
        "bounded_smoke_scope": int(gate["steps"]) <= 50 and gate["formal_training"] is False and gate["generation"] is False,
        "all_tensors_finite": bool(tensor_finite),
        "overall_loss_decreased": loss_checks["overall"],
        "wind_loss_decreased": loss_checks["Wind"],
        "solar_loss_decreased": loss_checks["Solar"],
        "load_loss_decreased": loss_checks["Load"],
        "core_backbone_updated": changed.get(core_name, 0.0) > 0,
        "all_registered_buffers_unchanged": all(delta == 0 for delta in buffer_changes.values()),
    }
    passed = all(checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Gate 1 fixed-batch short overfit only; no formal training; no generation",
        "device": str(device),
        "amp": bool(args.amp),
        "steps": int(gate["steps"]),
        "batch_indices": gate["batch_indices"],
        "fixed_timestep": int(gate["fixed_timestep"]),
        "formal_objective": config["loss"]["formal_objective"],
        "checks": {name: "PASS" if value else "FAIL" for name, value in checks.items()},
        "initial_losses": {key: value for key, value in initial.items() if key != "step"},
        "final_losses": {key: value for key, value in final.items() if key != "step"},
        "loss_history": history,
        "changed_parameter_count": len(changed_nonzero),
        "core_backbone_max_abs_update": changed.get(core_name, 0.0),
        "buffer_max_abs_changes": buffer_changes,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "formal_training": "NOT RUN",
        "generation": "NOT RUN",
    }
    (output / "gate1_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Shandong91 heterogeneous Raw Body — Gate 1 smoke",
        "", f"- Overall: **{report['status']}**", f"- Device: **{device}**",
        f"- AMP: **{'ON' if args.amp else 'OFF'}**", "- Formal training: **NOT RUN**",
        "- Generation: **NOT RUN**", "", "## Checks", "",
    ]
    lines.extend(f"- {name}: **{status}**" for name, status in report["checks"].items())
    lines.extend(["", "## Initial/final losses", "", "```json", json.dumps({"initial": report["initial_losses"], "final": report["final_losses"]}, indent=2), "```", ""])
    (output / "gate1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
