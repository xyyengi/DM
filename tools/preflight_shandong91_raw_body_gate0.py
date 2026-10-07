"""CPU-only Gate 0 for the 91-node heterogeneous Raw Body."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.shandong91_reliable import CHANNELS, Shandong91ReliableDataset
from src.models.shandong91_conditioned_diffusion import (
    Shandong91HeterogeneousRawBody,
    Shandong91MaskedDiffusion,
)


def tensor_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all()) if value.is_floating_point() else True


def grad_summary(model: torch.nn.Module) -> dict[str, float | bool | int]:
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(tensor_finite(grad) for grad in gradients)
    squared = sum(float(grad.detach().double().square().sum()) for grad in gradients)
    nonzero = sum(int(torch.count_nonzero(grad)) for grad in gradients)
    return {
        "finite": finite,
        "l2_norm": squared ** 0.5,
        "nonzero_elements": nonzero,
        "tensor_count": len(gradients),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shandong91/raw_body_heterogeneous_168h.yaml")
    parser.add_argument("--output", default="local_checks/shandong91_raw_body_gate0_20261008")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Gate 0 output: {output}")
    output.mkdir(parents=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["gate0"]["seed"])
    torch.manual_seed(seed)
    device = torch.device("cpu")
    dtype = torch.float32
    dataset = Shandong91ReliableDataset(config["data"]["data_path"], "train")
    sample = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device=device)
        for key, value in sample.items()
        if key in {
            "actual", "forecast", "residual", "actual_valid_mask",
            "forecast_valid_mask", "residual_valid_mask", "time_mark",
            "effective_mask",
        }
    }
    for key in ("actual", "forecast", "residual", "time_mark"):
        batch[key] = batch[key].to(dtype=dtype)
    adjacency = dataset.adjacency_with_self.to(device=device)
    node_features = dataset.node_features.to(device=device, dtype=dtype)
    model_config = config["model"]
    diffusion_config = config["diffusion"]
    denoiser = Shandong91HeterogeneousRawBody(
        model_config, node_features, adjacency
    ).to(device=device, dtype=dtype)
    model = Shandong91MaskedDiffusion(denoiser, diffusion_config).to(device)
    model.train()

    checks: dict[str, dict[str, object]] = {}
    def record(name: str, passed: bool, detail: object) -> None:
        checks[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}

    columns = json.loads((dataset.root / "static/node_feature_columns.json").read_text(encoding="utf-8"))
    type_columns = columns[4:7]
    record("explicit_resource_type_encoding", type_columns == ["has_wind", "has_solar", "has_load"], type_columns)
    record("legacy_lead_branch_disabled", not model_config.get("use_lead_condition", False)
           and "lead" not in Shandong91HeterogeneousRawBody.forward.__annotations__,
           {"config": model_config.get("use_lead_condition"), "forward_arguments": list(Shandong91HeterogeneousRawBody.forward.__code__.co_varnames[:6])})
    record("tail_aux_event_partial_disabled", not any(bool(model_config.get(name, False)) for name in (
        "use_body_tail_experts", "use_jstd_tail", "use_joint_multiresidual_tail",
        "use_event_balanced_sampling", "use_self_localization", "use_protected_partial",
        "use_auxiliary_losses")), "all forbidden switches false")

    batch_size = batch["residual"].shape[0]
    timestep = torch.full((batch_size,), 137, device=device, dtype=torch.long)
    noise = torch.randn_like(batch["residual"])
    prediction, element_loss = model.prediction_and_error(batch, timestep, noise)
    mask = batch["effective_mask"].bool()
    overall_loss = model.masked_loss(element_loss, mask)
    record("forward_shape", tuple(prediction.shape) == (1, 168, 91, 3), list(prediction.shape))
    record("nan_inf", tensor_finite(prediction) and tensor_finite(element_loss) and tensor_finite(overall_loss),
           {"prediction": tensor_finite(prediction), "loss": tensor_finite(overall_loss)})

    resource_stats: dict[str, dict[str, float | int]] = {}
    total_count = int(mask.sum())
    resource_means = []
    for index, name in enumerate(CHANNELS):
        resource_mask = mask[..., index]
        count = int(resource_mask.sum())
        numerator = float((element_loss[..., index] * resource_mask).sum().detach())
        mean = numerator / max(count, 1)
        contribution = numerator / max(total_count, 1)
        resource_means.append(mean)
        resource_stats[name] = {
            "valid_elements": count,
            "valid_share": count / max(total_count, 1),
            "unweighted_masked_loss": mean,
            "elementwise_objective_contribution": contribution,
            "objective_contribution_share": contribution / max(float(overall_loss.detach()), 1e-30),
        }
    balanced_loss = sum(resource_means) / len(resource_means)
    objective_diagnostic = {
        "elementwise_masked_mean": float(overall_loss.detach()),
        "resource_wise_balanced_mean_diagnostic_only": balanced_loss,
        "absolute_difference": abs(float(overall_loss.detach()) - balanced_loss),
        "relative_difference": abs(float(overall_loss.detach()) - balanced_loss) / max(abs(float(overall_loss.detach())), 1e-30),
        "formal_objective_changed": False,
        "largest_valid_count_resource": max(resource_stats, key=lambda key: resource_stats[key]["valid_elements"]),
    }

    model.zero_grad(set_to_none=True)
    overall_loss.backward()
    overall_grad = grad_summary(model)
    record("backward", len([p for p in model.parameters() if p.grad is not None]) > 0, overall_grad)
    record("overall_gradient_finite_nonzero", bool(overall_grad["finite"] and overall_grad["l2_norm"] > 0), overall_grad)

    resource_gradients: dict[str, dict[str, float | bool | int]] = {}
    for index, name in enumerate(CHANNELS):
        model.zero_grad(set_to_none=True)
        _, current_element_loss = model.prediction_and_error(batch, timestep, noise)
        current_mask = torch.zeros_like(mask)
        current_mask[..., index] = mask[..., index]
        loss = model.masked_loss(current_element_loss, current_mask)
        loss.backward()
        summary = grad_summary(model)
        core_grad = denoiser.encoder_blocks[0].conv1.weight.grad
        summary["core_backbone_l2_norm"] = float(core_grad.detach().double().norm()) if core_grad is not None else 0.0
        summary["loss"] = float(loss.detach())
        resource_gradients[name] = summary
        record(f"{name.lower()}_gradient_finite_nonzero",
               bool(summary["finite"] and summary["l2_norm"] > 0 and summary["core_backbone_l2_norm"] > 0), summary)

    # Mask invariance uses the same prediction/error outside the effective domain.
    changed_prediction = prediction.detach().clone()
    changed_noise = noise.clone()
    changed_prediction[~mask] += 123.0
    changed_noise[~mask] -= 321.0
    changed_loss = model.masked_loss((changed_prediction - changed_noise).square(), mask)
    invariance_delta = float((changed_loss - overall_loss.detach()).abs())
    record("mask_invariance", invariance_delta <= 1e-7,
           {"original": float(overall_loss.detach()), "changed": float(changed_loss), "absolute_delta": invariance_delta})

    # One Gate-0 optimizer step is not smoke training.
    model.zero_grad(set_to_none=True)
    _, step_element_loss = model.prediction_and_error(batch, timestep, noise)
    step_loss = model.masked_loss(step_element_loss, mask)
    step_loss.backward()
    core_parameter = denoiser.encoder_blocks[0].conv1.weight
    before = core_parameter.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["gate0"]["learning_rate"]))
    optimizer.step()
    delta = core_parameter.detach() - before
    update = {"max_abs_delta": float(delta.abs().max()), "l2_norm": float(delta.double().norm()), "finite": tensor_finite(core_parameter)}
    record("optimizer_core_update", bool(update["finite"] and update["max_abs_delta"] > 0), update)

    model.eval()
    with torch.no_grad():
        reference, _ = model.prediction_and_error(batch, timestep, noise)
    checkpoint_path = output / "reload_probe.pt"
    torch.save({"architecture": denoiser.architecture, "model_config": model_config,
                "diffusion_config": diffusion_config, "state_dict": model.state_dict()}, checkpoint_path)
    restored_denoiser = Shandong91HeterogeneousRawBody(model_config, node_features, adjacency).to(device=device, dtype=dtype)
    restored = Shandong91MaskedDiffusion(restored_denoiser, diffusion_config).to(device)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.eval()
    with torch.no_grad():
        reloaded, _ = restored.prediction_and_error(batch, timestep, noise)
    reload_delta = float((reference - reloaded).abs().max())
    record("checkpoint_save_reload", reload_delta <= 1e-6,
           {"max_abs_output_delta": reload_delta, "strict": True,
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()})

    tensor_contract = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device), "finite": tensor_finite(value)}
        for key, value in batch.items()
    }
    tensor_contract.update({
        "adjacency": {"shape": list(adjacency.shape), "dtype": str(adjacency.dtype), "device": str(adjacency.device), "finite": tensor_finite(adjacency)},
        "node_features": {"shape": list(node_features.shape), "dtype": str(node_features.dtype), "device": str(node_features.device), "finite": tensor_finite(node_features)},
        "prediction": {"shape": list(prediction.shape), "dtype": str(prediction.dtype), "device": str(prediction.device), "finite": tensor_finite(prediction)},
        "noise": {"shape": list(noise.shape), "dtype": str(noise.dtype), "device": str(noise.device), "finite": tensor_finite(noise)},
    })
    tensor_pass = all(item["device"] == "cpu" and item["finite"] for item in tensor_contract.values())
    record("tensor_device_dtype_shape", tensor_pass, tensor_contract)

    hardcoded = {
        "model_node_count": denoiser.node_count,
        "adjacency_shape": list(adjacency.shape),
        "node_embedding_rows": denoiser.condition_encoder.node_embedding.num_embeddings,
        "output_resources": denoiser.output_projection.out_channels,
        "legacy_24_in_new_model_state_shapes": [
            name for name, value in denoiser.state_dict().items() if 24 in value.shape
        ],
    }
    record("no_24_node_shape_residue", not hardcoded["legacy_24_in_new_model_state_shapes"], hardcoded)

    passed = all(item["status"] == "PASS" for item in checks.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "scope": "CPU-only Gate 0; one optimizer step; no smoke/formal training; no generation",
        "device": "cpu", "cuda_amp": "NOT RUN", "formal_training": "NOT RUN",
        "smoke_training": "NOT RUN", "generation": "NOT RUN",
        "architecture": denoiser.architecture, "checks": checks,
        "resource_gradients": resource_gradients,
        "resource_supervision_and_loss": resource_stats,
        "objective_diagnostic": objective_diagnostic,
        "hardcoded_24_audit": hardcoded,
    }
    (output / "gate0_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        "# Shandong91 heterogeneous Raw Body — CPU Gate 0",
        "", f"- Overall: **{report['status']}**", "- CUDA/AMP: **NOT RUN**",
        "- Smoke/formal training: **NOT RUN**", "- Formal generation: **NOT RUN**", "",
        "## Checks", "",
    ]
    summary.extend(f"- {name}: **{value['status']}**" for name, value in checks.items())
    summary.extend(["", "## Objective diagnostic", "", "```json",
                    json.dumps(objective_diagnostic, ensure_ascii=False, indent=2), "```", "",
                    "## Resource supervision and gradients", "", "```json",
                    json.dumps({"supervision": resource_stats, "gradients": resource_gradients}, ensure_ascii=False, indent=2), "```", ""])
    (output / "gate0_report.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

