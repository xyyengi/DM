"""Bounded CPU/CUDA engineering preflight for faithful24 Shandong91 V2."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, fit_faithful24_state_thresholds,
    threshold_document, threshold_sha256,
)
from src.models.shandong91_faithful24_diffusion import (
    Shandong91Faithful24Diffusion, Shandong91HeterogeneousRawBodyV2,
)

CHANNELS = ("Wind", "Solar", "Load")


def gradient(model, core) -> dict:
    values = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return {
        "finite": bool(values and all(torch.isfinite(value).all() for value in values)),
        "l2": float(sum(value.detach().double().square().sum() for value in values).sqrt()),
        "core_l2": float(core.grad.detach().double().norm()) if core.grad is not None else 0.0,
        "tensors": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shandong91/raw_body_v2_faithful24.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--full-model", action="store_true",
                        help="use formal widths; mandatory for target CUDA/AMP gate")
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.amp and args.device != "cuda": raise ValueError("AMP requires CUDA")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    config = yaml.safe_load(Path(args.config).read_text("utf-8"))
    torch.manual_seed(20271008)
    thresholds = fit_faithful24_state_thresholds(
        config["data"]["data_path"],
        low_quantile=float(config["model"]["state_low_quantile"]),
        high_quantile=float(config["model"]["state_high_quantile"]),
        ramp_quantile=float(config["model"]["state_ramp_quantile"]),
        ramp_lags=tuple(config["model"]["state_ramp_lags"]),
    )
    dataset = Shandong91Faithful24Dataset(config["data"]["data_path"], "train", thresholds)
    raw = dataset[0]
    batch = {key: value.unsqueeze(0).to(device) for key, value in raw.items()}
    for key in ("actual", "forecast", "residual", "time_mark", "recent_error", "node_state"):
        batch[key] = batch[key].float()
    small = copy.deepcopy(config["model"])
    if not args.full_model:
        small.update({"base_channels": 8, "channel_multipliers": [1, 2, 2],
                      "group_norm_groups": 4, "state_channels": [4, 8, 8], "dropout": 0.0})
    denoiser = Shandong91HeterogeneousRawBodyV2(
        small, dataset.node_features.to(device), dataset.adjacency_with_self.float().to(device)
    ).to(device)
    model = Shandong91Faithful24Diffusion(denoiser, config["diffusion"]).to(device)
    checks = {}
    def record(name, passed, detail): checks[name] = {"status": "PASS" if passed else "FAIL", "detail": detail}

    expected_shapes = {"forecast": [1,168,91,3], "residual": [1,168,91,3],
                       "effective_mask": [1,168,91,3], "node_state": [1,168,91,12],
                       "recent_error": [1,24,91,3], "recent_error_valid_mask": [1,24,91,3],
                       "time_mark": [1,168,8]}
    tensor_audit = {key: {"shape": list(batch[key].shape), "device": str(batch[key].device),
                          "dtype": str(batch[key].dtype),
                          "finite": bool(torch.isfinite(batch[key]).all()) if batch[key].is_floating_point() else True}
                    for key in expected_shapes}
    record("new_tensor_device_dtype_shape", all(tensor_audit[k]["shape"] == v and tensor_audit[k]["device"] == str(device) and tensor_audit[k]["finite"] for k,v in expected_shapes.items()), tensor_audit)
    record("lead_disabled", not small.get("use_lead_condition", False) and "lead" not in denoiser.forward.__code__.co_varnames, "no lead/window_position argument")
    record("physical_graph_only", small.get("spatial_mode") == "fixed_graph" and not small.get("use_dual_fixed_graph", False), "single published 91-node physical adjacency")
    condition_contract = dataset.condition_manifest()
    causal_detail = {
        "recent_error_future_actual_used": condition_contract["recent_error_future_actual_used"],
        "recent_error": condition_contract["recent_error"], "lead": condition_contract["lead"],
        "state_threshold_sha256": condition_contract["state_threshold_sha256"],
    }
    record("causal_conditions", condition_contract["recent_error_future_actual_used"] is False and condition_contract["lead"] == "disabled", causal_detail)

    timestep = torch.tensor([137], device=device); noise = torch.randn_like(batch["residual"])
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
        prediction, error = model.prediction_and_error(batch, timestep, noise)
        loss = model.masked_loss(error, batch["effective_mask"])
    record("forward_shape", list(prediction.shape) == [1,168,91,3], list(prediction.shape))
    record("forward_finite", bool(torch.isfinite(prediction).all() and torch.isfinite(loss)), float(loss.detach()))
    model.zero_grad(set_to_none=True); loss.backward()
    overall = gradient(model, denoiser.encoder_blocks[0].conv1.weight)
    record("backward_overall_finite_nonzero", overall["finite"] and overall["l2"] > 0 and overall["core_l2"] > 0, overall)
    resource_gradients = {}
    for channel, name in enumerate(CHANNELS):
        model.zero_grad(set_to_none=True)
        _, current_error = model.prediction_and_error(batch, timestep, noise)
        current_mask = torch.zeros_like(batch["effective_mask"], dtype=torch.bool)
        current_mask[..., channel] = batch["effective_mask"][..., channel]
        current_loss = model.masked_loss(current_error, current_mask)
        current_loss.backward(); item = gradient(model, denoiser.encoder_blocks[0].conv1.weight)
        item["loss"] = float(current_loss.detach()); item["effective_elements"] = int(current_mask.sum())
        resource_gradients[name] = item
        record(f"{name.lower()}_gradient_finite_nonzero", item["finite"] and item["l2"] > 0 and item["core_l2"] > 0, item)

    changed_prediction = prediction.detach().clone(); changed_noise = noise.clone()
    inactive = ~batch["effective_mask"].bool(); changed_prediction[inactive] += 123; changed_noise[inactive] -= 321
    changed_loss = model.masked_loss((changed_prediction - changed_noise).square(), batch["effective_mask"])
    delta = float(abs(changed_loss - loss.detach()))
    record("mask_invariance", delta <= 1e-7, {"absolute_delta": delta})

    model.zero_grad(set_to_none=True)
    _, update_error = model.prediction_and_error(batch, timestep, noise)
    update_loss = model.masked_loss(update_error, batch["effective_mask"])
    core = denoiser.encoder_blocks[0].conv1.weight; before = core.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    update_loss.backward(); optimizer.step(); update_delta = core.detach() - before
    record("optimizer_core_update", bool(torch.isfinite(core).all() and update_delta.abs().max() > 0), {"max_abs_delta": float(update_delta.abs().max()), "l2": float(update_delta.double().norm())})

    model.eval(); probe_noise = torch.randn_like(batch["residual"])
    with torch.no_grad(): before_reload = model.denoiser(probe_noise, timestep, batch["forecast"], batch["forecast_valid_mask"], batch["time_mark"], batch["recent_error"], batch["recent_error_valid_mask"], batch["node_state"])
    checkpoint = output / "reload_probe.pt"
    torch.save({"model_identifier": denoiser.architecture, "model_state_dict": model.state_dict(),
                "state_thresholds": threshold_document(thresholds), "state_threshold_sha256": threshold_sha256(thresholds)}, checkpoint)
    reload_denoiser = Shandong91HeterogeneousRawBodyV2(small, dataset.node_features.to(device), dataset.adjacency_with_self.float().to(device)).to(device)
    reloaded = Shandong91Faithful24Diffusion(reload_denoiser, config["diffusion"]).to(device)
    reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"], strict=True); reloaded.eval()
    with torch.no_grad(): after_reload = reloaded.denoiser(probe_noise, timestep, batch["forecast"], batch["forecast_valid_mask"], batch["time_mark"], batch["recent_error"], batch["recent_error_valid_mask"], batch["node_state"])
    reload_delta = float((before_reload-after_reload).abs().max())
    record("checkpoint_save_reload", reload_delta <= 1e-6, {"max_abs_delta": reload_delta})

    generation_mask = batch["effective_mask"].bool(); initial = torch.randn_like(batch["forecast"])
    with torch.no_grad(): sampled = reloaded.sample(batch, generation_mask, method="ddim", inference_steps=2, initial_noise=initial)
    record("bounded_sampler_shape_finite", list(sampled.shape) == [1,168,91,3] and bool(torch.isfinite(sampled).all()), list(sampled.shape))
    inactive_max = float(sampled.masked_select(~generation_mask).abs().max())
    record("sampler_inactive_zero", inactive_max == 0.0, {"inactive_max_abs": inactive_max})
    left = dataset.denormalize_mw(batch["forecast"] + sampled)
    right = dataset.denormalize_mw(batch["forecast"]) + dataset.denormalize_mw(sampled)
    sign_delta = float((left-right).abs().max())
    sign_relative = float(((left-right).abs() / right.abs().clamp_min(1.0)).max())
    sign_match = bool(torch.allclose(left, right, rtol=1e-6, atol=1e-4))
    record("inverse_normalization_residual_sign", sign_match, {"max_abs_delta": sign_delta, "max_relative_delta": sign_relative, "formula": "actual=forecast+residual"})
    experiment_name = config["experiment"]["name"]
    record(
        "v1_v2_isolation",
        denoiser.architecture != "shandong91_heterogeneous_raw_body_v1"
        and experiment_name.startswith(denoiser.architecture),
        {"model": denoiser.architecture, "experiment": experiment_name},
    )

    status = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    report = {"status": status, "scope": "bounded engineering preflight; no formal training", "device": str(device), "amp": args.amp, "full_model": args.full_model, "checks": checks, "resource_gradients": resource_gradients, "formal_training": "NOT RUN", "formal_generation": "NOT RUN"}
    (output / "report.json").write_text(json.dumps(report, indent=2), "utf-8")
    lines = ["# Shandong91 faithful24 V2 preflight", "", f"- Overall: **{status}**", f"- Device: **{device}**", "- Formal training: **NOT RUN**", "- Formal generation: **NOT RUN**", ""]
    lines += [f"- {name}: **{item['status']}**" for name,item in checks.items()]
    (output / "report.md").write_text("\n".join(lines), "utf-8")
    print(json.dumps(report, indent=2))
    if status != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
