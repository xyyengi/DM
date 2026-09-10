"""Mandatory preflight for the full independent joint wind-solar tail expert."""

from __future__ import annotations

import argparse
import json
import math
import io
import hashlib
import platform
import time
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow both ``python tools/...py`` and ``python -m tools...``.  The server
# launch scripts use the former, so root import resolution must be explicit.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data, get_station_dataloader
from station_jstd_targets import fit_station_jstd_event_thresholds, build_station_jstd_target_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--secondary-adjacency", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def batch(device: torch.device, length: int) -> dict[str, torch.Tensor]:
    count = 2
    time_support = torch.zeros(count, length, device=device)
    time_support[0, length // 4:length // 2] = 1.0
    station_support = torch.zeros(count, 24, length, device=device)
    station_support[0, :13, length // 4:length // 2] = 1.0
    return {
        "residual_target": torch.randn(count, 24, length, device=device),
        "residual": torch.randn(count, 24, length, device=device),
        "residual_scale": torch.ones(count, 24, length, device=device),
        "forecast": torch.rand(count, 24, length, device=device),
        "calendar": torch.rand(count, 8, length, device=device),
        "lead": torch.rand(count, 2, length, device=device),
        "valid_mask": torch.ones(count, 24, length, device=device),
        "recent_error": torch.randn(count, 24, 24, device=device),
        "recent_error_mask": torch.ones(count, 24, 1, device=device),
        "node_state": torch.rand(count, 24, 4, length, device=device),
        "jstd_event_active": torch.tensor([1.0, 0.0], device=device),
        "jstd_event_time_support": time_support,
        "jstd_event_station_support": station_support,
        "jstd_sample_weight": torch.ones(count, device=device),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "FAIL", "reason": "preflight has not completed", "launch_eligible": False}))
    torch.manual_seed(2027)
    torch.set_num_threads(4)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = config["model"]
    if not bool(model_config.get("independent_joint_tail_training", False)):
        raise ValueError("preflight requires independent_joint_tail_training=true")
    if bool(model_config.get("use_body_tail_experts", False)):
        raise ValueError("independent tail must not contain a body-tail adapter")
    if bool(model_config.get("use_jstd_event_hypothesis", False)):
        raise ValueError("oracle event hypotheses are forbidden")
    event_balanced_names = (
        "event_balanced_ramp_loss_weight",
        "event_balanced_shape_loss_weight",
        "event_balanced_slow_loss_weight",
    )
    event_balanced_v2 = any(float(model_config.get(name, 0.0)) > 0.0 for name in event_balanced_names)
    if event_balanced_v2:
        if not all(float(model_config.get(name, 0.0)) > 0.0 for name in event_balanced_names):
            raise ValueError("V2 requires ramp, shape and slow event-balanced losses together")
        if float(model_config.get("ramp_auxiliary_loss_weight", 0.0)) != 0.0:
            raise ValueError("V2 must replace, not stack, the globally averaged ramp loss")
        if float(model_config.get("tail_multiscale_slow_loss_weight", 0.0)) != 0.0:
            raise ValueError("V2 must replace, not stack, the globally averaged slow loss")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless this is explicitly a CPU-only check")
    static = load_station_static_data(args.data_path)
    secondary = torch.as_tensor(
        np.load(args.secondary_adjacency), dtype=torch.float32
    )
    model = Station24DiffusionModel(
        model_config,
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
        secondary,
    ).to(device)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_state = source.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("checkpoint lacks raw model_state_dict")
    target = model.state_dict()
    compatible = {
        key: value for key, value in source_state.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    missing = sorted(set(target).difference(compatible))
    if missing:
        raise ValueError(f"Raw initialization incomplete: {missing}")
    model.load_state_dict(compatible, strict=True)
    if any("tail" in name for name, _ in model.named_parameters()):
        raise ValueError("independent tail unexpectedly contains a legacy tail parameter")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("independent full expert must have no frozen parameters")
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    fixed = batch(device, int(model_config["sequence_length"]))
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(fixed, timestep=torch.tensor([1, 2], device=device), noise=torch.randn(2, 24, int(model_config["sequence_length"]), device=device))
    else:
        loss = model(fixed, timestep=torch.tensor([1, 2], device=device), noise=torch.randn(2, 24, int(model_config["sequence_length"]), device=device))
    if not torch.isfinite(loss):
        raise ValueError("nonfinite independent-tail preflight loss")
    if event_balanced_v2:
        for name in ("event_balanced_ramp", "event_balanced_shape", "event_balanced_slow"):
            value = model.diffusion.last_loss_components[name]
            if not torch.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"V2 loss component is nonfinite or zero: {name}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = {
        name: float(parameter.grad.detach().norm().cpu())
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    if len(gradients) != len(list(model.parameters())):
        raise ValueError("one or more full-tail parameters did not receive gradients")
    if any(not math.isfinite(value) or value <= 0.0 for value in gradients.values()):
        raise ValueError("one or more full-tail parameter gradients are zero")
    scaler.step(optimizer)
    scaler.update()
    changed = [
        name for name, value in model.state_dict().items()
        if value.dtype.is_floating_point and not torch.equal(value, before[name])
    ]
    unchanged = [name for name, parameter in model.named_parameters()
                 if torch.equal(parameter.detach(), before[name])]
    if unchanged:
        raise ValueError(f"optimizer left parameters unchanged: {unchanged}")
    modified_buffers = [name for name, value in model.named_buffers()
                        if name in before and not torch.equal(value, before[name])]
    if modified_buffers:
        raise ValueError(f"fixed buffers changed: {modified_buffers}")
    model.eval()
    noise = torch.randn_like(fixed["residual_target"])
    timestep = torch.tensor([100, 200], device=device).clamp(max=model_config["num_steps"] - 1)
    learning = []
    for step in range(13):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = model(fixed, timestep=timestep, noise=noise)
        learning.append({key: float(value) for key, value in model.diffusion.last_loss_components.items()})
        if not torch.isfinite(loss):
            raise ValueError("fixed batch became nonfinite")
        if step < 12:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    if learning[-1]["epsilon"] >= learning[0]["epsilon"]:
        raise ValueError("synthetic fixed-batch epsilon failed to decrease")
    thresholds = fit_station_jstd_event_thresholds(args.data_path, model_config)
    targets = build_station_jstd_target_arrays(args.data_path, "train", thresholds,
        event_sampling_target_fraction=model_config["independent_tail_event_sampling_fraction"])
    _, dataset = get_station_dataloader(args.data_path, "train", source["residual_scale"],
        batch_size=2, seed=2027, condition_config=model_config,
        state_thresholds=source["state_thresholds"], jstd_targets=targets)
    strata = {}
    for source_kind in ("wind", "solar"):
        for direction in ("positive", "negative"):
            strata[source_kind + "_" + direction] = sorted({int(row["sample_index"])
                for row in targets.catalog if row["source"] == source_kind and row["direction"] == direction})[:2]
    strata["ordinary"] = np.flatnonzero(targets.event_active == 0)[:2].tolist()
    real_learning = {}
    for label, indices in strata.items():
        if not indices:
            raise ValueError(f"no training windows for required stratum {label}")
        indices = (indices * 2)[:2]
        rows = [dataset[index] for index in indices]
        current = {key: torch.stack([row[key] for row in rows]).to(device)
                   for key in rows[0] if key != "sample_index"}
        model.load_state_dict(compatible, strict=True)
        model.eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        fixed_noise = torch.randn_like(current["residual_target"])
        values = []
        for step in range(13):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(current, timestep=timestep, noise=fixed_noise)
            values.append({key: float(value) for key, value in model.diffusion.last_loss_components.items()})
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite real fixed batch: {label}")
            if step < 12:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        real_learning[label] = {"indices": indices, "initial": values[0], "final": values[-1]}
        if values[-1]["total"] >= values[0]["total"]:
            raise ValueError(f"real fixed-batch objective did not decrease: {label}")
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = torch.load(stream, map_location=device, weights_only=True)
    model.load_state_dict(restored, strict=True)
    # Full configured reverse chain; no reduced-step surrogate.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    generation_started = time.perf_counter()
    with torch.no_grad():
        torch.manual_seed(424242)
        generated = model.generate(fixed, n_samples=1)
        altered = {key: value.clone() for key, value in fixed.items()}
        for key in ("residual_target", "residual", "actual", "jstd_event_hypothesis"):
            altered[key] = torch.randn_like(fixed["residual_target"])
        torch.manual_seed(424242)
        regenerated = model.generate(altered, n_samples=1)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    if not torch.isfinite(generated).all() or not torch.equal(generated, regenerated):
        raise ValueError("generation is nonfinite or depends on future labels")
    report = {
        "status": "PASS",
        "scope": "CPU-only" if device.type == "cpu" else "CUDA/FP16 synthetic engineering checks",
        "seed": 2027,
        "command": sys.argv,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "secondary_graph_sha256": hashlib.sha256(Path(args.secondary_adjacency).read_bytes()).hexdigest(),
        "gradient_norms": gradients,
        "optimizer_unchanged_parameters": unchanged,
        "modified_persistent_buffers": modified_buffers,
        "nonpersistent_runtime_counters_excluded": [name for name, _ in model.named_buffers() if name not in before],
        "device": str(device),
        "cuda_amp_executed": device.type == "cuda",
        "raw_checkpoint_loaded_tensor_count": len(compatible),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "gradient_parameter_count": len(gradients),
        "optimizer_changed_tensor_count": len(changed),
        "loss": float(loss.detach().cpu()),
        "ramp_loss": float(model.diffusion.last_loss_components["ramp"].cpu()),
        "slow_projection_loss": float(model.diffusion.last_loss_components["tail_multiscale_slow"].cpu()),
        "event_balanced_losses": {
            name: float(learning[0][name])
            for name in ("event_balanced_ramp", "event_balanced_shape", "event_balanced_slow")
        },
        "event_labels_generation_condition": False,
        "causal_generation_invariance": "PASS",
        "synthetic_fixed_batch_learning": learning,
        "real_event_stratified_learning": real_learning,
        "event_sampler_audit": targets.audit,
        "final_event_quality": "NOT RUN",
        "target_cuda_performance_probe": (
            {
                "status": "PASS",
                "two_full_reverse_chains_seconds": generation_seconds,
                "peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3,
            }
            if device.type == "cuda"
            else "NOT RUN"
        ),
        "save_reload_generation": "PASS",
        "launch_eligible": device.type == "cuda",
        "formal_500_member_generation": "NOT RUN",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("INDEPENDENT_JOINT_TAIL_PREFLIGHT_PASS " + json.dumps(report))


if __name__ == "__main__":
    main()
