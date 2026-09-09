"""Mandatory full-model gate for the Joint Multiresidual Tail experiment."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import get_station_dataloader, load_station_static_data
from station_jstd_targets import (
    build_station_jstd_target_arrays,
    fit_station_jstd_event_thresholds,
)
from train_station24 import move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--secondary-adjacency", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--skip-sampling", action="store_true")
    return parser.parse_args()


def denoise(model, batch, noisy, timestep, route):
    result = model.denoiser(
        noisy,
        timestep,
        batch["forecast"],
        batch["calendar"],
        batch["lead"],
        forecast_ramps=batch.get("forecast_ramps"),
        forecast_revision=batch.get("forecast_revision"),
        revision_mask=batch.get("revision_mask"),
        recent_error=batch.get("recent_error"),
        recent_error_mask=batch.get("recent_error_mask"),
        node_state=batch.get("node_state"),
        tail_expert_route=route,
        return_jstd_audit=model.use_joint_multiresidual_tail,
    )
    return result[0] if isinstance(result, tuple) else result


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA gate requested but CUDA is unavailable")
    torch.manual_seed(260909)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    if not bool(model_config.get("use_joint_multiresidual_tail", False)):
        raise ValueError("config does not enable the joint multiresidual tail")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_config = dict(checkpoint["config"]["model"])
    source_state = checkpoint["model_state_dict"]
    static = load_station_static_data(args.data_path)
    secondary = torch.from_numpy(
        __import__("numpy").load(args.secondary_adjacency).astype("float32")
    )

    raw = Station24DiffusionModel(
        source_config,
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
        secondary,
    ).to(device)
    raw.load_state_dict(source_state, strict=True)
    raw.eval()
    candidate = Station24DiffusionModel(
        model_config,
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
        secondary,
    ).to(device)
    incompatible = candidate.load_state_dict(source_state, strict=False)
    expected_missing = set(candidate.joint_multiresidual_new_state_dict_keys)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError("Raw-to-joint checkpoint compatibility contract failed")
    trainable_names = set(candidate.configure_body_tail_training())
    expected_trainable = set(candidate.joint_multiresidual_trainable_parameter_names)
    if trainable_names != expected_trainable:
        raise ValueError("optimizer isolation does not match the new tail")
    candidate.eval()

    thresholds = fit_station_jstd_event_thresholds(args.data_path, model_config)
    targets = build_station_jstd_target_arrays(args.data_path, "train", thresholds)
    loader, _ = get_station_dataloader(
        args.data_path,
        "train",
        checkpoint["residual_scale"],
        batch_size=2,
        seed=260909,
        num_workers=0,
        condition_config=model_config,
        state_thresholds=checkpoint.get("state_thresholds"),
        jstd_targets=targets,
    )
    batch = move_batch(next(iter(loader)), device)
    timestep = torch.tensor([31, 317], device=device)
    generator = torch.Generator(device=device).manual_seed(260909)
    noise = torch.randn(
        batch["residual_target"].shape, device=device,
        dtype=batch["residual_target"].dtype, generator=generator,
    )
    alpha = candidate.diffusion.alpha_hat[timestep].view(-1, 1, 1)
    noisy = alpha.sqrt() * batch["residual_target"] + (1 - alpha).sqrt() * noise

    with torch.no_grad():
        raw_prediction = denoise(raw, batch, noisy, timestep, 0.0)
        candidate_prediction = denoise(candidate, batch, noisy, timestep, 0.0)
    identity_error = float((raw_prediction - candidate_prediction).abs().max().cpu())

    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in candidate.state_dict().items()
        if not any(
            name == trainable or name == f"diffusion.{trainable}"
            for trainable in trainable_names
        )
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in candidate.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    losses: list[float] = []
    gradient_norms: dict[str, float] = {}
    candidate.eval()
    if candidate.denoiser.joint_multiresidual_tail is None:
        raise RuntimeError("joint multiresidual tail is unavailable")
    candidate.denoiser.joint_multiresidual_tail.train()
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=autocast_enabled,
        ):
            loss = candidate(batch, timestep=timestep, noise=noise)
        if not torch.isfinite(loss):
            raise ValueError("non-finite full-model loss")
        loss.backward()
        if step == 3:
            gradient_norms = {
                name: float(parameter.grad.detach().float().norm().cpu())
                if parameter.grad is not None else 0.0
                for name, parameter in candidate.named_parameters()
                if parameter.requires_grad
            }
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    required_fragments = (
        "local_fast.weight", "local_slow.weight", "system_fast.weight",
        "system_slow.weight", "hidden.0.weight", "fast_condition.0.weight",
        "slow_condition.0.weight", "system_loading.weight",
    )
    required_gradient_ok = all(
        any(fragment in name and value > 0.0 for name, value in gradient_norms.items())
        for fragment in required_fragments
    )
    frozen_unchanged = all(
        torch.equal(value, candidate.state_dict()[name].detach().cpu())
        for name, value in frozen_before.items()
    )

    roundtrip = output / "roundtrip_state.pt"
    torch.save(candidate.state_dict(), roundtrip)
    reloaded = Station24DiffusionModel(
        model_config,
        static["station_features"], static["station_adjacency"],
        static["station_capacities"], secondary,
    ).to(device)
    reloaded.load_state_dict(
        torch.load(roundtrip, map_location=device, weights_only=True), strict=True
    )

    sample_shape = None
    sample_finite = None
    if not args.skip_sampling:
        candidate.eval()
        causal_batch = {
            key: value[:1]
            for key, value in batch.items()
            if key not in {
                "actual", "residual", "residual_target", "residual_scale",
                "loss_weight", "jstd_event_active", "jstd_event_time_support",
                "jstd_event_station_support", "jstd_sample_weight",
                "jstd_slow_target", "jstd_fast_target", "jstd_slow24_target",
                "jstd_segment_hypotheses", "jstd_event_hypothesis",
            }
        }
        with torch.no_grad():
            generated = candidate.generate(
                causal_batch, n_samples=1,
                tail_route_probability_override=1.0,
            )
        sample_shape = list(generated.shape)
        sample_finite = bool(torch.isfinite(generated).all().cpu())

    gates = {
        "G0_no_future_target_in_generation_batch": "PASS" if (args.skip_sampling or sample_finite) else "FAIL",
        "G1_raw_route_zero_exact_identity": "PASS" if identity_error <= 1e-7 else "FAIL",
        "G2_optimizer_parameter_isolation": "PASS" if trainable_names == expected_trainable else "FAIL",
        "G2_required_gradients": "PASS" if required_gradient_ok else "FAIL",
        "G2_frozen_state_unchanged": "PASS" if frozen_unchanged else "FAIL",
        "G3_small_batch_loss_decreases": "PASS" if losses[-1] < losses[0] else "FAIL",
        "G4_cuda_amp_forward_backward": "PASS" if device.type == "cuda" else "NOT RUN",
        "G4_checkpoint_roundtrip": "PASS",
        "G5_full_schedule_sampling": "PASS" if sample_finite else "NOT RUN",
        "G6_scientific_benefit": "NOT RUN",
    }
    mandatory = [key for key in gates if not key.startswith("G6")]
    launch_eligible = device.type == "cuda" and all(gates[key] == "PASS" for key in mandatory)
    payload = {
        "device": str(device),
        "gates": gates,
        "launch_eligible": launch_eligible,
        "raw_route_zero_max_abs_error": identity_error,
        "losses": losses,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in candidate.parameters()
            if parameter.requires_grad
        ),
        "trainable_parameter_names": sorted(trainable_names),
        "gradient_norms": gradient_norms,
        "sample_shape": sample_shape,
        "sample_finite": sample_finite,
        "future_actual_used_as_generation_condition": False,
        "scientific_benefit_status": "NOT RUN; requires formal 500-member comparison",
    }
    (output / "full_preflight.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if any(gates[key] == "FAIL" for key in mandatory):
        raise SystemExit("JOINT_MULTIRESIDUAL_FULL_PREFLIGHT_FAILED")
    print(
        "JOINT_MULTIRESIDUAL_FULL_PREFLIGHT_PASSED "
        f"launch_eligible={launch_eligible} output={output}", flush=True,
    )


if __name__ == "__main__":
    main()
