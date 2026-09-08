#!/usr/bin/env python3
"""Fail-fast structural and causality audit for Station-24 JSTD-MSEP."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import get_station_dataloader, load_station_static_data
from station_graph_prior import load_generation_graphs
from station_jstd_targets import (
    build_station_jstd_target_arrays,
    fit_station_jstd_event_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_config = config["model"]
    if not model_config.get("use_jstd_segment_prior", False):
        raise ValueError("MSEP config must enable the causal segment prior")
    if model_config.get("use_jstd_event_hypothesis", False):
        raise ValueError("MSEP must not enable validation-oracle H1 conditions")
    if float(model_config.get("jstd_issue_loss_weight", -1)) != 0.0:
        raise ValueError("MSEP must not reuse the failed issue-level BCE gate")
    if int(model_config.get("jstd_segment_max_events", 0)) != 2:
        raise ValueError("the audited MSEP protocol carries two ordered slots")
    if model_config.get("jstd_segment_scale_parameterization") != (
        "transformed_normal_v2"
    ):
        raise ValueError(
            "paid MSEP training is blocked: use transformed_normal_v2; "
            "legacy_clamp has zero scale gradients at initialization"
        )
    if not model_config.get("train_jstd_segment_prior_only", False):
        raise ValueError(
            "paid MSEP training is blocked: train the causal prior first while "
            "keeping the H1 renderer frozen"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but CUDA is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    data_path = Path(args.data_path)
    thresholds = fit_station_jstd_event_thresholds(data_path, model_config)
    train = build_station_jstd_target_arrays(data_path, "train", thresholds)
    val = build_station_jstd_target_arrays(data_path, "val", thresholds)
    for target, expected in ((train, 290), (val, 23)):
        if target.segment_hypotheses.shape != (expected, 2, 6):
            raise ValueError("unexpected continuous segment-target shape")
        if np.any(np.abs(target.segment_hypotheses[..., 3:5]) > 1.0 + 1e-6):
            raise ValueError("segment signed depth lies outside [-1,1]")
        if np.any(
            (target.segment_hypotheses[..., 5] < 0)
            | (target.segment_hypotheses[..., 5] > 1)
        ):
            raise ValueError("segment synchrony lies outside [0,1]")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("condition_variant") != (
        "geo_history_actual_jstd_event_hypothesis_h1"
    ):
        raise ValueError("MSEP must initialize from the completed H1 renderer")
    static = load_station_static_data(data_path)
    primary, secondary, _ = load_generation_graphs(
        data_path,
        Path(args.checkpoint).parent.parent,
        model_config,
        checkpoint,
    )
    model = Station24DiffusionModel(
        model_config,
        static["station_features"],
        primary,
        static["station_capacities"],
        secondary,
    ).to(device)
    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    if set(incompatible.missing_keys) != set(
        model.jstd_segment_prior_state_dict_keys
    ):
        raise ValueError(
            "unexpected MSEP initialization gaps: "
            f"{sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected MSEP initialization keys: {incompatible.unexpected_keys}"
        )
    trainable = model.configure_jstd_training()
    if any("issue_head" in name for name in trainable):
        raise ValueError("failed issue head must remain frozen")
    if set(trainable) != set(model.jstd_segment_prior_trainable_parameter_names):
        raise ValueError("MSEP V2 optimizer must contain only the segment prior")

    _, dataset = get_station_dataloader(
        data_path,
        "train",
        checkpoint["residual_scale"],
        batch_size=3,
        seed=2027,
        num_workers=0,
        condition_config=model_config,
        state_thresholds=checkpoint.get("state_thresholds"),
        jstd_targets=train,
    )
    counts = train.segment_hypotheses[..., 0].sum(axis=1).astype(int)
    selected = []
    for desired in (0, 1, 2):
        candidates = np.flatnonzero(counts == desired)
        if candidates.size == 0:
            raise ValueError(f"training data lacks a {desired}-event preflight case")
        selected.append(int(candidates[0]))
    batch = next(iter(DataLoader(Subset(dataset, selected), batch_size=3)))
    batch = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    frozen_before = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("denoiser.jstd_tail.segment_prior.")
        and not name.startswith("diffusion.denoiser.jstd_tail.segment_prior.")
    }
    loss, parts = model.denoiser.jstd_tail.segment_prior_loss(
        batch["forecast"],
        batch["jstd_segment_hypotheses"],
        batch["jstd_sample_weight"],
        recent_error=batch["recent_error"],
        recent_error_mask=batch["recent_error_mask"],
    )
    if not torch.isfinite(loss):
        raise ValueError("MSEP structured prior loss is not finite")
    initial_loss = float(loss.detach())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5.0e-4,
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    scale_gradient_by_channel = None
    changed_groups: set[str] = set()
    trainable_before = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    model.train()
    for optimization_step in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            model_loss = model(batch)
        if not torch.isfinite(model_loss):
            raise ValueError("MSEP prior-only objective is not finite")
        scaler.scale(model_loss).backward()
        scaler.unscale_(optimizer)
        if optimization_step == 0:
            attribute_bias = model.denoiser.jstd_tail.segment_prior.attribute_head.bias
            scale_gradient_by_channel = {
                str(index): float(attribute_bias.grad[index].detach().cpu())
                for index in (1, 3, 5, 8, 10, 12)
            }
            if any(
                not np.isfinite(value) or abs(value) <= 1.0e-10
                for value in scale_gradient_by_channel.values()
            ):
                raise ValueError(
                    "one or more duration/depth scale heads have zero gradients"
                )
        scaler.step(optimizer)
        scaler.update()
    final_loss = float(model(batch).detach())
    if not final_loss < initial_loss:
        raise ValueError(
            f"fixed-batch prior did not improve: {initial_loss} -> {final_loss}"
        )
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if not torch.equal(trainable_before[name], value.detach().cpu()):
            if ".count_head." in name:
                changed_groups.add("count")
            elif ".onset_head." in name:
                changed_groups.add("onset")
            elif ".attribute_head." in name:
                changed_groups.add("attributes")
            elif ".encoder." in name:
                changed_groups.add("encoder")
    required_groups = {"count", "onset", "attributes", "encoder"}
    if changed_groups != required_groups:
        raise ValueError(
            f"not every prior group updated; changed={sorted(changed_groups)}"
        )
    frozen_after = model.state_dict()
    changed_frozen = [
        name
        for name, before in frozen_before.items()
        if not torch.equal(before, frozen_after[name].detach().cpu())
    ]
    if changed_frozen:
        raise ValueError(f"frozen state changed: {changed_frozen[:10]}")

    serialized = io.BytesIO()
    torch.save(model.state_dict(), serialized)
    serialized.seek(0)
    reloaded = Station24DiffusionModel(
        model_config,
        static["station_features"],
        primary,
        static["station_capacities"],
        secondary,
    ).to(device)
    reloaded.load_state_dict(
        torch.load(serialized, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        original_prior = model.denoiser.jstd_tail.segment_prior_output(
            batch["forecast"],
            batch["recent_error"],
            batch["recent_error_mask"],
        )
        reloaded_prior = reloaded.denoiser.jstd_tail.segment_prior_output(
            batch["forecast"],
            batch["recent_error"],
            batch["recent_error_mask"],
        )
    for field in ("count_logits", "onset_logits", "attribute_raw"):
        original = getattr(original_prior, field)
        restored = getattr(reloaded_prior, field)
        if not torch.equal(original, restored):
            raise ValueError(f"save/reload changed segment-prior {field}")

    with torch.no_grad():
        renderer_loss = model(
            batch,
            timestep=torch.tensor([3, 101, 377], dtype=torch.long, device=device),
            noise=torch.zeros_like(batch["residual_target"]),
            include_auxiliary=False,
        )
    if not torch.isfinite(renderer_loss):
        raise ValueError("frozen H1 renderer smoke objective is not finite")
    torch.manual_seed(17)
    hypotheses, sampled = model.denoiser.jstd_tail.sample_segment_hypotheses(
        batch["forecast"],
        512,
        recent_error=batch["recent_error"],
        recent_error_mask=batch["recent_error_mask"],
    )
    if hypotheses.shape != (3, 512, 2, 6):
        raise ValueError("member-specific segment sampling shape is invalid")
    active_hypotheses = hypotheses[..., 0] > 0.5
    active_duration = hypotheses[..., 2][active_hypotheses]
    active_depth = hypotheses[..., 3:5][active_hypotheses]
    if torch.any((active_duration <= 0.0) | (active_duration >= 1.0)):
        raise ValueError("transformed duration sampling produced a boundary atom")
    if torch.any(active_depth.abs() >= 1.0):
        raise ValueError("transformed depth sampling produced a boundary atom")
    original_steps = model.diffusion.num_steps
    model.diffusion.num_steps = 2
    try:
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            generated, generation_audit = model.generate(
                batch, n_samples=3, return_expert_audit=True
            )
    finally:
        model.diffusion.num_steps = original_steps
    if generated.shape != (3, 3, 24, 168):
        raise ValueError("MSEP integrated member generation shape is invalid")
    if generation_audit["jstd_segment_hypotheses"].shape != (3, 3, 2, 6):
        raise ValueError("MSEP generated hypothesis audit shape is invalid")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        "method": "jstd_msep_causal_segment_prior_preflight_v2",
        "source_checkpoint": str(args.checkpoint),
        "source_variant": checkpoint["condition_variant"],
        "raw_body_frozen": True,
        "h1_renderer_reused": True,
        "h1_renderer_frozen_and_unchanged": True,
        "training_mode": "segment_prior_only",
        "failed_issue_gate_used": False,
        "validation_actual_used_as_generation_condition": False,
        "generation_condition_sources": [
            "current_issued_forecast",
            "previous_issue_recent_observed_error_when_available",
            "fixed_train_only_graph",
        ],
        "segment_fields": [
            "active",
            "onset_fraction",
            "actual_duration_fraction",
            "signed_wind_depth",
            "signed_solar_depth",
            "source_synchrony",
        ],
        "train_segment_distribution": train.audit[
            "segment_event_count_distribution"
        ],
        "val_segment_distribution_for_labels_only": val.audit[
            "segment_event_count_distribution"
        ],
        "initial_structured_loss": float(loss.detach()),
        "device": str(device),
        "cuda_amp_checked": device.type == "cuda",
        "initial_prior_loss": initial_loss,
        "fixed_batch_prior_loss_after_10_updates": final_loss,
        "frozen_renderer_smoke_loss": float(renderer_loss.detach()),
        "initial_loss_parts": {
            key: float(value) for key, value in parts.items()
        },
        "preflight_sampled_tail_fraction": float(
            (sampled["counts"] > 0).float().mean()
        ),
        "scale_parameterization": (
            model.denoiser.jstd_tail.segment_scale_parameterization
        ),
        "scale_gradient_by_attribute_channel": scale_gradient_by_channel,
        "updated_prior_groups": sorted(changed_groups),
        "boundary_atom_check": "PASS",
        "integrated_generation_smoke_test": True,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "trainable_parameter_names": list(trainable),
        "reportable_as_causal_forecast": True,
        "test_generation_allowed_before_model_lock": False,
        "dataset_condition_audit": dataset.condition_audit,
        "checks": {
            "checkpoint_contract": "PASS",
            "causal_condition_contract": "PASS",
            "all_scale_gradients": "PASS",
            "all_prior_groups_update": "PASS",
            "frozen_renderer_unchanged": "PASS",
            "fixed_batch_learning": "PASS",
            "save_reload": "PASS",
            "final_500_member_branch_ablation": "NOT RUN",
            "cuda_amp": "PASS" if device.type == "cuda" else "NOT RUN",
        },
        "launch_eligible_on_this_environment": device.type == "cuda",
    }
    (output / "jstd_msep_preflight.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
