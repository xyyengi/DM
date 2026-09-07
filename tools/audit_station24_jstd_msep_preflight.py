#!/usr/bin/env python3
"""Fail-fast structural and causality audit for Station-24 JSTD-MSEP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

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
        raise ValueError("the formal MSEP V1 protocol carries two ordered slots")

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
    )
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

    loader, dataset = get_station_dataloader(
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
    batch = next(iter(loader))
    loss, parts = model.denoiser.jstd_tail.segment_prior_loss(
        batch["forecast"],
        batch["jstd_segment_hypotheses"],
        batch["jstd_sample_weight"],
        recent_error=batch["recent_error"],
        recent_error_mask=batch["recent_error_mask"],
    )
    if not torch.isfinite(loss):
        raise ValueError("MSEP structured prior loss is not finite")
    model_loss = model(
        batch,
        timestep=torch.tensor([3, 101, 377], dtype=torch.long),
        noise=torch.zeros_like(batch["residual_target"]),
    )
    if not torch.isfinite(model_loss):
        raise ValueError("MSEP integrated diffusion objective is not finite")
    torch.manual_seed(17)
    hypotheses, sampled = model.denoiser.jstd_tail.sample_segment_hypotheses(
        batch["forecast"],
        32,
        recent_error=batch["recent_error"],
        recent_error_mask=batch["recent_error_mask"],
    )
    if hypotheses.shape != (3, 32, 2, 6):
        raise ValueError("member-specific segment sampling shape is invalid")
    original_steps = model.diffusion.num_steps
    model.diffusion.num_steps = 2
    try:
        with torch.no_grad():
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
        "method": "jstd_msep_causal_segment_prior_preflight_v1",
        "source_checkpoint": str(args.checkpoint),
        "source_variant": checkpoint["condition_variant"],
        "raw_body_frozen": True,
        "h1_renderer_reused": True,
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
        "initial_integrated_diffusion_loss": float(model_loss.detach()),
        "initial_loss_parts": {
            key: float(value) for key, value in parts.items()
        },
        "initial_sampled_tail_fraction": float(
            (sampled["counts"] > 0).float().mean()
        ),
        "integrated_generation_smoke_test": True,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "trainable_parameter_names": list(trainable),
        "reportable_as_causal_forecast": True,
        "test_generation_allowed_before_model_lock": False,
        "dataset_condition_audit": dataset.condition_audit,
    }
    (output / "jstd_msep_preflight.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
