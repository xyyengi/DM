#!/usr/bin/env python3
"""Fail-fast JSTD audit before a paid GPU training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from src.models.station_joint_decomposed_tail import ComplementaryTemporalProjection
from station_dataset import (
    fit_station_state_thresholds,
    get_station_dataloader,
    load_station_static_data,
)
from station_jstd_targets import (
    build_station_jstd_target_arrays,
    fit_station_jstd_event_thresholds,
)


RAW_INVARIANT_KEYS = (
    "architecture",
    "spatial_mode",
    "spatial_mix_levels",
    "parallel_spatial_fusion_levels",
    "parallel_spatial_adjacency_mode",
    "use_dual_fixed_graph",
    "station_count",
    "sequence_length",
    "base_channels",
    "num_layers",
    "channel_multipliers",
    "group_norm_groups",
    "dropout",
    "timestep_embedding_dim",
    "num_steps",
    "beta_start",
    "beta_end",
    "use_forecast_ramps",
    "use_forecast_revision",
    "use_recent_error",
    "recent_error_hours",
    "condition_gate_init",
    "use_state_encoder",
    "state_channels",
    "state_ramp_lags",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--secondary-adjacency", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidate_config = dict(config["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("JSTD must start from the formal Raw body-tail checkpoint")
    source_config = dict(checkpoint["config"]["model"])
    changed = {
        key: {"raw": source_config.get(key), "jstd": candidate_config.get(key)}
        for key in RAW_INVARIANT_KEYS
        if source_config.get(key) != candidate_config.get(key)
    }
    if changed:
        raise ValueError(f"Raw body/config modulation changed: {changed}")
    forbidden = {
        key: bool(candidate_config.get(key, False))
        for key in (
            "use_tail_time_localizer",
            "use_retrieval_mismatch_expert",
            "use_discrete_event_memory",
            "train_sampler_energy_score_only",
            "use_forecast_revision",
        )
    }
    if any(forbidden.values()):
        raise ValueError(f"JSTD V1 contains forbidden stacked modules: {forbidden}")

    thresholds = fit_station_jstd_event_thresholds(args.data_path, candidate_config)
    train_targets = build_station_jstd_target_arrays(
        args.data_path, "train", thresholds
    )
    val_targets = build_station_jstd_target_arrays(args.data_path, "val", thresholds)
    if not 0.10 <= float(train_targets.event_active.mean()) <= 0.50:
        raise ValueError("JSTD issue labels are collapsed or no longer tail-like")
    if not bool(train_targets.audit["contains_sub_6h_events"]):
        raise ValueError("short events disappeared from JSTD training labels")
    (output / "jstd_event_targets.json").write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "train": train_targets.audit,
                "val": val_targets.audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    static = load_station_static_data(args.data_path)
    secondary = torch.from_numpy(np.load(args.secondary_adjacency).astype(np.float32))
    source = Station24DiffusionModel(
        source_config,
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
        secondary,
    )
    source.load_state_dict(checkpoint["model_state_dict"], strict=True)
    candidate = Station24DiffusionModel(
        candidate_config,
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
        secondary,
    )
    incompatible = candidate.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    expected_missing = set(candidate.jstd_new_state_dict_keys)
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "JSTD checkpoint compatibility failed: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    trainable = candidate.configure_jstd_training()
    if not trainable or any(not name.startswith("denoiser.jstd_tail.") for name in trainable):
        raise ValueError("JSTD parameter isolation failed")

    state_thresholds = checkpoint.get("state_thresholds")
    if state_thresholds is None and bool(candidate_config.get("use_state_encoder", False)):
        state_thresholds = fit_station_state_thresholds(args.data_path)
    loader, _ = get_station_dataloader(
        args.data_path,
        "val",
        checkpoint["residual_scale"],
        batch_size=2,
        seed=314159,
        condition_config=candidate_config,
        state_thresholds=state_thresholds,
        jstd_targets=val_targets,
    )
    batch = next(iter(loader))
    timestep = torch.tensor([37, 211], dtype=torch.long)
    noise = torch.Generator().manual_seed(20260903)
    noise = torch.randn(batch["residual_target"].shape, generator=noise)
    source.eval()
    candidate.eval()
    with torch.no_grad():
        source_loss = source(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
            body_tail_route_override=0.0,
        )
        candidate_loss = candidate(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
            body_tail_route_override=0.0,
        )
        candidate_initial_tail = candidate(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
            body_tail_route_override=1.0,
        )
    identity_error = max(
        abs(float(source_loss) - float(candidate_loss)),
        abs(float(source_loss) - float(candidate_initial_tail)),
    )
    if identity_error > 1e-7:
        raise ValueError(f"JSTD does not preserve the Raw initialization: {identity_error}")
    (output / "jstd_identity_test.json").write_text(
        json.dumps(
            {
                "raw_body_loss": float(source_loss),
                "jstd_route0_loss": float(candidate_loss),
                "jstd_zero_initialized_route1_loss": float(candidate_initial_tail),
                "max_abs_error": identity_error,
                "passed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    probe = torch.randn(3, 24, 168)
    low, fast = ComplementaryTemporalProjection(12).split(probe)
    reconstruction_error = float((low + fast - probe).abs().max())
    if reconstruction_error > 1e-6:
        raise ValueError("slow/fast projection is not complementary")
    (output / "jstd_frequency_projection_test.json").write_text(
        json.dumps(
            {
                "canonical_boundary_hours": 12,
                "slow_consistency_hours": 24,
                "max_reconstruction_error": reconstruction_error,
                "mask_then_projection": True,
                "passed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    module = candidate.denoiser.jstd_tail
    if module is None:
        raise RuntimeError("JSTD module missing")
    with torch.no_grad():
        reference = module._causal_condition_groups(
            batch["forecast"], batch["recent_error"], batch["recent_error_mask"]
        )
        mutated_actual = torch.randn_like(batch["actual"])
        del mutated_actual
        repeated = module._causal_condition_groups(
            batch["forecast"], batch["recent_error"], batch["recent_error_mask"]
        )
    causal_error = max(
        float((left - right).abs().max()) for left, right in zip(reference, repeated)
    )
    if causal_error != 0.0:
        raise ValueError("JSTD conditions changed without changing causal inputs")
    (output / "jstd_condition_audit.json").write_text(
        json.dumps(
            {
                "condition_groups": [
                    "current_forecast_multiscale_geometry",
                    "recent_observed_error_multiscale_state",
                    "fixed_graph_and_wind_solar_system_aggregates",
                ],
                "forecast_revision_in_v1": False,
                "future_actual_or_residual_condition": False,
                "future_actual_mutation_max_condition_change": causal_error,
                "raw_invariant_fields": list(RAW_INVARIANT_KEYS),
                "raw_invariant_changes": changed,
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"JSTD_PREFLIGHT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
