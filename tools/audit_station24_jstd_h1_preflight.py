#!/usr/bin/env python3
"""Fail-fast audit for the validation-oracle JSTD H1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data
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
    if not model_config.get("use_jstd_event_hypothesis", False):
        raise ValueError("H1 config must enable continuous event hypotheses")
    if float(model_config.get("tail_gate_loss_weight", -1)) != 0.0:
        raise ValueError("H1 must not optimize the failed issue gate")
    if float(model_config.get("jstd_issue_loss_weight", -1)) != 0.0:
        raise ValueError("H1 must not optimize the failed JSTD issue head")

    data_path = Path(args.data_path)
    thresholds = fit_station_jstd_event_thresholds(data_path, model_config)
    train = build_station_jstd_target_arrays(data_path, "train", thresholds)
    val = build_station_jstd_target_arrays(data_path, "val", thresholds)
    if train.event_hypothesis.shape != (290, 6):
        raise ValueError("unexpected train H1 hypothesis shape")
    if val.event_hypothesis.shape != (23, 6):
        raise ValueError("unexpected validation H1 hypothesis shape")
    if not np.all((train.event_hypothesis[:, 0] == 0) | (train.event_hypothesis[:, 0] == 1)):
        raise ValueError("H1 active indicator must be binary")
    if np.any(np.abs(train.event_hypothesis[:, 3:5]) > 1.0 + 1e-6):
        raise ValueError("H1 signed depth must be normalized to [-1,1]")
    if np.any((train.event_hypothesis[:, 5] < 0) | (train.event_hypothesis[:, 5] > 1)):
        raise ValueError("H1 synchrony must be in [0,1]")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("condition_variant") != "geo_history_actual_jstd_tail_v1":
        raise ValueError("H1 must initialize from the completed JSTD V1 checkpoint")
    source_model_config = dict(checkpoint["config"]["model"])
    allowed_model_changes = {
        "use_jstd_event_hypothesis",
        "jstd_h1_tail_fraction",
        "jstd_hypothesis_edge_temperature_hours",
        "tail_gate_loss_weight",
        "jstd_issue_loss_weight",
        "jstd_outside_zero_loss_weight",
    }
    ignored_runtime_fields = {"secondary_adjacency_path"}
    changed_fields = sorted(
        key
        for key in set(source_model_config) | set(model_config)
        if key not in allowed_model_changes | ignored_runtime_fields
        and source_model_config.get(key) != model_config.get(key)
    )
    if changed_fields:
        raise ValueError(
            "H1 changed fields outside the event-hypothesis mechanism: "
            f"{changed_fields}"
        )
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
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if set(incompatible.missing_keys) != set(model.jstd_hypothesis_state_dict_keys):
        raise ValueError(
            f"unexpected H1 initialization gaps: {sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected H1 initialization keys: {incompatible.unexpected_keys}"
        )
    trainable = model.configure_jstd_training()
    if any("issue_head" in name for name in trainable):
        raise ValueError("failed issue gate must remain frozen in H1")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    audit = {
        "method": "jstd_continuous_event_hypothesis_h1_preflight",
        "source_checkpoint": str(args.checkpoint),
        "source_variant": checkpoint["condition_variant"],
        "raw_body_frozen": True,
        "issue_gate_used": False,
        "hypothesis_dimension": 6,
        "hypothesis_fields": train.audit["h1_event_hypothesis_fields"],
        "allowed_model_changes": sorted(allowed_model_changes),
        "unintended_model_changes": changed_fields,
        "train_event_issue_count": int(train.event_hypothesis[:, 0].sum()),
        "val_oracle_event_issue_count": int(val.event_hypothesis[:, 0].sum()),
        "validation_actual_used_as_generation_condition": True,
        "test_generation_allowed": False,
        "reportable_as_causal_forecast": False,
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "trainable_parameter_names": list(trainable),
    }
    (output / "jstd_h1_preflight.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
