#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate existing generation runs with overall, extreme, and tail metrics.

This script does not train models. It first tries to load saved arrays from a
run directory. If arrays are missing, it can regenerate scenarios from the run
checkpoint and config.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def apply_experiment_switches(config: dict) -> dict:
    """Copy top-level target/condition/guidance switches into model config."""
    model_cfg = config.setdefault("model", {})
    target_cfg = config.get("target", {})
    condition_cfg = config.get("condition", {})
    guidance_cfg = config.get("guidance", {})
    sampling_cfg = config.get("sampling", {})

    if "type" in target_cfg:
        model_cfg["target_type"] = target_cfg["type"]
    if "mode" in condition_cfg:
        model_cfg["condition_mode"] = condition_cfg["mode"]
    if "use_forecast" in condition_cfg:
        model_cfg["use_forecast"] = condition_cfg["use_forecast"]
    if "use_network_condition" in condition_cfg:
        model_cfg["use_network_condition"] = condition_cfg["use_network_condition"]
    if "use_guidance" in condition_cfg:
        model_cfg["use_guidance"] = condition_cfg["use_guidance"]
    if "cond_mask" in condition_cfg:
        model_cfg["cond_mask"] = condition_cfg["cond_mask"]
    if "enable" in guidance_cfg:
        model_cfg["use_guidance"] = guidance_cfg["enable"]
    if {"wind_scale", "pv_scale", "load_scale"} <= set(guidance_cfg):
        model_cfg["guidance_scales"] = [
            guidance_cfg["wind_scale"],
            guidance_cfg["pv_scale"],
            guidance_cfg["load_scale"],
        ]
        model_cfg["guidance_scale"] = max(model_cfg["guidance_scales"])
    if "input_channels" in model_cfg:
        model_cfg["in_channels"] = model_cfg["input_channels"]
    if "reverse_variance_type" in sampling_cfg:
        model_cfg["reverse_variance_type"] = sampling_cfg["reverse_variance_type"]
    return config
from src.eval.extreme_metrics import (  # noqa: E402
    ScenarioArrays,
    apply_extreme_flags,
    evaluate_rows,
    fit_extreme_thresholds,
    load_actual_from_split,
    rank_histogram_rows,
    save_flags,
    save_thresholds,
    tail_rows,
    write_csv,
)


def load_config(run_dir: str, fallback_config: Optional[str]) -> Tuple[dict, str]:
    config_path = os.path.join(run_dir, "config_used.yaml")
    if os.path.exists(config_path):
        path = config_path
    elif fallback_config:
        path = fallback_config
    else:
        raise FileNotFoundError(f"No config_used.yaml in {run_dir}; pass --config")
    with open(path, "r", encoding="utf-8") as f:
        return apply_experiment_switches(yaml.safe_load(f) or {}), path


def load_saved_arrays(run_dir: str) -> Optional[ScenarioArrays]:
    actual_samples_path = os.path.join(run_dir, "actual_scenarios.npy")
    actual_path = os.path.join(run_dir, "actual_data.npy")
    forecast_path = os.path.join(run_dir, "forecast_data.npy")
    if not os.path.exists(actual_samples_path) or not os.path.exists(actual_path):
        return None
    samples = np.load(actual_samples_path).astype(np.float64)
    actual = np.load(actual_path).astype(np.float64)
    forecast = np.load(forecast_path).astype(np.float64) if os.path.exists(forecast_path) else None
    return ScenarioArrays(samples=samples, actual=actual, forecast=forecast)


def regenerate_arrays(
    run_dir: str,
    config: dict,
    data_path: str,
    n_samples: int,
    max_batches: Optional[int],
    batch_size: Optional[int],
) -> ScenarioArrays:
    import torch

    from dataset_multivariate import get_dataloader_multivariate
    from diff_models_multivariate import MultiChannelCSDI
    from generate import (
        denormalize_channels,
        generate_scenarios,
        get_checkpoint_path,
        load_denormalization_scales,
        model_output_to_actual,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = get_checkpoint_path(run_dir, "best")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    build_kde = bool(config["model"].get("use_guidance", False))
    eval_batch_size = batch_size or min(int(config.get("train", {}).get("batch_size", 16)), 16)
    test_loader, _, max_values = get_dataloader_multivariate(
        data_path,
        eval_batch_size,
        "test",
        config["model"]["n_intervals"],
        build_kde=build_kde,
        residual_standardization=config.get("target", {}).get(
            "residual_standardization", {"enabled": False}
        ),
    )

    model = MultiChannelCSDI(config["model"], device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    raw_samples, forecast, _residual, actual = generate_scenarios(
        model,
        test_loader,
        device,
        n_samples=n_samples,
        max_batches=max_batches,
    )
    actual_samples = model_output_to_actual(
        raw_samples,
        forecast,
        config["model"].get("target_type", "residual"),
        test_loader.dataset.residual_standardizer,
    )
    scales, _ = load_denormalization_scales(data_path, max_values)
    return ScenarioArrays(
        samples=denormalize_channels(actual_samples, scales).astype(np.float64),
        actual=denormalize_channels(actual, scales).astype(np.float64),
        forecast=denormalize_channels(forecast, scales).astype(np.float64),
    )


def limit_windows(arrays: ScenarioArrays, max_samples: Optional[int]) -> ScenarioArrays:
    if max_samples is None or arrays.samples.shape[0] <= max_samples:
        return arrays
    return ScenarioArrays(
        samples=arrays.samples[:max_samples],
        actual=arrays.actual[:max_samples],
        forecast=arrays.forecast[:max_samples] if arrays.forecast is not None else None,
    )


def output_dir_for(run_dir: str, output_root: str) -> str:
    run_id = os.path.basename(os.path.abspath(run_dir))
    return os.path.join(output_root, run_id)


def evaluate_run(
    run_dir: str,
    data_path: str,
    output_root: str,
    fallback_config: Optional[str],
    n_samples: int,
    max_batches: Optional[int],
    max_samples: Optional[int],
    batch_size: Optional[int],
) -> str:
    run_id = os.path.basename(os.path.abspath(run_dir))
    config, _config_path = load_config(run_dir, fallback_config)

    arrays = load_saved_arrays(run_dir)
    if arrays is None:
        print(f"[{run_id}] saved arrays not found; regenerating from checkpoint.")
        arrays = regenerate_arrays(run_dir, config, data_path, n_samples, max_batches, batch_size)
    else:
        print(f"[{run_id}] loaded saved arrays.")

    arrays = limit_windows(arrays, max_samples)
    train_actual = load_actual_from_split(data_path, "train")
    test_actual_for_flags = load_actual_from_split(data_path, "test")[:arrays.actual.shape[0]]
    thresholds = fit_extreme_thresholds(train_actual)
    flags = apply_extreme_flags(test_actual_for_flags, thresholds)

    out_dir = output_dir_for(run_dir, output_root)
    os.makedirs(out_dir, exist_ok=True)

    overall = evaluate_rows(arrays, run_id, subset="overall")
    subset_rows = []
    for name, mask in flags.items():
        subset_rows.extend(evaluate_rows(arrays, run_id, subset=name, mask=mask))

    write_csv(os.path.join(out_dir, "metrics_overall.csv"), overall)
    write_csv(os.path.join(out_dir, "metrics_extreme_subsets.csv"), subset_rows)
    write_csv(os.path.join(out_dir, "metrics_tail.csv"), tail_rows(arrays, thresholds, run_id, reference_actual=test_actual_for_flags))
    write_csv(os.path.join(out_dir, "rank_histogram.csv"), rank_histogram_rows(arrays, run_id))
    save_flags(flags, os.path.join(out_dir, "extreme_flags_test.csv"))
    save_thresholds(thresholds, os.path.join(out_dir, "extreme_thresholds_train.json"))

    print(f"[{run_id}] advanced evaluation saved to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Advanced evaluation for existing v-mix/V2 runs.")
    parser.add_argument("--run", action="append", required=True, help="Run directory. Can be passed multiple times.")
    parser.add_argument("--data_path", default="input_4.27")
    parser.add_argument("--output_root", default="outputs/advanced_eval")
    parser.add_argument("--config", default=None, help="Fallback config if run/config_used.yaml is missing.")
    parser.add_argument("--n_samples", type=int, default=10, help="Samples to regenerate when saved arrays are missing.")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit regenerated test batches for smoke tests.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit windows after loading/regeneration for fast evaluation.")
    parser.add_argument("--batch_size", type=int, default=None, help="Generation batch size.")
    args = parser.parse_args()

    for run_dir in args.run:
        evaluate_run(
            run_dir=run_dir,
            data_path=args.data_path,
            output_root=args.output_root,
            fallback_config=args.config,
            n_samples=args.n_samples,
            max_batches=args.max_batches,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()


