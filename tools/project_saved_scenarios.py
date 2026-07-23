#!/usr/bin/env python
"""Add physically constrained artifacts to saved scenario result directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import evaluate_multichannel
from generate import add_basic_mae, save_scenarios_npz
from src.eval.experiment_logger import json_safe
from src.eval.physical_projection import (
    daylight_mask_from_export_metadata,
    daylight_mask_from_train_support,
    project_power_scenarios,
)


OUTPUT_NAMES = (
    "actual_scenarios_constrained.npy",
    "actual_scenarios_constrained_normalized.npy",
    "metrics_constrained.json",
    "physical_projection.json",
)


def project_result(
    result_dir: Path,
    solar_night_threshold_mw: float = 1.0,
    solar_night_mode: str = "train_support",
    data_path: Path | None = None,
    overwrite: bool = False,
) -> dict:
    required = (
        "actual_scenarios.npy",
        "actual_data.npy",
        "forecast_data.npy",
        "denormalization_used.json",
        "metrics.json",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{result_dir}: missing {missing}")
    existing = [name for name in OUTPUT_NAMES if (result_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{result_dir}: refusing to overwrite existing projection files {existing}"
        )

    raw = np.load(result_dir / "actual_scenarios.npy")
    actual = np.load(result_dir / "actual_data.npy")
    forecast = np.load(result_dir / "forecast_data.npy")
    with (result_dir / "denormalization_used.json").open(
        "r", encoding="utf-8"
    ) as handle:
        denormalization = json.load(handle)
    scales = np.asarray(denormalization["scales"], dtype=np.float64)
    with (result_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        raw_metrics = json.load(handle)

    solar_daylight_mask = None
    solar_daylight_metadata = None
    if solar_night_mode in {"train_support", "astronomical_shandong"}:
        config_path = result_dir / "generation_config_used.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"{result_dir}: missing generation_config_used.yaml")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        resolved_data_path = data_path
        if resolved_data_path is None:
            configured = Path(config.get("data", {}).get("data_path", ""))
            resolved_data_path = (
                configured if configured.is_absolute() else REPO_ROOT / configured
            )
        data_split = str(raw_metrics.get(
            "data_split",
            config.get("evaluation", {}).get("evaluated_split", "val"),
        ))
        daylight_builder = (
            daylight_mask_from_train_support
            if solar_night_mode == "train_support"
            else daylight_mask_from_export_metadata
        )
        builder_path = (
            resolved_data_path
            if solar_night_mode == "train_support"
            else resolved_data_path / "export_metadata.json"
        )
        solar_daylight_mask, solar_daylight_metadata = (
            daylight_builder(
                builder_path,
                data_split=data_split,
                window_count=raw.shape[0],
                sequence_length=raw.shape[3],
            )
        )
    elif solar_night_mode != "forecast_threshold":
        raise ValueError(
            "solar_night_mode must be 'train_support', "
            "'astronomical_shandong', or 'forecast_threshold'"
        )
    constrained, report = project_power_scenarios(
        raw,
        forecast,
        scales,
        solar_night_threshold_mw=solar_night_threshold_mw,
        solar_daylight_mask=solar_daylight_mask,
        solar_daylight_metadata=solar_daylight_metadata,
    )
    constrained_normalized = (
        constrained / scales.reshape(1, 1, 3, 1)
    ).astype(constrained.dtype, copy=False)

    quantiles = [
        int(key.split("_")[-1].rstrip("%")) / 100.0
        for key in raw_metrics
        if key.startswith("total_coverage_")
    ]
    quantiles = sorted(set(quantiles), reverse=True) or [0.95, 0.9, 0.8]
    metrics = evaluate_multichannel(
        constrained, actual, quantiles=quantiles, verbose=False
    )
    metrics = add_basic_mae(metrics, constrained, actual)
    for key in (
        "run_id", "timestamp", "checkpoint_path", "figure_dir",
        "scenario_shape", "reverse_variance_type",
        "residual_standardization_enabled", "data_split", "condition_ablation",
    ):
        if key in raw_metrics:
            metrics[key] = raw_metrics[key]
    metrics["scenario_profile"] = "physical_projection"
    metrics["physical_projection"] = report

    np.save(result_dir / "actual_scenarios_constrained.npy", constrained)
    np.save(
        result_dir / "actual_scenarios_constrained_normalized.npy",
        constrained_normalized,
    )
    scenario_path = save_scenarios_npz(
        constrained,
        forecast,
        str(result_dir),
        filename="scenarios_constrained.npz",
    )
    metrics["scenario_path"] = scenario_path
    with (result_dir / "metrics_constrained.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(metrics), handle, ensure_ascii=False, indent=2)
    with (result_dir / "physical_projection.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--solar-night-mode",
        choices=("train_support", "astronomical_shandong", "forecast_threshold"),
        default="train_support",
    )
    parser.add_argument("--solar-night-threshold-mw", type=float, default=1.0)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for result_dir in args.result_dirs:
        report = project_result(
            result_dir.resolve(),
            solar_night_threshold_mw=args.solar_night_threshold_mw,
            solar_night_mode=args.solar_night_mode,
            data_path=args.data_path,
            overwrite=args.overwrite,
        )
        print(
            "PROJECTED "
            f"result={result_dir} "
            f"raw_violation={report['raw_boundary_rates']['any_physical_violation_pct']:.6f} "
            f"projected_violation={report['projected_boundary_rates']['any_physical_violation_pct']:.6f}"
        )


if __name__ == "__main__":
    main()
