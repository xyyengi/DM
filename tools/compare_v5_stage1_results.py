#!/usr/bin/env python
"""Compare fixed-protocol V4-RS, V5-T, and V5-TF validation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_model


ARCHITECTURE_ORDER = {"v4_legacy": 0, "v5_t": 1, "v5_tf": 2}


def _percentage(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) * 100.0 / mask.size)


def _ramp_mae(generated: np.ndarray, actual: np.ndarray, horizon: int) -> float:
    generated_ramp = generated[..., horizon:] - generated[..., :-horizon]
    actual_ramp = actual[..., horizon:] - actual[..., :-horizon]
    return float(np.mean(np.abs(generated_ramp - actual_ramp)))


def _correlation_matrix(values: np.ndarray) -> np.ndarray:
    flattened = np.moveaxis(values, 1, 0).reshape(3, -1)
    return np.corrcoef(flattened)


def compute_saved_diagnostics(
    result_dir: Path,
    scenarios_filename: str = "actual_scenarios.npy",
) -> dict[str, float]:
    scenarios = np.load(result_dir / scenarios_filename, mmap_mode="r")
    actual = np.load(result_dir / "actual_data.npy", mmap_mode="r")
    if scenarios.ndim != 4 or scenarios.shape[2] != 3:
        raise ValueError(f"actual_scenarios must be [N,S,3,L], got {scenarios.shape}")
    if actual.shape != (scenarios.shape[0], 3, scenarios.shape[3]):
        raise ValueError(
            f"actual_data shape {actual.shape} is incompatible with {scenarios.shape}"
        )
    with (result_dir / "denormalization_used.json").open("r", encoding="utf-8") as handle:
        denormalization = json.load(handle)
    scales = np.asarray(denormalization["scales"], dtype=np.float64)
    if scales.shape != (3,) or np.any(scales <= 0):
        raise ValueError(f"invalid physical scales: {scales}")

    scenario_mean = np.mean(scenarios, axis=1, dtype=np.float64)
    actual_float = np.asarray(actual, dtype=np.float64)
    generated_net_load = scenario_mean[:, 2] - scenario_mean[:, 0] - scenario_mean[:, 1]
    actual_net_load = actual_float[:, 2] - actual_float[:, 0] - actual_float[:, 1]
    generated_corr = _correlation_matrix(scenario_mean)
    actual_corr = _correlation_matrix(actual_float)
    off_diagonal = ~np.eye(3, dtype=bool)

    wind = scenarios[:, :, 0, :]
    solar = scenarios[:, :, 1, :]
    load = scenarios[:, :, 2, :]
    physical_invalid = (
        (wind < 0.0)
        | (wind > scales[0])
        | (solar < 0.0)
        | (solar > scales[1])
        | (load < 0.0)
    )
    return {
        "wind_below_zero_pct": _percentage(wind < 0.0),
        "wind_above_capacity_pct": _percentage(wind > scales[0]),
        "solar_below_zero_pct": _percentage(solar < 0.0),
        "solar_above_capacity_pct": _percentage(solar > scales[1]),
        "load_below_zero_pct": _percentage(load < 0.0),
        "load_above_train_denominator_pct": _percentage(load > scales[2]),
        "any_physical_violation_pct": _percentage(physical_invalid),
        "net_load_mae_mw": float(np.mean(np.abs(generated_net_load - actual_net_load))),
        "net_load_ramp_1h_mae_mw": _ramp_mae(
            generated_net_load, actual_net_load, 1
        ),
        "net_load_ramp_6h_mae_mw": _ramp_mae(
            generated_net_load, actual_net_load, 6
        ),
        "wind_ramp_1h_mae_mw": _ramp_mae(scenario_mean[:, 0], actual_float[:, 0], 1),
        "solar_ramp_1h_mae_mw": _ramp_mae(scenario_mean[:, 1], actual_float[:, 1], 1),
        "load_ramp_1h_mae_mw": _ramp_mae(scenario_mean[:, 2], actual_float[:, 2], 1),
        "cross_variable_corr_mae": float(
            np.mean(np.abs(generated_corr[off_diagonal] - actual_corr[off_diagonal]))
        ),
    }


def read_env_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def model_parameter_count(run_dir: Path) -> int:
    with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = build_model(config["model"], torch.device("cpu"))
    return int(sum(parameter.numel() for parameter in model.parameters()))


def load_result(result_dir: Path, parameter_cache: dict[Path, int]) -> dict:
    required = [
        "metrics.json",
        "validation_metadata.json",
        "generation_config_used.yaml",
        "actual_scenarios.npy",
        "actual_data.npy",
        "denormalization_used.json",
    ]
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{result_dir}: missing {missing}")
    with (result_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (result_dir / "validation_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    run_dir = Path(metadata["training_run_dir"]).resolve()
    if not run_dir.is_dir():
        portable_run_dir = result_dir.parent / metadata["training_run_name"]
        if portable_run_dir.is_dir():
            run_dir = portable_run_dir.resolve()
        else:
            raise FileNotFoundError(
                f"training run is unavailable at {run_dir} or {portable_run_dir}"
            )
    if run_dir not in parameter_cache:
        parameter_cache[run_dir] = model_parameter_count(run_dir)
    run_record = read_env_record(run_dir / "logs" / "server_run_record.env")
    diagnostics = compute_saved_diagnostics(result_dir)

    row = {
        "architecture": metadata["architecture"],
        "training_run": run_dir.name,
        "checkpoint_rank": int(metadata["checkpoint_rank"]),
        "checkpoint_epoch": int(metadata["checkpoint_epoch"]),
        "validation_epsilon_mse": float(metadata["validation_epsilon_mse"]),
        "condition_ablation": metadata["condition_ablation"],
        "generation_seed": int(metadata["generation_seed"]),
        "n_samples": int(metadata["n_samples"]),
        "reverse_variance_type": metadata["reverse_variance_type"],
        "data_split": metadata["data_split"],
        "commit": metadata["commit"],
        "parameter_count": parameter_cache[run_dir],
        "training_seconds": int(run_record.get("training_seconds", -1)),
        "generation_seconds": int(metadata["generation_seconds"]),
        "result_dir": str(result_dir.resolve()),
        "checkpoint_path": metadata["checkpoint_path"],
        "multivariate_es": float(metrics["multivariate_es"]),
        "total_crps": float(metrics["total_crps"]),
        "total_energy_score": float(metrics["total_energy_score"]),
        "total_acf_mae": float(metrics["total_acf_mae"]),
    }
    for channel in ("wind", "solar", "load"):
        row[f"{channel}_crps"] = float(metrics[f"{channel}_crps"])
        row[f"{channel}_acf_mae"] = float(metrics[f"{channel}_acf_mae"])
    for nominal in (80, 90, 95):
        coverage = float(metrics[f"total_coverage_{nominal}%"])
        row[f"total_coverage_{nominal}_pct"] = coverage
        row[f"total_coverage_deviation_{nominal}_pct"] = coverage - nominal
        row[f"total_width_{nominal}_pct"] = float(metrics[f"total_width_{nominal}%"])
    row.update(diagnostics)
    constrained_metrics_path = result_dir / "metrics_constrained.json"
    constrained_scenarios_path = result_dir / "actual_scenarios_constrained.npy"
    row["physical_projection_available"] = bool(
        constrained_metrics_path.is_file() and constrained_scenarios_path.is_file()
    )
    constrained_fields = (
        "multivariate_es",
        "total_crps",
        "total_energy_score",
        "total_acf_mae",
        "total_coverage_90_pct",
        "total_width_90_pct",
        "any_physical_violation_pct",
        "net_load_mae_mw",
        "net_load_ramp_6h_mae_mw",
        "cross_variable_corr_mae",
    )
    for field in constrained_fields:
        row[f"constrained_{field}"] = ""
    if row["physical_projection_available"]:
        with constrained_metrics_path.open("r", encoding="utf-8") as handle:
            constrained_metrics = json.load(handle)
        constrained_diagnostics = compute_saved_diagnostics(
            result_dir,
            scenarios_filename="actual_scenarios_constrained.npy",
        )
        row.update({
            "constrained_multivariate_es": float(
                constrained_metrics["multivariate_es"]
            ),
            "constrained_total_crps": float(constrained_metrics["total_crps"]),
            "constrained_total_energy_score": float(
                constrained_metrics["total_energy_score"]
            ),
            "constrained_total_acf_mae": float(
                constrained_metrics["total_acf_mae"]
            ),
            "constrained_total_coverage_90_pct": float(
                constrained_metrics["total_coverage_90%"]
            ),
            "constrained_total_width_90_pct": float(
                constrained_metrics["total_width_90%"]
            ),
            "constrained_any_physical_violation_pct": float(
                constrained_diagnostics["any_physical_violation_pct"]
            ),
            "constrained_net_load_mae_mw": float(
                constrained_diagnostics["net_load_mae_mw"]
            ),
            "constrained_net_load_ramp_6h_mae_mw": float(
                constrained_diagnostics["net_load_ramp_6h_mae_mw"]
            ),
            "constrained_cross_variable_corr_mae": float(
                constrained_diagnostics["cross_variable_corr_mae"]
            ),
        })
    return row


def validate_protocol(rows: list[dict]) -> None:
    expected = {
        "generation_seed": 424242,
        "n_samples": 20,
        "reverse_variance_type": "posterior",
        "data_split": "val",
    }
    for row in rows:
        mismatches = {
            key: row[key] for key, value in expected.items() if row[key] != value
        }
        if mismatches:
            raise ValueError(f"protocol mismatch in {row['result_dir']}: {mismatches}")


def add_baseline_deltas(rows: list[dict]) -> None:
    baseline = next((
        row for row in rows
        if row["architecture"] == "v4_legacy"
        and row["checkpoint_rank"] == 1
        and row["condition_ablation"] == "none"
    ), None)
    for row in rows:
        for metric in ("total_crps", "multivariate_es", "total_acf_mae"):
            key = f"{metric}_delta_vs_v4_rank1_pct"
            row[key] = ""
            if baseline is not None and baseline[metric] != 0:
                row[key] = (row[metric] - baseline[metric]) * 100.0 / baseline[metric]


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.6g}")
            else:
                formatted.append(str(value))
        lines.append("| " + " | ".join(formatted) + " |")
    return lines


def write_markdown(rows: list[dict], path: Path) -> None:
    primary = [row for row in rows if row["condition_ablation"] == "none"]
    ablations = [row for row in rows if row["condition_ablation"] != "none"]
    lines = [
        "# V5 Stage-1 validation comparison",
        "",
        "All rows use validation data, posterior reverse variance, 20 ensemble members, and generation seed 424242. Lower is better for MSE, CRPS, Energy Score, ACF error, ramp error, correlation error, and boundary violations. Coverage deviations should approach zero.",
        "",
        "## Primary top-3 checkpoint results",
        "",
    ]
    lines.extend(markdown_table(
        ["architecture", "rank", "epoch", "val MSE", "CRPS", "MV ES", "cov90 dev", "width90", "ACF", "ramp6h", "net-load MAE"],
        [[
            row["architecture"], row["checkpoint_rank"], row["checkpoint_epoch"],
            row["validation_epsilon_mse"], row["total_crps"], row["multivariate_es"],
            row["total_coverage_deviation_90_pct"], row["total_width_90_pct"],
            row["total_acf_mae"], row["net_load_ramp_6h_mae_mw"],
            row["net_load_mae_mw"],
        ] for row in primary]
    ))
    constrained = [row for row in rows if row["physical_projection_available"]]
    if constrained:
        lines.extend(["", "## Physical-projection results", ""])
        lines.append(
            "Raw artifacts remain unchanged. These columns use the separately "
            "saved constrained scenarios."
        )
        lines.append("")
        lines.extend(markdown_table(
            [
                "architecture", "rank", "ablation", "CRPS", "MV ES",
                "cov90", "width90", "ACF", "ramp6h", "net-load MAE",
                "any violation",
            ],
            [[
                row["architecture"], row["checkpoint_rank"],
                row["condition_ablation"], row["constrained_total_crps"],
                row["constrained_multivariate_es"],
                row["constrained_total_coverage_90_pct"],
                row["constrained_total_width_90_pct"],
                row["constrained_total_acf_mae"],
                row["constrained_net_load_ramp_6h_mae_mw"],
                row["constrained_net_load_mae_mw"],
                row["constrained_any_physical_violation_pct"],
            ] for row in constrained],
        ))
    lines.extend(["", "## Raw physical-boundary diagnostics", ""])
    lines.extend(markdown_table(
        ["architecture", "rank", "ablation", "wind<0", "wind>cap", "solar<0", "solar>cap", "load<0", "any violation"],
        [[
            row["architecture"], row["checkpoint_rank"], row["condition_ablation"],
            row["wind_below_zero_pct"], row["wind_above_capacity_pct"],
            row["solar_below_zero_pct"], row["solar_above_capacity_pct"],
            row["load_below_zero_pct"], row["any_physical_violation_pct"],
        ] for row in rows]
    ))
    if ablations:
        reference = next((
            row for row in primary
            if row["architecture"] == "v5_tf" and row["checkpoint_rank"] == 1
        ), None)
        lines.extend(["", "## V5-TF rank-1 condition ablations", ""])
        ablation_rows = []
        for row in ablations:
            delta = "NA" if reference is None else row["total_crps"] - reference["total_crps"]
            ablation_rows.append([
                row["condition_ablation"], row["total_crps"], delta,
                row["multivariate_es"], row["total_acf_mae"],
                row["net_load_ramp_6h_mae_mw"], row["cross_variable_corr_mae"],
            ])
        lines.extend(markdown_table(
            ["ablation", "CRPS", "CRPS delta", "MV ES", "ACF", "ramp6h", "corr error"],
            ablation_rows,
        ))
    lines.extend([
        "",
        "No final checkpoint is selected automatically. Selection must jointly consider probability scores, calibration, temporal structure, physical diagnostics, and condition dependence.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    parameter_cache: dict[Path, int] = {}
    rows = [load_result(path.resolve(), parameter_cache) for path in args.result_dirs]
    validate_protocol(rows)
    rows.sort(key=lambda row: (
        ARCHITECTURE_ORDER[row["architecture"]],
        row["checkpoint_rank"],
        row["condition_ablation"],
    ))
    add_baseline_deltas(rows)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(rows[0])
    with (args.output_dir / "v5_stage1_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "v5_stage1_comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    write_markdown(rows, args.output_dir / "v5_stage1_comparison.md")
    print(f"WROTE_COMPARISON rows={len(rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
