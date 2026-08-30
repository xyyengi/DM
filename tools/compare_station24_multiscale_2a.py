"""Compare State V1 bottleneck-only graph mixing with Experiment 2A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["state_v1_fixed_graph", "state_v1_multiscale_graph"]
LABELS = {
    "state_v1_fixed_graph": "State V1 / bottleneck graph",
    "state_v1_multiscale_graph": "Experiment 2A / multiscale graph",
}
TITLE = "Station24 Experiment 2A paired validation comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs=2)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-variant", default=ORDER[0])
    parser.add_argument("--candidate-variant", default=ORDER[1])
    parser.add_argument("--baseline-label", default=LABELS[ORDER[0]])
    parser.add_argument("--candidate-label", default=LABELS[ORDER[1]])
    parser.add_argument("--baseline-spatial-levels", nargs="+", default=["bottleneck"])
    parser.add_argument(
        "--candidate-spatial-levels",
        nargs="+",
        default=["encoder_0", "encoder_1", "bottleneck"],
    )
    parser.add_argument("--baseline-parallel-levels", nargs="*", default=[])
    parser.add_argument("--candidate-parallel-levels", nargs="*", default=[])
    parser.add_argument("--baseline-parallel-adjacency", default="fixed")
    parser.add_argument("--candidate-parallel-adjacency", default="fixed")
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--figure-prefix", default="multiscale_2a")
    return parser.parse_args()


def configure_comparison(args: argparse.Namespace) -> None:
    global ORDER, LABELS, TITLE
    if args.baseline_variant == args.candidate_variant:
        raise ValueError("baseline and candidate variants must differ")
    ORDER = [args.baseline_variant, args.candidate_variant]
    LABELS = {
        ORDER[0]: args.baseline_label,
        ORDER[1]: args.candidate_label,
    }
    TITLE = args.title


def nested(metrics: dict, *keys: str) -> float:
    value = metrics
    for key in keys:
        value = value[key]
    return float(value)


def load_results(
    paths: list[str] | tuple[str, ...],
    expected_baseline_levels: list[str],
    expected_candidate_levels: list[str],
    expected_baseline_parallel_levels: list[str],
    expected_candidate_parallel_levels: list[str],
    expected_baseline_parallel_adjacency: str,
    expected_candidate_parallel_adjacency: str,
) -> dict[str, dict]:
    results = {}
    signatures = set()
    for raw in paths:
        path = Path(raw)
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        run = metrics["run"]
        variant = run.get("condition_variant")
        if variant not in ORDER or variant in results:
            raise ValueError(f"unexpected or duplicate comparison variant {variant}")
        if run["spatial_mode"] != "fixed_graph":
            raise ValueError("paired comparison requires fixed_graph")
        if bool(run.get("test_used")) or run["split"] != "val":
            raise ValueError("paired comparison is validation-only")
        signatures.add(
            (
                run["split"],
                int(run["n_samples"]),
                int(run["generation_seed"]),
                run["physical_projection"],
            )
        )
        results[variant] = {"path": path, "metrics": metrics}
    if set(results) != set(ORDER) or len(signatures) != 1:
        raise ValueError("variants or generation protocols do not match")
    observed_baseline_levels = results[ORDER[0]]["metrics"]["run"].get(
        "spatial_mix_levels", ["bottleneck"]
    )
    observed_candidate_levels = results[ORDER[1]]["metrics"]["run"].get(
        "spatial_mix_levels"
    )
    if observed_baseline_levels != list(expected_baseline_levels):
        raise ValueError(
            f"unexpected baseline graph levels: {observed_baseline_levels}; "
            f"expected={list(expected_baseline_levels)}"
        )
    if observed_candidate_levels != list(expected_candidate_levels):
        raise ValueError(
            f"unexpected candidate graph levels: {observed_candidate_levels}; "
            f"expected={list(expected_candidate_levels)}"
        )
    for variant, expected_levels, expected_adjacency, role in [
        (
            ORDER[0],
            expected_baseline_parallel_levels,
            expected_baseline_parallel_adjacency,
            "baseline",
        ),
        (
            ORDER[1],
            expected_candidate_parallel_levels,
            expected_candidate_parallel_adjacency,
            "candidate",
        ),
    ]:
        run = results[variant]["metrics"]["run"]
        observed_parallel = run.get("parallel_spatial_fusion_levels", [])
        if observed_parallel != list(expected_levels):
            raise ValueError(
                f"unexpected {role} parallel levels: {observed_parallel}; "
                f"expected={list(expected_levels)}"
            )
        observed_adjacency = run.get("parallel_spatial_adjacency_mode", "fixed")
        if observed_adjacency != expected_adjacency:
            raise ValueError(
                f"unexpected {role} parallel adjacency: {observed_adjacency}; "
                f"expected={expected_adjacency}"
            )
    return results


def build_summary(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for variant in ORDER:
        metrics = results[variant]["metrics"]
        run = metrics["run"]
        row = {
            "variant": variant,
            "label": LABELS[variant],
            "parameter_count": int(run["parameter_count"]),
            "validation_objective": float(
                run.get(
                    "checkpoint_validation_objective",
                    run["checkpoint_validation_mse"],
                )
            ),
            "validation_objective_type": str(
                run.get("checkpoint_validation_objective_type", "legacy_unknown")
            ),
            "wind_station_crps": nested(metrics, "station_average", "wind", "crps"),
            "solar_daylight_crps": nested(
                metrics, "station_average", "solar_daylight", "crps"
            ),
            "wind_aggregate_mw_crps": nested(
                metrics, "aggregate_mw", "wind", "crps"
            ),
            "renewable_aggregate_mw_crps": nested(
                metrics, "aggregate_mw", "renewable", "crps"
            ),
            "energy_score_pu": nested(metrics, "joint", "energy_score_pu"),
            "variogram_score": nested(
                metrics, "joint", "adjacency_variogram_score"
            ),
            "spatial_corr_rmse": nested(
                metrics, "joint", "spatial_corr_rmse_all_pairs"
            ),
        }
        for level in [80, 90, 95]:
            for label, path in [
                ("wind_station", ("station_average", "wind")),
                ("solar_daylight", ("station_average", "solar_daylight")),
                ("wind_aggregate_mw", ("aggregate_mw", "wind")),
                ("renewable_aggregate_mw", ("aggregate_mw", "renewable")),
            ]:
                row[f"{label}_coverage_{level}"] = nested(
                    metrics, *path, f"coverage_{level}"
                )
                row[f"{label}_width_{level}"] = nested(
                    metrics, *path, f"width_{level}"
                )
        for lag in [1, 3, 6]:
            row[f"wind_ramp_crps_{lag}h"] = nested(
                metrics, "ramps", "wind", f"lag_{lag}h", "crps"
            )
            row[f"wind_extreme_ramp_coverage_90_{lag}h"] = nested(
                metrics, "extreme_ramps", "wind", f"lag_{lag}h", "coverage_90"
            )
        row["wind_peak_coverage_90"] = nested(
            metrics, "extreme_high_daily_peak_mw", "wind", "coverage_90"
        )
        row["solar_peak_coverage_95"] = nested(
            metrics, "extreme_high_daily_peak_mw", "solar_daylight", "coverage_95"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_gate_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for variant in ORDER:
        run = results[variant]["metrics"]["run"]
        for gate, value in run.get("spatial_gate_values", {}).items():
            rows.append(
                {
                    "variant": variant,
                    "label": LABELS[variant],
                    "gate": gate,
                    "sigmoid_gate_value": float(value),
                }
            )
        for gate, value in run.get(
            "parallel_spatial_gate_statistics", {}
        ).items():
            rows.append(
                {
                    "variant": variant,
                    "label": LABELS[variant],
                    "gate": f"parallel/{gate}",
                    "sigmoid_gate_value": float(value),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    x = np.arange(len(summary))
    labels = [LABELS[variant] for variant in ORDER]
    plots = [
        ("wind_station_coverage_90", "Wind station 90% coverage", 0.90),
        ("wind_aggregate_mw_coverage_90", "Wind aggregate 90% coverage", 0.90),
        ("solar_daylight_coverage_90", "Solar daylight 90% coverage", 0.90),
        ("wind_station_crps", "Wind station CRPS", None),
        ("wind_aggregate_mw_crps", "Wind aggregate MW CRPS", None),
        ("wind_extreme_ramp_coverage_90_3h", "Extreme wind 3h coverage", 0.90),
        ("energy_score_pu", "Joint energy score", None),
        ("spatial_corr_rmse", "Spatial correlation RMSE", None),
    ]
    for axis, (column, title, nominal) in zip(axes.flat, plots, strict=True):
        axis.bar(x, summary[column], color=["#64748b", "#0f766e"])
        if nominal is not None:
            axis.axhline(nominal, color="#dc2626", linestyle="--", linewidth=1)
            axis.set_ylim(0, 1)
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(TITLE)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_typical(results: dict[str, dict], stations: pd.DataFrame, output: Path) -> int:
    base = results[ORDER[0]]["path"]
    actual = np.load(base / "actual_data_normalized.npy")
    forecast = np.load(base / "forecast_data_normalized.npy")
    order = np.argsort(np.mean(np.abs(actual - forecast), axis=(1, 2)))
    typical = int(order[len(order) // 2])
    types = stations.data_type.to_numpy()
    capacities = stations.capacity_mw.to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.5), sharex=True)
    for row, variant in enumerate(ORDER):
        samples = np.load(
            results[variant]["path"] / "actual_scenarios_normalized.npy",
            mmap_mode="r",
        )
        for column, station_type in enumerate(["wind", "solar"]):
            indices = np.flatnonzero(types == station_type)
            capacity = capacities[indices]
            scenario = np.sum(
                samples[typical][:, :, indices] * capacity[None, None, :], axis=-1
            )
            truth = np.sum(actual[typical][:, indices] * capacity[None, :], axis=-1)
            point = np.sum(
                forecast[typical][:, indices] * capacity[None, :], axis=-1
            )
            axis = axes[row, column]
            lead = np.arange(1, 169)
            axis.fill_between(
                lead,
                np.quantile(scenario, 0.05, axis=0),
                np.quantile(scenario, 0.95, axis=0),
                color="#fb7185",
                alpha=0.25,
            )
            axis.plot(lead, np.quantile(scenario, 0.50, axis=0), color="#e11d48")
            axis.plot(lead, point, color="#0d9488", linestyle="--")
            axis.plot(lead, truth, color="#111827")
            axis.set_title(f"{LABELS[variant]} - {station_type}")
            axis.set_ylabel("Aggregated MW")
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Lead hour")
    fig.suptitle(f"Representative validation issue index {typical}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return typical


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = [
            f"{value:.5f}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    configure_comparison(args)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)
    results = load_results(
        args.result_dirs,
        list(args.baseline_spatial_levels),
        list(args.candidate_spatial_levels),
        list(args.baseline_parallel_levels),
        list(args.candidate_parallel_levels),
        str(args.baseline_parallel_adjacency),
        str(args.candidate_parallel_adjacency),
    )
    summary = build_summary(results)
    gates = build_gate_table(results)
    summary.to_csv(output / "comparison_summary.csv", index=False)
    gates.to_csv(output / "spatial_gate_values.csv", index=False)
    plot_summary(summary, figures / f"{args.figure_prefix}_key_metrics.png")
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    )
    typical = plot_typical(
        results, stations, figures / "typical_scenario_envelopes.png"
    )
    columns = [
        "label",
        "parameter_count",
        "validation_objective",
        "validation_objective_type",
        "wind_station_crps",
        "wind_station_coverage_90",
        "wind_aggregate_mw_crps",
        "wind_aggregate_mw_coverage_90",
        "solar_daylight_crps",
        "solar_daylight_coverage_90",
        "wind_extreme_ramp_coverage_90_3h",
        "energy_score_pu",
        "variogram_score",
        "spatial_corr_rmse",
    ]
    report = f"# {TITLE}\n\n"
    report += (
        f"Paired comparison: {LABELS[ORDER[0]]} versus {LABELS[ORDER[1]]}. "
        "State V1 features, residual target, FiLM, optimizer, validation members, "
        "generation seed, reverse steps, and physical projection are unchanged. "
        f"Candidate sequential graph levels={list(args.candidate_spatial_levels)}; "
        f"parallel fusion levels={list(args.candidate_parallel_levels)}; "
        f"parallel adjacency={args.candidate_parallel_adjacency}.\n\n"
    )
    report += markdown_table(summary[columns]) + "\n\n"
    report += f"Representative validation issue index: `{typical}`.\n"
    (output / "comparison_report.md").write_text(report, encoding="utf-8")
    print(f"MULTISCALE_2A_COMPARISON_COMPLETE output={output}")


if __name__ == "__main__":
    main()
