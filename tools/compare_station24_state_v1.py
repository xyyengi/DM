"""Compare the paired Fixed Graph control and causal state-v1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["ramp36_control", "state_v1_fixed_graph"]
LABELS = {
    "ramp36_control": "Recent error + raw 3/6h ramps",
    "state_v1_fixed_graph": "Recent error + State V1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs=2)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def nested(metrics, *keys):
    value = metrics
    for key in keys:
        value = value[key]
    return float(value)


def load_results(paths):
    results = {}
    signatures = set()
    for raw in paths:
        path = Path(raw)
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        variant = metrics["run"].get("condition_variant")
        if variant not in ORDER or variant in results:
            raise ValueError(f"unexpected or duplicate state-v1 variant {variant}")
        if metrics["run"]["spatial_mode"] != "fixed_graph":
            raise ValueError("state-v1 comparison requires fixed_graph")
        signatures.add(
            (
                metrics["run"]["split"],
                int(metrics["run"]["n_samples"]),
                int(metrics["run"]["generation_seed"]),
            )
        )
        results[variant] = {"path": path, "metrics": metrics}
    if set(results) != set(ORDER) or len(signatures) != 1:
        raise ValueError("state-v1 variants or generation protocols do not match")
    if next(iter(signatures))[0] != "val":
        raise ValueError("state-v1 comparison must use validation only")
    return results


def build_summary(results):
    rows = []
    for variant in ORDER:
        metrics = results[variant]["metrics"]
        row = {
            "variant": variant,
            "label": LABELS[variant],
            "parameter_count": int(metrics["run"]["parameter_count"]),
            "validation_epsilon_mse": float(metrics["run"]["checkpoint_validation_mse"]),
            "wind_crps": nested(metrics, "station_average", "wind", "crps"),
            "solar_daylight_crps": nested(metrics, "station_average", "solar_daylight", "crps"),
            "renewable_mw_crps": nested(metrics, "aggregate_mw", "renewable", "crps"),
            "energy_score_pu": nested(metrics, "joint", "energy_score_pu"),
            "variogram_score": nested(metrics, "joint", "adjacency_variogram_score"),
            "spatial_corr_rmse": nested(metrics, "joint", "spatial_corr_rmse_all_pairs"),
        }
        for level in [80, 90, 95]:
            for scope, path in [
                ("wind", ("station_average", "wind")),
                ("solar_daylight", ("station_average", "solar_daylight")),
                ("renewable_mw", ("aggregate_mw", "renewable")),
            ]:
                row[f"{scope}_coverage_{level}"] = nested(
                    metrics, *path, f"coverage_{level}"
                )
                row[f"{scope}_width_{level}"] = nested(
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
        row["solar_peak_coverage_90"] = nested(
            metrics, "extreme_high_daily_peak_mw", "solar_daylight", "coverage_90"
        )
        row["solar_peak_coverage_95"] = nested(
            metrics, "extreme_high_daily_peak_mw", "solar_daylight", "coverage_95"
        )
        for name, value in metrics["run"].get("condition_gate_values", {}).items():
            row[f"condition_gate_{name}"] = float(value)
        for name, value in metrics["run"].get("state_gate_values", {}).items():
            row[f"state_gate_{name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    x = np.arange(len(summary))
    labels = ["Raw ramps", "State V1"]
    plots = [
        ("wind_coverage_90", "Wind 90% coverage", 0.90),
        ("solar_daylight_coverage_90", "Solar daylight 90% coverage", 0.90),
        ("renewable_mw_coverage_90", "Renewable 90% coverage", 0.90),
        ("wind_crps", "Wind CRPS", None),
        ("wind_extreme_ramp_coverage_90_3h", "Extreme wind 3h-ramp coverage", 0.90),
        ("solar_peak_coverage_95", "Solar daily-peak 95% coverage", 0.95),
    ]
    for axis, (column, title, nominal) in zip(axes.flat, plots, strict=True):
        axis.bar(x, summary[column], color=["#64748b", "#0f766e"])
        if nominal is not None:
            axis.axhline(nominal, color="#dc2626", linestyle="--", linewidth=1)
            axis.set_ylim(0, 1)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Station24 State V1 paired validation comparison")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_typical(results, stations, output):
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
            point = np.sum(forecast[typical][:, indices] * capacity[None, :], axis=-1)
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
            axis.plot(lead, truth, color="black")
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


def markdown_table(frame):
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join(["---"] * len(frame.columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = [f"{value:.5f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)
    results = load_results(args.result_dirs)
    summary = build_summary(results)
    summary.to_csv(output / "comparison_summary.csv", index=False)
    plot_summary(summary, figures / "state_v1_key_metrics.png")
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    )
    typical = plot_typical(results, stations, figures / "typical_scenario_envelopes.png")
    columns = [
        "label",
        "parameter_count",
        "validation_epsilon_mse",
        "wind_crps",
        "wind_coverage_90",
        "solar_daylight_crps",
        "solar_daylight_coverage_90",
        "renewable_mw_crps",
        "renewable_mw_coverage_90",
        "wind_extreme_ramp_coverage_90_3h",
        "solar_peak_coverage_95",
        "energy_score_pu",
        "spatial_corr_rmse",
    ]
    report = "# Station24 State V1 comparison\n\n"
    report += "Paired validation runs use Fixed Graph, seed 2027, 80 members, 500 reverse steps, identical physical projection, and sealed test data.\n\n"
    report += markdown_table(summary[columns]) + "\n\n"
    report += f"Representative validation issue index: `{typical}`.\n"
    (output / "comparison_report.md").write_text(report, encoding="utf-8")
    print(f"STATE_V1_COMPARISON_COMPLETE output={output}")


if __name__ == "__main__":
    main()
