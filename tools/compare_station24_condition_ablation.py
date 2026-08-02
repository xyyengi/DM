"""Compare the three forecast-information condition ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["revision_ramp", "history_ramp", "revision_history_ramp"]
LABELS = {
    "revision_ramp": "Revision + ramps",
    "history_ramp": "Recent error + ramps",
    "revision_history_ramp": "Revision + recent error + ramps",
}
COLORS = ["#277da1", "#f8961e", "#7b2cbf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs=3)
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
            raise ValueError(f"unexpected or duplicate condition variant {variant}")
        if metrics["run"]["spatial_mode"] != "fixed_graph":
            raise ValueError("condition ablation requires fixed_graph for every run")
        signatures.add(
            (
                metrics["run"]["split"],
                int(metrics["run"]["n_samples"]),
                int(metrics["run"]["generation_seed"]),
            )
        )
        results[variant] = {"path": path, "metrics": metrics}
    if set(results) != set(ORDER) or len(signatures) != 1:
        raise ValueError("condition variants or generation protocols do not match")
    if next(iter(signatures))[0] != "val":
        raise ValueError("condition ablation must compare validation results")
    return results


def build_summary(results):
    rows = []
    for variant in ORDER:
        metrics = results[variant]["metrics"]
        row = {
            "condition_variant": variant,
            "label": LABELS[variant],
            "parameter_count": int(metrics["run"]["parameter_count"]),
            "validation_epsilon_mse": float(
                metrics["run"]["checkpoint_validation_mse"]
            ),
            "wind_crps": nested(metrics, "station_average", "wind", "crps"),
            "solar_daylight_crps": nested(
                metrics, "station_average", "solar_daylight", "crps"
            ),
            "renewable_mw_crps": nested(
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
            row[f"wind_coverage_{level}"] = nested(
                metrics, "station_average", "wind", f"coverage_{level}"
            )
            row[f"solar_daylight_coverage_{level}"] = nested(
                metrics,
                "station_average",
                "solar_daylight",
                f"coverage_{level}",
            )
            row[f"renewable_mw_coverage_{level}"] = nested(
                metrics, "aggregate_mw", "renewable", f"coverage_{level}"
            )
            row[f"solar_peak_coverage_{level}"] = nested(
                metrics,
                "extreme_high_daily_peak_mw",
                "solar_daylight",
                f"coverage_{level}",
            )
            row[f"wind_peak_coverage_{level}"] = nested(
                metrics,
                "extreme_high_daily_peak_mw",
                "wind",
                f"coverage_{level}",
            )
        for lag in [1, 3, 6]:
            row[f"wind_ramp_crps_{lag}h"] = nested(
                metrics, "ramps", "wind", f"lag_{lag}h", "crps"
            )
            row[f"wind_extreme_ramp_coverage_90_{lag}h"] = nested(
                metrics,
                "extreme_ramps",
                "wind",
                f"lag_{lag}h",
                "coverage_90",
            )
            row[f"solar_extreme_ramp_coverage_90_{lag}h"] = nested(
                metrics,
                "extreme_ramps",
                "solar_daylight",
                f"lag_{lag}h",
                "coverage_90",
            )
        for name, value in metrics["run"].get("condition_gate_values", {}).items():
            row[f"condition_gate_{name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_coverages(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    x = np.arange(len(summary))
    width = 0.22
    for axis, prefix, title in [
        (axes[0], "wind", "Wind stations"),
        (axes[1], "solar_daylight", "Solar stations (daylight)"),
        (axes[2], "renewable_mw", "Aggregated renewable MW"),
    ]:
        for offset, level, color in zip(
            [-width, 0, width], [80, 90, 95], ["#90be6d", "#277da1", "#f94144"], strict=True
        ):
            axis.bar(x + offset, summary[f"{prefix}_coverage_{level}"], width, color=color, label=f"{level}%")
            axis.axhline(level / 100, color=color, linestyle=":", linewidth=.8)
        axis.set_xticks(x)
        axis.set_xticklabels(summary.label, rotation=14, ha="right")
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Empirical coverage")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_event_metrics(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(summary))
    width = .22
    for offset, lag, color in zip([-width, 0, width], [1, 3, 6], ["#43aa8b", "#277da1", "#f3722c"], strict=True):
        axes[0].bar(x + offset, summary[f"wind_ramp_crps_{lag}h"], width, color=color, label=f"{lag}h")
        axes[1].bar(x + offset, summary[f"wind_extreme_ramp_coverage_90_{lag}h"], width, color=color, label=f"{lag}h")
    for offset, level, color in zip([-width, 0, width], [80, 90, 95], ["#90be6d", "#277da1", "#f94144"], strict=True):
        axes[2].bar(x + offset, summary[f"solar_peak_coverage_{level}"], width, color=color, label=f"{level}%")
        axes[2].axhline(level / 100, color=color, linestyle=":", linewidth=.8)
    for axis, title in zip(axes, ["Wind ramp CRPS", "Wind extreme-ramp 90% coverage", "Solar daily-peak coverage"], strict=True):
        axis.set_xticks(x)
        axis.set_xticklabels(summary.label, rotation=14, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=.25)
    axes[1].axhline(.90, color="black", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_typical_issue(actual, forecast):
    order = np.argsort(np.mean(np.abs(actual - forecast), axis=(1, 2)))
    return int(order[len(order) // 2])


def plot_typical(results, stations, output):
    base = results[ORDER[0]]["path"]
    actual = np.load(base / "actual_data_normalized.npy")
    forecast = np.load(base / "forecast_data_normalized.npy")
    typical = select_typical_issue(actual, forecast)
    types = stations.data_type.to_numpy()
    capacities = stations.capacity_mw.to_numpy(float)
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for row, variant in enumerate(ORDER):
        scenarios = np.load(results[variant]["path"] / "actual_scenarios_normalized.npy", mmap_mode="r")
        for col, station_type in enumerate(["wind", "solar"]):
            idx = np.flatnonzero(types == station_type)
            cap = capacities[idx]
            ss = np.sum(scenarios[typical][:, :, idx] * cap[None, None, :], axis=-1)
            aa = np.sum(actual[typical][:, idx] * cap[None, :], axis=-1)
            ff = np.sum(forecast[typical][:, idx] * cap[None, :], axis=-1)
            axis = axes[row, col]
            h = np.arange(1, 169)
            axis.fill_between(h, np.quantile(ss, .05, axis=0), np.quantile(ss, .95, axis=0), color="#ef476f", alpha=.22)
            axis.plot(h, np.quantile(ss, .5, axis=0), color="#ef476f", linewidth=1.2)
            axis.plot(h, ff, color="#2a9d8f", linestyle="--", linewidth=1)
            axis.plot(h, aa, color="black", linewidth=1.2)
            axis.set_title(f"{LABELS[variant]} — {station_type}")
            axis.set_ylabel("Aggregated MW")
            axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Lead hour")
    fig.suptitle(f"Representative validation issue index {typical}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return typical


def markdown_table(frame):
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in headers:
            value = row[column]
            values.append(f"{value:.5f}" if isinstance(value, float) else str(value))
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
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values("channel_index")
    plot_coverages(summary, figures / "coverage_80_90_95.png")
    plot_event_metrics(summary, figures / "ramp_and_extreme_metrics.png")
    typical = plot_typical(results, stations, figures / "typical_scenario_envelopes.png")
    report = "# Station24 condition-ablation comparison\n\n"
    report += "All runs use fixed graph, the same split, seeds, 80 members, 500 reverse steps, and physical projection. Test data are sealed.\n\n"
    report_columns = [
        "label",
        "parameter_count",
        "wind_crps",
        "wind_coverage_90",
        "solar_daylight_crps",
        "solar_daylight_coverage_90",
        "renewable_mw_crps",
        "renewable_mw_coverage_90",
        "wind_extreme_ramp_coverage_90_1h",
        "solar_peak_coverage_95",
        "energy_score_pu",
        "spatial_corr_rmse",
    ]
    report += markdown_table(summary[report_columns]) + "\n\n"
    report += f"Representative validation issue index: `{typical}`.\n"
    (output / "comparison_report.md").write_text(report, encoding="utf-8")
    print(f"CONDITION_COMPARISON_COMPLETE output={output}")


if __name__ == "__main__":
    main()
