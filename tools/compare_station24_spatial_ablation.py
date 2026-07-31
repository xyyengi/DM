"""Compare the three station24 spatial ablations on the same validation protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["none", "fixed_graph", "type_gated_graph"]
LABELS = {
    "none": "No spatial",
    "fixed_graph": "Fixed graph",
    "type_gated_graph": "Type-gated graph",
}
COLORS = {
    "none": "#6c757d",
    "fixed_graph": "#277da1",
    "type_gated_graph": "#7b2cbf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs=3)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def nested(metrics: dict[str, object], *keys: str) -> float:
    value: object = metrics
    for key in keys:
        value = value[key]  # type: ignore[index]
    return float(value)


def load_results(paths: list[str]) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    signatures = set()
    for raw_path in paths:
        path = Path(raw_path)
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        mode = metrics["run"]["spatial_mode"]
        if mode in results:
            raise ValueError(f"duplicate spatial mode {mode}")
        signatures.add(
            (
                metrics["run"]["split"],
                int(metrics["run"]["n_samples"]),
                int(metrics["run"]["generation_seed"]),
            )
        )
        results[mode] = {
            "path": path,
            "metrics": metrics,
            "station": pd.read_csv(path / "station_metrics.csv"),
            "lead": pd.read_csv(path / "lead_metrics.csv"),
        }
    if set(results) != set(ORDER):
        raise ValueError(f"expected modes {ORDER}, got {sorted(results)}")
    if len(signatures) != 1:
        raise ValueError(f"generation protocols differ: {signatures}")
    if next(iter(signatures))[0] != "val":
        raise ValueError("spatial ablation comparison must use validation split")
    return results


def build_summary(results: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for mode in ORDER:
        metrics = results[mode]["metrics"]
        joint = metrics["joint"]
        row = {
            "spatial_mode": mode,
            "label": LABELS[mode],
            "parameter_count": int(metrics["run"]["parameter_count"]),
            "validation_epsilon_mse": float(
                metrics["run"]["checkpoint_validation_mse"]
            ),
            "wind_crps": nested(metrics, "station_average", "wind", "crps"),
            "solar_crps": nested(metrics, "station_average", "solar", "crps"),
            "wind_coverage_90": nested(
                metrics, "station_average", "wind", "coverage_90"
            ),
            "solar_coverage_90": nested(
                metrics, "station_average", "solar", "coverage_90"
            ),
            "wind_width_90": nested(
                metrics, "station_average", "wind", "width_90"
            ),
            "solar_width_90": nested(
                metrics, "station_average", "solar", "width_90"
            ),
            "renewable_mw_crps": nested(
                metrics, "aggregate_mw", "renewable", "crps"
            ),
            "renewable_mw_coverage_90": nested(
                metrics, "aggregate_mw", "renewable", "coverage_90"
            ),
            "energy_score_pu": float(joint["energy_score_pu"]),
            "variogram_score": float(joint["adjacency_variogram_score"]),
            "spatial_corr_rmse": float(joint["spatial_corr_rmse_all_pairs"]),
            "spatial_corr_rmse_adjacent": float(
                joint["spatial_corr_rmse_adjacent_pairs"]
            ),
            "raw_below_zero_rate": nested(
                metrics, "physical", "raw_below_zero_rate"
            ),
            "raw_above_one_rate": nested(
                metrics, "physical", "raw_above_one_rate"
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_marginal(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    x = np.arange(len(summary))
    width = 0.36
    for axis, metric, title, target in [
        (axes[0], "crps", "Station-average CRPS", None),
        (axes[1], "coverage_90", "90% interval coverage", 0.90),
        (axes[2], "width_90", "90% interval width", None),
    ]:
        axis.bar(
            x - width / 2,
            summary[f"wind_{metric}"],
            width,
            label="wind",
            color="#277da1",
        )
        axis.bar(
            x + width / 2,
            summary[f"solar_{metric}"],
            width,
            label="solar",
            color="#f4a261",
        )
        if target is not None:
            axis.axhline(target, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(x)
        axis.set_xticklabels(summary.label, rotation=15, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_joint(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("energy_score_pu", "Energy Score"),
        ("variogram_score", "Adjacency Variogram Score"),
        ("spatial_corr_rmse", "Spatial correlation RMSE"),
    ]
    x = np.arange(len(summary))
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        axis.bar(x, summary[metric], color=[COLORS[value] for value in summary.spatial_mode])
        axis.set_xticks(x)
        axis.set_xticklabels(summary.label, rotation=15, ha="right")
        axis.set_title(title + " (lower is better)")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lead(results: dict[str, dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for axis, station_type in zip(axes, ["wind", "solar"], strict=True):
        for mode in ORDER:
            lead = results[mode]["lead"]
            subset = lead.loc[lead.station_type.eq(station_type)].sort_values("lead_day")
            axis.plot(
                subset.lead_day,
                subset.crps,
                marker="o",
                color=COLORS[mode],
                label=LABELS[mode],
            )
        axis.set_title(f"{station_type.capitalize()} CRPS by lead day")
        axis.set_xlabel("Lead day")
        axis.set_ylabel("CRPS (p.u.)")
        axis.set_xticks(range(1, 8))
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_typical_issue(actual: np.ndarray, forecast: np.ndarray) -> int:
    issue_mae = np.mean(np.abs(actual - forecast), axis=(1, 2))
    order = np.argsort(issue_mae)
    return int(order[len(order) // 2])


def plot_typical_scenarios(
    results: dict[str, dict[str, object]],
    stations: pd.DataFrame,
    output: Path,
) -> int:
    base_path = results["none"]["path"]
    actual = np.load(base_path / "actual_data_normalized.npy")
    forecast = np.load(base_path / "forecast_data_normalized.npy")
    typical = select_typical_issue(actual, forecast)
    capacities = stations.capacity_mw.to_numpy(dtype=np.float64)
    types = stations.data_type.to_numpy()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for row, mode in enumerate(ORDER):
        path = results[mode]["path"]
        scenarios = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
        for column, station_type in enumerate(["wind", "solar"]):
            indices = np.flatnonzero(types == station_type)
            selected_capacity = capacities[indices]
            scenario_mw = np.sum(
                scenarios[typical][:, :, indices]
                * selected_capacity[None, None, :],
                axis=-1,
            )
            actual_mw = np.sum(
                actual[typical][:, indices] * selected_capacity[None, :], axis=-1
            )
            forecast_mw = np.sum(
                forecast[typical][:, indices] * selected_capacity[None, :], axis=-1
            )
            lower = np.quantile(scenario_mw, 0.05, axis=0)
            median = np.quantile(scenario_mw, 0.50, axis=0)
            upper = np.quantile(scenario_mw, 0.95, axis=0)
            axis = axes[row, column]
            hours = np.arange(1, actual_mw.size + 1)
            axis.fill_between(hours, lower, upper, color="#ef476f", alpha=0.22, label="90% interval")
            axis.plot(hours, median, color="#ef476f", linewidth=1.2, label="scenario median")
            axis.plot(hours, forecast_mw, color="#2a9d8f", linestyle="--", linewidth=1.0, label="forecast")
            axis.plot(hours, actual_mw, color="black", linewidth=1.2, label="actual")
            axis.set_title(f"{LABELS[mode]} — {station_type}")
            axis.set_ylabel("Aggregated MW")
            axis.grid(alpha=0.20)
    for axis in axes[-1, :]:
        axis.set_xlabel("Lead hour")
    axes[0, 1].legend(frameon=False, ncol=2)
    fig.suptitle(f"Representative validation issue index {typical}")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return typical


def write_report(summary: pd.DataFrame, output_dir: Path, typical_issue: int) -> None:
    best_crps = summary.loc[summary.renewable_mw_crps.idxmin(), "label"]
    best_energy = summary.loc[summary.energy_score_pu.idxmin(), "label"]
    best_spatial = summary.loc[summary.spatial_corr_rmse.idxmin(), "label"]
    table = summary[
        [
            "label",
            "parameter_count",
            "wind_crps",
            "solar_crps",
            "wind_coverage_90",
            "solar_coverage_90",
            "energy_score_pu",
            "variogram_score",
            "spatial_corr_rmse",
        ]
    ].copy()
    headers = list(table.columns)
    rendered_rows = []
    for _, row in table.iterrows():
        rendered = []
        for header in headers:
            value = row[header]
            rendered.append(f"{value:.5f}" if isinstance(value, float) else str(value))
        rendered_rows.append("| " + " | ".join(rendered) + " |")
    markdown_table = (
        "| " + " | ".join(headers) + " |\n"
        + "|" + "|".join(["---"] * len(headers)) + "|\n"
        + "\n".join(rendered_rows)
    )
    text = "# 24场站空间消融实验比较\n\n"
    text += "三项实验使用同一训练/验证划分、训练种子、生成种子、80个场景成员和物理投影。测试集未使用。\n\n"
    text += markdown_table
    text += "\n\n## 自动汇总结论\n\n"
    text += f"- 省级风光总量CRPS最优：**{best_crps}**。\n"
    text += f"- 联合Energy Score最优：**{best_energy}**。\n"
    text += f"- 空间相关矩阵恢复最优：**{best_spatial}**。\n"
    text += "- 最终选择不能只看单一指标；空间模型应在不明显损害逐站CRPS和覆盖率的前提下改善Energy/Variogram Score及空间相关性。\n"
    text += f"- 典型验证批次按确定性预测MAE的中位数选取，索引为 `{typical_issue}`，不是人工挑选最好看的窗口。\n\n"
    text += "## 图表\n\n"
    text += "![边际概率指标](figures/marginal_metrics.png)\n\n"
    text += "![联合空间指标](figures/joint_metrics.png)\n\n"
    text += "![分时距CRPS](figures/lead_crps.png)\n\n"
    text += "![典型场景包络](figures/typical_scenario_envelopes.png)\n"
    (output_dir / "comparison_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True)
    results = load_results(args.result_dirs)
    summary = build_summary(results)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    plot_marginal(summary, figure_dir / "marginal_metrics.png")
    plot_joint(summary, figure_dir / "joint_metrics.png")
    plot_lead(results, figure_dir / "lead_crps.png")
    typical = plot_typical_scenarios(
        results, stations, figure_dir / "typical_scenario_envelopes.png"
    )
    write_report(summary, output_dir, typical)
    print(f"COMPARISON_COMPLETE output={output_dir}")


if __name__ == "__main__":
    main()
