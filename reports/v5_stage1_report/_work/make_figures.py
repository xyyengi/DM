from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\DM_local")
OUT = ROOT / "reports" / "v5_stage1_report" / "_work" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COMPARISON = (
    ROOT
    / "outputs_shandong"
    / "v5_stage1"
    / "comparisons"
    / "20260723_physical_projection"
    / "v5_stage1_comparison.csv"
)

RUNS = {
    "V4-RS": ROOT
    / "outputs_shandong"
    / "v5_stage1"
    / "20260722_143437_v4rs_repro_stage1_seed2026_20260722_143431",
    "V5-T": ROOT
    / "outputs_shandong"
    / "v5_stage1"
    / "20260722_151755_v5_t_stage1_seed2026_20260722_151749",
    "V5-TF": ROOT
    / "outputs_shandong"
    / "v5_stage1"
    / "20260722_155013_v5_tf_stage1_seed2026_20260722_155007",
}

RESULTS = {
    "V4-RS": Path(
        r"D:\DM_local\outputs_shandong\v5_stage1\20260722_143437_v4rs_repro_stage1_seed2026_20260722_143431_val_rank1_epoch11_posterior_n20_seed424242"
    ),
    "V5-T": Path(
        r"D:\DM_local\outputs_shandong\v5_stage1\20260722_151755_v5_t_stage1_seed2026_20260722_151749_val_rank1_epoch29_posterior_n20_seed424242"
    ),
    "V5-TF": Path(
        r"D:\DM_local\outputs_shandong\v5_stage1\20260722_155013_v5_tf_stage1_seed2026_20260722_155007_val_rank1_epoch8_posterior_n20_seed424242"
    ),
}

COLORS = {"V4-RS": "#777777", "V5-T": "#4C78A8", "V5-TF": "#E45756"}
CHANNELS = ["风电", "光伏", "负荷"]
UNITS = ["MW", "MW", "MW"]

matplotlib.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_rows() -> list[dict[str, str]]:
    with COMPARISON.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rank1_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    arch_name = {"v4_legacy": "V4-RS", "v5_t": "V5-T", "v5_tf": "V5-TF"}
    out = {}
    for row in rows:
        if row["checkpoint_rank"] == "1" and row["condition_ablation"] == "none":
            out[arch_name[row["architecture"]]] = row
    return out


def parse_training_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pattern = re.compile(
        r"Epoch\s+(\d+).*?Train=([0-9.eE+-]+).*?Val=([0-9.eE+-]+)"
    )
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            values.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    arr = np.asarray(values, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def make_training_curves(selected_epochs: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), sharey=True)
    for ax, (name, run) in zip(axes, RUNS.items()):
        epoch, train, val = parse_training_log(run / "logs" / "train_log.txt")
        ax.plot(epoch, train, color="#9ecae1", lw=1.2, label="训练损失")
        ax.plot(epoch, val, color=COLORS[name], lw=1.8, label="验证损失")
        chosen = selected_epochs[name]
        idx = np.where(epoch == chosen)[0][0]
        ax.scatter([chosen], [val[idx]], color="#111111", s=28, zorder=4)
        ax.axvline(chosen, color="#111111", ls="--", lw=0.8, alpha=0.55)
        ax.set_title(f"{name}（选中 epoch {chosen}）")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.115, max(0.255, float(train.max()) * 1.02))
    axes[0].set_ylabel("噪声预测 MSE")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("三种架构的训练与验证收敛曲线", fontsize=13, y=1.03)
    save(fig, "fig01_training_curves.png")


def make_metric_ratios(rank1: dict[str, dict[str, str]]) -> None:
    metrics = [
        ("constrained_total_crps", "CRPS"),
        ("constrained_multivariate_es", "MVES"),
        ("constrained_total_acf_mae", "ACF误差"),
        ("constrained_net_load_mae_mw", "净负荷MAE"),
        ("constrained_net_load_ramp_6h_mae_mw", "6h爬坡MAE"),
    ]
    x = np.arange(len(metrics))
    width = 0.24
    base = np.array([float(rank1["V4-RS"][m]) for m, _ in metrics])
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for i, name in enumerate(["V4-RS", "V5-T", "V5-TF"]):
        vals = np.array([float(rank1[name][m]) for m, _ in metrics]) / base
        bars = ax.bar(x + (i - 1) * width, vals, width, color=COLORS[name], label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhline(1.0, color="#333333", lw=0.8)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("相对 V4-RS（越低越好）")
    ax.set_ylim(0, 1.58)
    ax.legend(ncol=3, loc="upper center")
    ax.set_title("物理投影后 Rank-1 综合指标对比")
    save(fig, "fig02_metric_ratios.png")


def load_arrays(result: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual = np.load(result / "actual_data.npy")
    forecast = np.load(result / "forecast_data.npy")
    scenarios = np.load(result / "actual_scenarios_constrained.npy")
    return actual, forecast, scenarios


def representative_window(actual: np.ndarray, forecast: np.ndarray) -> tuple[int, float]:
    actual_net = actual[:, 2] - actual[:, 0] - actual[:, 1]
    forecast_net = forecast[:, 2] - forecast[:, 0] - forecast[:, 1]
    score = np.mean(np.abs(forecast_net - actual_net), axis=1)
    median = float(np.median(score))
    idx = int(np.argmin(np.abs(score - median)))
    return idx, float(score[idx])


def make_envelopes(rep_idx: int) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.5), sharex=True)
    for col, (name, result) in enumerate(RESULTS.items()):
        actual, forecast, scenarios = load_arrays(result)
        hours = np.arange(actual.shape[-1])
        for channel in range(3):
            ax = axes[channel, col]
            q05, q50, q95 = np.quantile(scenarios[rep_idx, :, channel, :], [0.05, 0.5, 0.95], axis=0)
            ax.fill_between(hours, q05, q95, color=COLORS[name], alpha=0.22, label="90%包络")
            ax.plot(hours, q50, color=COLORS[name], lw=1.25, label="场景中位数")
            ax.plot(hours, actual[rep_idx, channel], color="#111111", lw=1.05, label="实测")
            ax.plot(hours, forecast[rep_idx, channel], color="#2A9D8F", lw=0.9, ls="--", label="预测")
            if channel == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"{CHANNELS[channel]} / {UNITS[channel]}")
            if channel == 2:
                ax.set_xlabel("预测时效 / h")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"代表性验证窗口（样本索引 {rep_idx}）的 90% 场景包络", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save(fig, "fig03_envelope_comparison.png")


def make_scenario_curves(rep_idx: int) -> None:
    actual, forecast, scenarios = load_arrays(RESULTS["V5-TF"])
    hours = np.arange(actual.shape[-1])
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.3), sharex=True)
    for channel, ax in enumerate(axes):
        member = scenarios[rep_idx, :, channel, :]
        q05, q50, q95 = np.quantile(member, [0.05, 0.5, 0.95], axis=0)
        for s in range(member.shape[0]):
            ax.plot(hours, member[s], color=COLORS["V5-TF"], alpha=0.11, lw=0.55)
        ax.fill_between(hours, q05, q95, color=COLORS["V5-TF"], alpha=0.20, label="90%包络")
        ax.plot(hours, q50, color=COLORS["V5-TF"], lw=1.3, label="场景中位数")
        ax.plot(hours, actual[rep_idx, channel], color="#111111", lw=1.15, label="实测")
        ax.plot(hours, forecast[rep_idx, channel], color="#2A9D8F", ls="--", lw=0.9, label="预测")
        ax.set_ylabel(f"{CHANNELS[channel]} / MW")
    axes[-1].set_xlabel("预测时效 / h")
    axes[0].legend(ncol=4, loc="upper center", fontsize=8)
    fig.suptitle("V5-TF Rank-1：20 条场景曲线及概率包络", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig04_v5tf_scenario_curves.png")


def mean_acf(x: np.ndarray, max_lag: int = 24) -> np.ndarray:
    # x: (..., L); compute each sequence's normalized autocorrelation, then average.
    centered = x - x.mean(axis=-1, keepdims=True)
    denom = np.sum(centered * centered, axis=-1)
    out = []
    for lag in range(max_lag + 1):
        if lag == 0:
            corr = np.ones_like(denom)
        else:
            numer = np.sum(centered[..., :-lag] * centered[..., lag:], axis=-1)
            corr = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
        out.append(float(np.mean(corr)))
    return np.asarray(out)


def make_acf_curves() -> dict[str, dict[str, list[float]]]:
    actual, _, _ = load_arrays(RESULTS["V5-TF"])
    lags = np.arange(25)
    summary: dict[str, dict[str, list[float]]] = {}
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.45), sharey=True)
    for channel, ax in enumerate(axes):
        actual_acf = mean_acf(actual[:, channel, :])
        summary.setdefault("实测", {})[CHANNELS[channel]] = actual_acf.tolist()
        ax.plot(lags, actual_acf, color="#111111", lw=2.0, label="实测")
        for name, result in RESULTS.items():
            _, _, scenarios = load_arrays(result)
            model_acf = mean_acf(scenarios[:, :, channel, :])
            summary.setdefault(name, {})[CHANNELS[channel]] = model_acf.tolist()
            ax.plot(lags, model_acf, color=COLORS[name], lw=1.35, label=name)
        ax.set_title(CHANNELS[channel])
        ax.set_xlabel("滞后阶数 / h")
    axes[0].set_ylabel("平均自相关系数")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("物理投影后场景的平均自相关函数（ACF）", fontsize=13, y=1.03)
    save(fig, "fig05_acf_curves.png")
    return summary


def make_calibration(rank1: dict[str, dict[str, str]]) -> None:
    nominal = np.array([80.0, 90.0, 95.0])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([75, 97], [75, 97], color="#222222", ls="--", lw=1.0, label="理想校准")
    for name in ["V4-RS", "V5-T", "V5-TF"]:
        metrics = json.loads((RESULTS[name] / "metrics_constrained.json").read_text(encoding="utf-8"))
        observed = np.array(
            [
                float(metrics["total_coverage_80%"]),
                float(metrics["total_coverage_90%"]),
                float(metrics["total_coverage_95%"]),
            ]
        )
        ax.plot(nominal, observed, marker="o", color=COLORS[name], lw=1.6, label=name)
    ax.set_xlim(78, 96)
    ax.set_ylim(66, 91)
    ax.set_xlabel("名义覆盖率 / %")
    ax.set_ylabel("实测覆盖率 / %")
    ax.set_title("预测区间校准曲线")
    ax.legend()
    save(fig, "fig06_calibration.png")


def make_ablation_projection(rows: list[dict[str, str]], rank1: dict[str, dict[str, str]]) -> None:
    base = rank1["V5-TF"]
    ablations = {
        "完整条件": base,
        "去日历": next(r for r in rows if r["architecture"] == "v5_tf" and r["condition_ablation"] == "calendar"),
        "去预测": next(r for r in rows if r["architecture"] == "v5_tf" and r["condition_ablation"] == "forecast"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    labels = list(ablations)
    crps = [float(r["total_crps"]) for r in ablations.values()]
    acf = [float(r["total_acf_mae"]) for r in ablations.values()]
    x = np.arange(3)
    colors = ["#E45756", "#F2CF5B", "#72B7B2"]
    bars = axes[0].bar(x, crps, color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("CRPS（越低越好）")
    axes[0].set_title("条件消融")
    for b, v in zip(bars, crps):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 15, f"{v:.1f}", ha="center", fontsize=8)
    ax2 = axes[0].twinx()
    ax2.plot(x, acf, color="#333333", marker="o", lw=1.4)
    ax2.set_ylabel("ACF误差")

    names = ["V4-RS", "V5-T", "V5-TF"]
    raw = [float(rank1[n]["any_physical_violation_pct"]) for n in names]
    constrained = [float(rank1[n]["constrained_any_physical_violation_pct"]) for n in names]
    width = 0.34
    axes[1].bar(x - width / 2, raw, width, color="#D95F02", label="原始输出")
    axes[1].bar(x + width / 2, constrained, width, color="#1B9E77", label="物理投影后")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("任一物理越界率 / %")
    axes[1].set_title("物理投影效果")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "fig07_ablation_projection.png")


def main() -> None:
    rows = read_rows()
    rank1 = rank1_rows(rows)
    selected_epochs = {name: int(row["checkpoint_epoch"]) for name, row in rank1.items()}
    make_training_curves(selected_epochs)
    make_metric_ratios(rank1)
    actual, forecast, _ = load_arrays(RESULTS["V5-TF"])
    rep_idx, rep_score = representative_window(actual, forecast)
    make_envelopes(rep_idx)
    make_scenario_curves(rep_idx)
    acf = make_acf_curves()
    make_calibration(rank1)
    make_ablation_projection(rows, rank1)

    summary = {
        "representative_window_index_zero_based": rep_idx,
        "representative_window_number_one_based": rep_idx + 1,
        "selection_rule": "验证集中净负荷日前预测 MAE 最接近全体中位数的窗口",
        "representative_window_forecast_netload_mae_mw": rep_score,
        "rank1_selected_epochs": selected_epochs,
        "acf_curves": acf,
    }
    (OUT / "figure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "acf_curves"}, ensure_ascii=False, indent=2))
    for path in sorted(OUT.glob("*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
