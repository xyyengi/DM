from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"D:\DM_local")
OUT = ROOT / "reports" / "v5_stage1_report" / "replacement_figures"
OUT.mkdir(parents=True, exist_ok=True)
IDX = 493

RUNS = {
    "V4-RS": ROOT / "outputs_shandong/v5_stage1/20260722_143437_v4rs_repro_stage1_seed2026_20260722_143431_val_rank1_epoch11_posterior_n20_seed424242",
    "V5-T": ROOT / "outputs_shandong/v5_stage1/20260722_151755_v5_t_stage1_seed2026_20260722_151749_val_rank1_epoch29_posterior_n20_seed424242",
    "V5-TF": ROOT / "outputs_shandong/v5_stage1/20260722_155013_v5_tf_stage1_seed2026_20260722_155007_val_rank1_epoch8_posterior_n20_seed424242",
}
COLORS = {"V4-RS": "#777777", "V5-T": "#4C78A8", "V5-TF": "#E45756"}
CHANNELS = ["风电", "光伏", "负荷"]

matplotlib.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "legend.frameon": False,
})


def arrays(path):
    return (
        np.load(path / "actual_data.npy"),
        np.load(path / "forecast_data.npy"),
        np.load(path / "actual_scenarios_constrained.npy"),
    )


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def comparison():
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.5), sharex=True)
    stats = {}
    for col, (name, path) in enumerate(RUNS.items()):
        actual, forecast, scenarios = arrays(path)
        hours = np.arange(168)
        for ch in range(3):
            member = scenarios[IDX, :, ch, :]
            q05, q50, q95 = np.quantile(member, [0.05, 0.5, 0.95], axis=0)
            ax = axes[ch, col]
            ax.fill_between(hours, q05, q95, color=COLORS[name], alpha=0.22, label="90%包络")
            ax.plot(hours, q50, color=COLORS[name], lw=1.25, label="场景中位数")
            ax.plot(hours, actual[IDX, ch], color="#111111", lw=1.05, label="实测")
            ax.plot(hours, forecast[IDX, ch], color="#2A9D8F", lw=0.9, ls="--", label="日前预测")
            if ch == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"{CHANNELS[ch]} / MW")
            if ch == 2:
                ax.set_xlabel("预测时效 / h")
        day = forecast[IDX, 1] > 1.0
        q05, q50, q95 = np.quantile(scenarios[IDX, :, 1, :], [0.05, 0.5, 0.95], axis=0)
        inside = (actual[IDX, 1] >= q05) & (actual[IDX, 1] <= q95)
        stats[name] = {
            "solar_daylight_coverage_pct": float(inside[day].mean() * 100),
            "solar_daylight_median_mae_mw": float(np.mean(np.abs(q50[day] - actual[IDX, 1, day]))),
            "solar_daylight_mean_width_mw": float(np.mean((q95 - q05)[day])),
        }
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("光伏特征典型的独立候选窗口（第494个）的90%场景包络", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    save(fig, "replacement_fig03_window494_three_model_envelope.png")
    return stats


def v5tf_curves():
    actual, forecast, scenarios = arrays(RUNS["V5-TF"])
    hours = np.arange(168)
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.3), sharex=True)
    for ch, ax in enumerate(axes):
        member = scenarios[IDX, :, ch, :]
        q05, q50, q95 = np.quantile(member, [0.05, 0.5, 0.95], axis=0)
        for s in range(member.shape[0]):
            ax.plot(hours, member[s], color=COLORS["V5-TF"], alpha=0.11, lw=0.55)
        ax.fill_between(hours, q05, q95, color=COLORS["V5-TF"], alpha=0.20, label="90%包络")
        ax.plot(hours, q50, color=COLORS["V5-TF"], lw=1.3, label="场景中位数")
        ax.plot(hours, actual[IDX, ch], color="#111111", lw=1.15, label="实测")
        ax.plot(hours, forecast[IDX, ch], color="#2A9D8F", ls="--", lw=0.9, label="日前预测")
        ax.set_ylabel(f"{CHANNELS[ch]} / MW")
    axes[-1].set_xlabel("预测时效 / h")
    axes[0].legend(ncol=4, loc="upper center", fontsize=8)
    fig.suptitle("V5-TF Rank-1：第494个窗口的20条场景及概率包络", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "replacement_fig04_window494_v5tf_20_scenarios.png")


stats = comparison()
v5tf_curves()
(OUT / "replacement_selection_and_stats.json").write_text(
    json.dumps({
        "index_zero_based": IDX,
        "window_one_based": IDX + 1,
        "selection_rule": (
            "按光伏日前预测日间MAE、窗口光伏总能量和平均爬坡强度的稳健标准化距离排序；"
            "为避免168小时滑窗高度重叠，选择与第一候选至少间隔168个窗口的下一独立候选。"
        ),
        "models": stats,
        "median_definition": (
            "对每个变量、每个小时，将20个场景值排序；偶数样本中位数为第10与第11个有序值的平均。"
        ),
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(stats, ensure_ascii=False, indent=2))
for p in sorted(OUT.glob("*.png")):
    print(p, p.stat().st_size)
