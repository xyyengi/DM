from pathlib import Path
import os

ROOT = Path(r"D:\DM_local")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "ppt_update_20260701" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = ROOT / "tmp" / "step1_results_fixed_20260622" / "outputs" / "advanced_eval_fixed"
DIRECT = BASE / "20260621_220732_v2_csdi_cond_actual_given_forecast_168h" / "metrics_extreme_subsets.csv"
RESIDUAL = BASE / "20260621_233256_v_mix_residual_forecast_concat_guidance" / "metrics_extreme_subsets.csv"
OUT_DIR = ROOT / "outputs" / "ppt_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

subset_order = ["low_renewable", "high_load", "high_netload", "high_ramp"]
subset_labels = ["低新能源", "高负荷", "高净负荷", "高爬坡"]
scheme_labels = ["直接条件生成", "残差条件生成"]
colors = ["#2f73b7", "#d9a23f"]


def load_total(path: Path, scheme: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["channel"] == "total") & (df["subset"].isin(subset_order))].copy()
    df["scheme"] = scheme
    df["subset"] = pd.Categorical(df["subset"], categories=subset_order, ordered=True)
    return df.sort_values("subset")


data = pd.concat(
    [load_total(DIRECT, scheme_labels[0]), load_total(RESIDUAL, scheme_labels[1])],
    ignore_index=True,
)


def grouped_bar(metric: str, ylabel: str, title: str, out_name: str, ylim: tuple[float, float]):
    x = np.arange(len(subset_order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    for i, scheme in enumerate(scheme_labels):
        vals = (
            data[data["scheme"] == scheme]
            .set_index("subset")
            .loc[subset_order, metric]
            .astype(float)
            .to_numpy()
        )
        pos = x + (i - 0.5) * width
        bars = ax.bar(pos, vals, width, label=scheme, color=colors[i], edgecolor="white", linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] * 0.012),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#263238",
            )

    ax.set_title(title, fontsize=18, fontweight="bold", color="#142d4a", pad=14)
    ax.set_ylabel(ylabel, fontsize=12, color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels(subset_labels, fontsize=11)
    ax.tick_params(axis="y", labelsize=10, colors="#4b5563")
    ax.grid(axis="y", color="#d8dee8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=11)
    fig.text(
        0.01,
        0.01,
        "注：极端子集阈值由训练集分位数拟合，图中为测试集 total 指标。",
        fontsize=8.5,
        color="#6b7280",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


grouped_bar(
    "coverage_90",
    "Coverage 90%（%）",
    "极端子集 Coverage 90% 对比",
    "extreme_subset_coverage90.png",
    (0, 100),
)
grouped_bar(
    "interval_width_100",
    "Width 100%（归一化区间宽度，%）",
    "极端子集 Width 100% 对比",
    "extreme_subset_width100.png",
    (0, 75),
)

print(OUT_DIR / "extreme_subset_coverage90.png")
print(OUT_DIR / "extreme_subset_width100.png")
