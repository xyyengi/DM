import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("outputs/ppt_diffusion_summary")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "V0": "outputs/20260525_001242_v0_uncond_ddpm_actual_168h",
    "V1": "outputs/20260525_014620_v1_2023_guidance_actual_168h",
    "V2": "outputs/20260525_033557_v2_csdi_cond_actual_given_forecast_168h",
    "Vmix": "outputs/20260525_050050_v_mix_residual_forecast_concat_guidance",
}

CGAN_LABEL1 = "CGAN/lable1_metrics_diffusion_style"
CHANNELS = ["Wind", "PV", "Load"]
COLORS = {"Wind": "#2b6cb0", "PV": "#d69e2e", "Load": "#2f855a"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run(run_dir):
    run_dir = Path(run_dir)
    return {
        "samples": np.load(run_dir / "actual_scenarios.npy"),
        "actual": np.load(run_dir / "actual_data.npy"),
        "forecast": np.load(run_dir / "forecast_data.npy"),
        "metrics": load_json(run_dir / "metrics.json"),
    }


def style_axes(ax):
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


data = {name: load_run(path) for name, path in RUNS.items()}
cgan = load_json(CGAN_LABEL1)

vmix = data["Vmix"]
forecast_mae = np.mean(np.abs(vmix["forecast"] - vmix["actual"]), axis=(1, 2))
sample_idx = int(np.argsort(np.abs(forecast_mae - np.median(forecast_mae)))[0])
t = np.arange(vmix["actual"].shape[-1])

# Figure 1: generated scenarios with actual and forecast baselines.
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for c, ax in enumerate(axes):
    name = CHANNELS[c]
    for s in range(min(12, vmix["samples"].shape[1])):
        ax.plot(t, vmix["samples"][sample_idx, s, c], color=COLORS[name], alpha=0.18, linewidth=0.9)
    ax.plot(t, vmix["actual"][sample_idx, c], color="#111827", linewidth=1.7, label="Actual")
    ax.plot(t, vmix["forecast"][sample_idx, c], color="#6b7280", linestyle="--", linewidth=1.3, label="Forecast")
    ax.set_ylabel(name)
    style_axes(ax)
axes[0].legend(loc="upper right", fontsize=8, frameon=False)
axes[-1].set_xlabel("Hour")
fig.suptitle("Vmix Diffusion Generated Scenarios", fontsize=13, y=1.02)
save(FIG_DIR / "fig1_vmix_generated_scenarios.png")

# Figure 2: quantile envelope.
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for c, ax in enumerate(axes):
    name = CHANNELS[c]
    samples = vmix["samples"][sample_idx, :, c]
    p10, p25, p50, p75, p90 = np.percentile(samples, [10, 25, 50, 75, 90], axis=0)
    ax.fill_between(t, p10, p90, color=COLORS[name], alpha=0.16, label="P10-P90")
    ax.fill_between(t, p25, p75, color=COLORS[name], alpha=0.28, label="P25-P75")
    ax.plot(t, p50, color=COLORS[name], linewidth=1.4, label="Median")
    ax.plot(t, vmix["actual"][sample_idx, c], color="#111827", linewidth=1.5, label="Actual")
    ax.plot(t, vmix["forecast"][sample_idx, c], color="#6b7280", linestyle="--", linewidth=1.2, label="Forecast")
    ax.set_ylabel(name)
    style_axes(ax)
axes[0].legend(loc="upper right", ncol=5, fontsize=7, frameon=False)
axes[-1].set_xlabel("Hour")
fig.suptitle("Vmix Diffusion Quantile Envelope", fontsize=13, y=1.02)
save(FIG_DIR / "fig2_vmix_quantile_envelope.png")

# Figure 3: CGAN vs current DM metrics.
metrics = ["total_crps", "total_coverage_100%", "total_width_100%", "total_acf_mae"]
labels = ["CRPS", "Coverage 100%", "Width 100%", "ACF MAE"]
dm_vals = [vmix["metrics"][m] for m in metrics]
cgan_vals = [cgan[m] for m in metrics]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for i, ax in enumerate(axes):
    ax.bar(["DM", "CGAN"], [dm_vals[i], cgan_vals[i]], color=["#2563eb", "#dc2626"], width=0.58)
    ax.set_title(labels[i], fontsize=10)
    style_axes(ax)
fig.suptitle("Current Diffusion Model vs CGAN(label1)", fontsize=13, y=1.05)
save(FIG_DIR / "fig3_dm_vs_cgan_metrics.png")

# Figure 4: version metrics.
version_labels = list(data.keys())
version_metrics = ["mean_MAE", "total_crps", "total_coverage_100%", "total_acf_mae"]
version_titles = ["Mean MAE", "CRPS", "Coverage 100%", "ACF MAE"]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
for i, ax in enumerate(axes):
    vals = [data[v]["metrics"][version_metrics[i]] for v in version_labels]
    ax.bar(version_labels, vals, color=["#64748b", "#7c3aed", "#0891b2", "#16a34a"], width=0.62)
    ax.set_title(version_titles[i], fontsize=10)
    style_axes(ax)
fig.suptitle("Conditioning Strategy Ablation", fontsize=13, y=1.05)
save(FIG_DIR / "fig4_condition_versions_metrics.png")

# Figure 5: generated mean and envelope by version on the same sample.
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
version_colors = {"V0": "#64748b", "V1": "#7c3aed", "V2": "#0891b2", "Vmix": "#16a34a"}
for c, ax in enumerate(axes):
    for v in version_labels:
        samples = data[v]["samples"][sample_idx, :, c]
        p10, p90 = np.percentile(samples, [10, 90], axis=0)
        mean = samples.mean(axis=0)
        ax.fill_between(t, p10, p90, color=version_colors[v], alpha=0.08)
        ax.plot(t, mean, color=version_colors[v], linewidth=1.3, label=v)
    ax.plot(t, vmix["actual"][sample_idx, c], color="#111827", linewidth=1.7, label="Actual")
    ax.plot(t, vmix["forecast"][sample_idx, c], color="#6b7280", linestyle="--", linewidth=1.1, label="Forecast")
    ax.set_ylabel(CHANNELS[c])
    style_axes(ax)
axes[0].legend(loc="upper right", ncol=6, fontsize=7, frameon=False)
axes[-1].set_xlabel("Hour")
fig.suptitle("Same Case Comparison Across Conditioning Versions", fontsize=13, y=1.02)
save(FIG_DIR / "fig5_condition_versions_curves.png")

# Figure 6: correlation heatmaps using all test points.
actual_flat = np.transpose(vmix["actual"], (0, 2, 1)).reshape(-1, 3)
gen_flat = np.transpose(vmix["samples"].mean(axis=1), (0, 2, 1)).reshape(-1, 3)
corr_actual = np.corrcoef(actual_flat, rowvar=False)
corr_gen = np.corrcoef(gen_flat, rowvar=False)
corr_diff = corr_gen - corr_actual
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
for ax, mat, title, vmin, vmax, cmap in [
    (axes[0], corr_actual, "Actual corr", -1, 1, "RdBu_r"),
    (axes[1], corr_gen, "DM corr", -1, 1, "RdBu_r"),
    (axes[2], corr_diff, "Difference", -0.5, 0.5, "RdBu_r"),
]:
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(3), CHANNELS, fontsize=8)
    ax.set_yticks(range(3), CHANNELS, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Actual vs DM Generated Correlation", fontsize=13, y=1.04)
save(FIG_DIR / "fig6_correlation_heatmaps.png")

# Figure 7: distribution comparison.
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for c, ax in enumerate(axes):
    actual_vals = actual_flat[:, c]
    gen_vals = np.transpose(vmix["samples"], (0, 1, 3, 2)).reshape(-1, 3)[:, c]
    rng = (min(actual_vals.min(), gen_vals.min()), max(actual_vals.max(), gen_vals.max()))
    ax.hist(actual_vals, bins=60, range=rng, density=True, alpha=0.45, color="#111827", label="Actual")
    ax.hist(gen_vals, bins=60, range=rng, density=True, alpha=0.45, color=COLORS[CHANNELS[c]], label="DM")
    ax.set_title(CHANNELS[c], fontsize=10)
    style_axes(ax)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Actual vs DM Generated Distribution", fontsize=13, y=1.05)
save(FIG_DIR / "fig7_distribution_comparison.png")

manifest = {
    "sample_idx": sample_idx,
    "median_forecast_mae": float(np.median(forecast_mae)),
    "selected_forecast_mae": float(forecast_mae[sample_idx]),
    "figures": sorted(str(p).replace("\\", "/") for p in FIG_DIR.glob("*.png")),
}
with open(OUT_DIR / "figure_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
