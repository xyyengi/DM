#!/usr/bin/env python
"""Audit timestep denoising and condition usage of an existing checkpoint.

This tool never trains or updates the model. It uses a fixed validation subset,
fixed forward noise, and identical noisy targets across condition ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_multivariate import get_dataloader_multivariate
from diff_models_multivariate import MultiChannelCSDI
from generate import apply_experiment_switches, load_denormalization_scales


CHANNELS = ("wind", "solar", "load")
COLORS = ("#2a6fbb", "#e69f00", "#8f55a6")
MODES = (
    "normal",
    "forecast_shuffled",
    "forecast_zero",
    "time_shuffled",
    "time_zero",
    "forecast_time_zero",
)
DEFAULT_TIMESTEPS = (10, 40, 70, 100, 160, 220, 270, 320, 390, 470)
TIME_BINS = (
    ("0-49", 0, 49),
    ("50-129", 50, 129),
    ("130-242", 130, 242),
    ("243-336", 243, 336),
    ("337-499", 337, 499),
)


def load_config(train_dir: Path) -> dict:
    with (train_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
        return apply_experiment_switches(yaml.safe_load(handle))


def load_model(train_dir: Path, config: dict, device: torch.device) -> tuple[MultiChannelCSDI, dict, Path]:
    checkpoint_path = train_dir / "checkpoints" / "model_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = MultiChannelCSDI(config["model"], device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {"diffusion.beta", "diffusion.alpha", "diffusion.alpha_hat"}
    real_missing = [key for key in missing if key not in allowed_missing]
    if real_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={real_missing}, unexpected={unexpected}")
    model.eval()
    return model, checkpoint, checkpoint_path


def make_validation_loader(data_path: Path, config: dict, batch_size: int, max_windows: int, seed: int):
    loader, _, max_values = get_dataloader_multivariate(
        str(data_path),
        batch_size=batch_size,
        mode="val",
        n_intervals=config["model"]["n_intervals"],
        build_kde=False,
        residual_standardization=config.get("target", {}).get(
            "residual_standardization", {"enabled": False}
        ),
    )
    dataset = loader.dataset
    count = min(max_windows, len(dataset))
    rng = np.random.default_rng(seed)
    # Keep the random order. Sorting would put highly overlapping adjacent
    # 168h windows in the same batch and make shuffled-condition ablations too weak.
    indices = rng.choice(len(dataset), size=count, replace=False)
    subset_loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return subset_loader, dataset, max_values, indices


def condition_variants(
    forecast: torch.Tensor,
    time_encoding: torch.Tensor,
    permutation: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    zeros_forecast = torch.zeros_like(forecast)
    zeros_time = torch.zeros_like(time_encoding)
    return [
        (forecast, time_encoding),
        (forecast[permutation], time_encoding),
        (zeros_forecast, time_encoding),
        (forecast, time_encoding[permutation]),
        (forecast, zeros_time),
        (zeros_forecast, zeros_time),
    ]


def bin_name(timestep: int) -> str:
    for label, low, high in TIME_BINS:
        if low <= timestep <= high:
            return label
    return "other"


def audit_timesteps(
    model: MultiChannelCSDI,
    loader: DataLoader,
    timesteps: tuple[int, ...],
    residual_std: np.ndarray,
    physical_scales: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    accum = defaultdict(lambda: {"eps_sq": np.zeros(3), "x0_sq": np.zeros(3), "x0_sum": np.zeros(3), "n": 0})
    baseline = defaultdict(lambda: {"eps_sq": np.zeros(3), "x0_sq": np.zeros(3), "n": 0})

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            x0 = model._select_target(batch)
            forecast = batch["forecast_3ch"].to(device)
            time_encoding = batch["time_encoding"].to(device)
            timepoints = batch["timepoints"].to(device)
            batch_size = x0.shape[0]
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + batch_index * 10007)
            permutation = torch.randperm(batch_size, generator=generator, device=device)
            if batch_size > 1 and torch.equal(permutation, torch.arange(batch_size, device=device)):
                permutation = torch.roll(permutation, 1)
            base_time_feat = model.get_time_features(timepoints)

            for timestep in timesteps:
                generator.manual_seed(seed + batch_index * 10007 + timestep * 97)
                noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
                alpha_hat = model.diffusion.alpha_hat[timestep]
                sqrt_alpha = alpha_hat.sqrt()
                sqrt_noise = (1.0 - alpha_hat).sqrt()
                x_t = sqrt_alpha * x0 + sqrt_noise * noise

                variants = condition_variants(forecast, time_encoding, permutation)
                x_stack = torch.cat([x_t for _ in MODES], dim=0)
                forecast_stack = torch.cat([pair[0] for pair in variants], dim=0)
                time_stack = torch.cat([pair[1] for pair in variants], dim=0)
                time_feat_stack = torch.cat([base_time_feat for _ in MODES], dim=0)
                model_input = model.build_model_input(
                    x_stack,
                    forecast_3ch=forecast_stack if model.use_forecast else None,
                    time_encoding=time_stack,
                )
                epsilon_hat = model.diffusion.model(model_input, time_feat_stack).reshape(
                    len(MODES), batch_size, 3, 168
                )
                target_noise = noise.unsqueeze(0)
                target_x0 = x0.unsqueeze(0)
                x0_hat = (x_t.unsqueeze(0) - sqrt_noise * epsilon_hat) / sqrt_alpha.clamp_min(1e-12)

                for mode_index, mode in enumerate(MODES):
                    eps_error = epsilon_hat[mode_index] - noise
                    x0_error = x0_hat[mode_index] - x0
                    key = (timestep, mode)
                    accum[key]["eps_sq"] += torch.sum(eps_error * eps_error, dim=(0, 2)).cpu().numpy()
                    accum[key]["x0_sq"] += torch.sum(x0_error * x0_error, dim=(0, 2)).cpu().numpy()
                    accum[key]["x0_sum"] += torch.sum(x0_error, dim=(0, 2)).cpu().numpy()
                    accum[key]["n"] += batch_size * 168

                base_x0_hat = x_t / sqrt_alpha.clamp_min(1e-12)
                base_x0_error = base_x0_hat - x0
                baseline[timestep]["eps_sq"] += torch.sum(noise * noise, dim=(0, 2)).cpu().numpy()
                baseline[timestep]["x0_sq"] += torch.sum(base_x0_error * base_x0_error, dim=(0, 2)).cpu().numpy()
                baseline[timestep]["n"] += batch_size * 168

    rows = []
    for timestep in timesteps:
        for mode in MODES:
            item = accum[(timestep, mode)]
            base = baseline[timestep]
            for channel_index, channel in enumerate(CHANNELS):
                eps_mse = item["eps_sq"][channel_index] / item["n"]
                x0_mse = item["x0_sq"][channel_index] / item["n"]
                base_eps_mse = base["eps_sq"][channel_index] / base["n"]
                base_x0_mse = base["x0_sq"][channel_index] / base["n"]
                mw_factor = residual_std[channel_index] * physical_scales[channel_index]
                rows.append({
                    "timestep": timestep,
                    "time_bin": bin_name(timestep),
                    "mode": mode,
                    "channel": channel,
                    "epsilon_mse": float(eps_mse),
                    "epsilon_mse_vs_zero_baseline": float(eps_mse / base_eps_mse),
                    "x0_rmse_standardized": float(np.sqrt(x0_mse)),
                    "x0_rmse_mw": float(np.sqrt(x0_mse) * mw_factor),
                    "x0_bias_mw": float(item["x0_sum"][channel_index] / item["n"] * mw_factor),
                    "x0_mse_vs_zero_baseline": float(x0_mse / base_x0_mse),
                })

    bin_rows = []
    for label, _, _ in TIME_BINS:
        for mode in MODES:
            for channel in CHANNELS:
                selected = [row for row in rows if row["time_bin"] == label and row["mode"] == mode and row["channel"] == channel]
                if not selected:
                    continue
                normal = [row for row in rows if row["time_bin"] == label and row["mode"] == "normal" and row["channel"] == channel]
                eps = float(np.mean([row["epsilon_mse"] for row in selected]))
                normal_eps = float(np.mean([row["epsilon_mse"] for row in normal]))
                bin_rows.append({
                    "time_bin": label,
                    "mode": mode,
                    "channel": channel,
                    "epsilon_mse": eps,
                    "epsilon_mse_ratio_to_normal": eps / normal_eps,
                    "x0_rmse_mw": float(np.mean([row["x0_rmse_mw"] for row in selected])),
                    "x0_bias_mw": float(np.mean([row["x0_bias_mw"] for row in selected])),
                    "epsilon_mse_vs_zero_baseline": float(np.mean([row["epsilon_mse_vs_zero_baseline"] for row in selected])),
                })
    return rows, bin_rows


def reverse_trajectory(
    model: MultiChannelCSDI,
    dataset,
    dataset_index: int,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    physical_scales: np.ndarray,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict:
    batch = {key: value.unsqueeze(0) for key, value in dataset[dataset_index].items()}
    forecast = batch["forecast_3ch"].to(device)
    time_encoding = batch["time_encoding"].to(device)
    timepoints = batch["timepoints"].to(device)
    actual = batch["actual_3ch"].numpy()[0]
    time_feat = model.get_time_features(timepoints)
    capture_before = {499, 337, 243, 130, 50}
    states = {}
    torch.manual_seed(seed)
    x_t = torch.randn(1, 3, 168, device=device)
    with torch.no_grad():
        for timestep in range(model.diffusion.num_steps - 1, -1, -1):
            if timestep in capture_before:
                states[timestep] = x_t.detach().cpu().numpy()[0]
            model_input = model.build_model_input(x_t, forecast, time_encoding)
            x_t, _ = model.diffusion.denoise_step(
                x_t, timestep, model_input, time_feat,
                cond_matrix=None, forecast=forecast, debug=False,
            )
    states[0] = x_t.detach().cpu().numpy()[0]

    order = [499, 337, 243, 130, 50, 0]
    hours = np.arange(168)
    fig, axes = plt.subplots(3, len(order), figsize=(18, 8), sharex=True, constrained_layout=True)
    for channel in range(3):
        limit = max(3.0, np.percentile(np.abs(np.concatenate([states[t][channel] for t in order])), 99))
        for col, timestep in enumerate(order):
            axes[channel, col].plot(hours, states[timestep][channel], color=COLORS[channel], lw=0.75)
            axes[channel, col].axhline(0, color="0.6", lw=0.6)
            axes[channel, col].set_ylim(-limit, limit)
            axes[channel, col].grid(alpha=0.18)
            if channel == 0:
                axes[channel, col].set_title(f"t={timestep}")
            if col == 0:
                axes[channel, col].set_ylabel(f"{CHANNELS[channel]}\nstandardized")
            if channel == 2:
                axes[channel, col].set_xlabel("Lead hour")
    fig.suptitle(f"Reverse trajectory in model space (validation window {dataset_index})")
    fig.savefig(output_dir / "reverse_trajectory_model_space.png", dpi=180)
    plt.close(fig)

    forecast_np = forecast.cpu().numpy()[0]
    forecast_mw = forecast_np * physical_scales.reshape(3, 1)
    actual_mw = actual * physical_scales.reshape(3, 1)
    fig, axes = plt.subplots(3, len(order), figsize=(18, 8), sharex=True, constrained_layout=True)
    for channel in range(3):
        all_estimates = []
        for timestep in order:
            residual_norm = states[timestep][channel] * residual_std[channel] + residual_mean[channel]
            all_estimates.append(forecast_mw[channel] - residual_norm * physical_scales[channel])
        lower = min(np.percentile(np.concatenate(all_estimates), 1), np.min(actual_mw[channel]))
        upper = max(np.percentile(np.concatenate(all_estimates), 99), np.max(actual_mw[channel]))
        margin = max(1.0, 0.05 * (upper - lower))
        for col, timestep in enumerate(order):
            axes[channel, col].plot(hours, all_estimates[col], color=COLORS[channel], lw=0.8, label="estimated actual")
            axes[channel, col].plot(hours, actual_mw[channel], color="black", lw=0.8, label="actual")
            axes[channel, col].plot(hours, forecast_mw[channel], color="0.45", ls="--", lw=0.7, label="forecast")
            axes[channel, col].set_ylim(lower - margin, upper + margin)
            axes[channel, col].grid(alpha=0.18)
            if channel == 0:
                axes[channel, col].set_title(f"t={timestep}")
            if col == 0:
                axes[channel, col].set_ylabel(f"{CHANNELS[channel]}\nMW")
            if channel == 2:
                axes[channel, col].set_xlabel("Lead hour")
            if channel == 0 and col == len(order) - 1:
                axes[channel, col].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Reverse trajectory mapped to actual power (validation window {dataset_index})")
    fig.savefig(output_dir / "reverse_trajectory_actual_space.png", dpi=180)
    plt.close(fig)
    return {"dataset_index": dataset_index, "captured_timesteps": order}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostics(rows: list[dict], bin_rows: list[dict], output_dir: Path) -> None:
    normal = [row for row in rows if row["mode"] == "normal"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for channel_index, channel in enumerate(CHANNELS):
        data = sorted([row for row in normal if row["channel"] == channel], key=lambda row: row["timestep"])
        t = [row["timestep"] for row in data]
        axes[0, channel_index].plot(t, [row["epsilon_mse_vs_zero_baseline"] for row in data], marker="o", color=COLORS[channel_index])
        axes[0, channel_index].axhline(1.0, color="0.35", ls="--", lw=1)
        axes[0, channel_index].set_title(channel)
        axes[0, channel_index].set_ylabel("epsilon MSE / zero baseline")
        axes[0, channel_index].grid(alpha=0.2)
        axes[1, channel_index].semilogy(t, [row["x0_rmse_mw"] for row in data], marker="o", color=COLORS[channel_index])
        axes[1, channel_index].set_xlabel("Diffusion timestep")
        axes[1, channel_index].set_ylabel("x0 reconstruction RMSE (MW)")
        axes[1, channel_index].grid(alpha=0.2)
    fig.suptitle("Checkpoint denoising quality by diffusion timestep")
    fig.savefig(output_dir / "timestep_denoising_quality.png", dpi=180)
    plt.close(fig)

    ablation_modes = [mode for mode in MODES if mode != "normal"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    x = np.arange(len(TIME_BINS))
    for channel_index, channel in enumerate(CHANNELS):
        ax = axes[channel_index]
        for mode in ablation_modes:
            values = []
            for label, _, _ in TIME_BINS:
                item = next(row for row in bin_rows if row["time_bin"] == label and row["mode"] == mode and row["channel"] == channel)
                values.append(item["epsilon_mse_ratio_to_normal"])
            ax.plot(x, values, marker="o", lw=1.4, label=mode.replace("_", " "))
        ax.axhline(1.0, color="0.3", ls="--", lw=1)
        ax.set_xticks(x, [item[0] for item in TIME_BINS], rotation=25)
        ax.set_title(channel)
        ax.set_xlabel("Diffusion timestep bin")
        ax.set_ylabel("MSE / normal-condition MSE")
        ax.grid(alpha=0.2)
        if channel_index == 2:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Condition ablation: values above 1 mean the condition helps")
    fig.savefig(output_dir / "condition_ablation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("diffusion_npy_normalized"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timesteps", type=int, nargs="+", default=list(DEFAULT_TIMESTEPS))
    parser.add_argument("--trajectory-index", type=int, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.train_dir)
    model, checkpoint, checkpoint_path = load_model(args.train_dir, config, device)
    loader, dataset, max_values, indices = make_validation_loader(
        args.data_path, config, args.batch_size, args.max_windows, args.seed
    )
    standardizer = dataset.residual_standardizer
    if not standardizer or not standardizer.get("enabled"):
        raise ValueError("Residual standardization stats are required for MW diagnostics")
    residual_mean = np.asarray(standardizer["mean"], dtype=np.float64)
    residual_std = np.asarray(standardizer["std"], dtype=np.float64)
    physical_scales, scale_source = load_denormalization_scales(str(args.data_path), max_values)
    timesteps = tuple(sorted(set(args.timesteps)))
    if not timesteps or timesteps[0] < 0 or timesteps[-1] >= model.diffusion.num_steps:
        raise ValueError(f"Timesteps must lie in [0, {model.diffusion.num_steps - 1}]")

    rows, bin_rows = audit_timesteps(
        model, loader, timesteps, residual_std, physical_scales, device, args.seed
    )
    write_csv(args.output_dir / "timestep_condition_metrics.csv", rows)
    write_csv(args.output_dir / "timestep_bin_condition_summary.csv", bin_rows)
    plot_diagnostics(rows, bin_rows, args.output_dir)
    trajectory_index = args.trajectory_index if args.trajectory_index is not None else int(indices[len(indices) // 2])
    trajectory = reverse_trajectory(
        model, dataset, trajectory_index, residual_mean, residual_std,
        physical_scales, device, args.seed, args.output_dir,
    )

    report = {
        "diagnostic_only": True,
        "trained_or_updated": False,
        "split": "val",
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "n_validation_windows": int(len(indices)),
        "validation_indices": indices.tolist(),
        "timesteps": list(timesteps),
        "condition_modes": list(MODES),
        "residual_standardization": standardizer,
        "physical_scales": physical_scales.tolist(),
        "physical_scale_source": scale_source,
        "trajectory": trajectory,
        "bin_summary": bin_rows,
    }
    (args.output_dir / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "n_validation_windows": len(indices),
        "trajectory_index": trajectory_index,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
