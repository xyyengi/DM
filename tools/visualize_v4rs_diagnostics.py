#!/usr/bin/env python
"""Create forward-noise and ensemble-envelope diagnostics for a saved run."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("Wind", "Solar", "Load")
COLORS = ("#2a6fbb", "#e69f00", "#8f55a6")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_arrays(run_dir: Path) -> dict[str, np.ndarray]:
    arrays = {
        "scenarios": np.load(run_dir / "actual_scenarios.npy", mmap_mode="r"),
        "actual": np.load(run_dir / "actual_data.npy", mmap_mode="r"),
        "forecast": np.load(run_dir / "forecast_data.npy", mmap_mode="r"),
        "actual_norm": np.load(run_dir / "actual_data_normalized.npy", mmap_mode="r"),
        "forecast_norm": np.load(run_dir / "forecast_data_normalized.npy", mmap_mode="r"),
    }
    expected = (arrays["actual"].shape[0], 3, arrays["actual"].shape[-1])
    if arrays["actual"].shape != expected or arrays["forecast"].shape != expected:
        raise ValueError(f"Unexpected actual/forecast shapes: {arrays['actual'].shape}, {arrays['forecast'].shape}")
    if arrays["scenarios"].shape[0] != expected[0] or arrays["scenarios"].shape[2:] != expected[1:]:
        raise ValueError(f"Unexpected scenario shape: {arrays['scenarios'].shape}")
    return arrays


def schedule(num_steps: int, beta_start: float, beta_end: float) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
    alpha_hat = np.cumprod(1.0 - beta)
    return beta, alpha_hat


def save_snr_plot(alpha_hat: np.ndarray, output: Path) -> None:
    steps = np.arange(len(alpha_hat))
    snr = alpha_hat / np.maximum(1.0 - alpha_hat, 1e-15)
    signal_amp = np.sqrt(alpha_hat)
    noise_amp = np.sqrt(1.0 - alpha_hat)
    marked = [0, 49, 99, 249, len(alpha_hat) - 1]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].semilogy(steps, snr, color=COLORS[0], lw=2)
    axes[0].axhline(1.0, color="0.45", ls="--", lw=1)
    axes[0].set_ylabel("SNR (log scale)")
    axes[0].set_title("Diffusion signal-to-noise schedule")
    axes[0].grid(alpha=0.25)
    for t in marked:
        axes[0].scatter(t, snr[t], color=COLORS[0], s=26, zorder=3)
        axes[0].annotate(f"t={t}\n{snr[t]:.2g}", (t, snr[t]), xytext=(5, 7), textcoords="offset points", fontsize=8)

    axes[1].plot(steps, signal_amp, label="signal amplitude", color=COLORS[0], lw=2)
    axes[1].plot(steps, noise_amp, label="noise amplitude", color=COLORS[1], lw=2)
    axes[1].set_xlabel("Diffusion timestep")
    axes[1].set_ylabel("Coefficient")
    axes[1].set_ylim(-0.02, 1.03)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def choose_forward_window(actual_norm: np.ndarray, forecast_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> int:
    residual_internal = forecast_norm - actual_norm
    z = (residual_internal - mean.reshape(1, 3, 1)) / std.reshape(1, 3, 1)
    rms = np.sqrt(np.mean(z * z, axis=(1, 2)))
    return int(np.argsort(rms)[len(rms) // 2])


def save_forward_noise_plots(
    actual_norm: np.ndarray,
    forecast_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    scales: np.ndarray,
    alpha_hat: np.ndarray,
    output_dir: Path,
    seed: int,
) -> int:
    window_id = choose_forward_window(actual_norm, forecast_norm, mean, std)
    residual_internal = np.asarray(forecast_norm[window_id] - actual_norm[window_id], dtype=np.float64)
    clean = (residual_internal - mean.reshape(3, 1)) / std.reshape(3, 1)
    rng = np.random.default_rng(seed)
    epsilon = rng.standard_normal(clean.shape)
    stages: list[tuple[str, np.ndarray]] = [("clean", clean)]
    for t in (0, 49, 99, 249, len(alpha_hat) - 1):
        xt = np.sqrt(alpha_hat[t]) * clean + np.sqrt(1.0 - alpha_hat[t]) * epsilon
        stages.append((f"t={t}", xt))

    hours = np.arange(clean.shape[-1])
    for space in ("model", "physical"):
        fig, axes = plt.subplots(3, len(stages), figsize=(18, 8), sharex=True, constrained_layout=True)
        for channel in range(3):
            model_limit = max(3.0, np.percentile(np.abs(np.concatenate([x[channel] for _, x in stages])), 99))
            physical_series = [((x[channel] * std[channel]) + mean[channel]) * scales[channel] for _, x in stages]
            physical_limit = max(1.0, np.percentile(np.abs(np.concatenate(physical_series)), 99))
            for col, (label, values) in enumerate(stages):
                y = values[channel]
                if space == "physical":
                    y = (y * std[channel] + mean[channel]) * scales[channel]
                ax = axes[channel, col]
                ax.plot(hours, y, color=COLORS[channel], lw=0.8)
                ax.axhline(0, color="0.6", lw=0.6)
                ax.grid(alpha=0.18)
                ax.set_ylim(((-model_limit, model_limit) if space == "model" else (-physical_limit, physical_limit)))
                if channel == 0:
                    ax.set_title(label)
                if col == 0:
                    unit = "standardized" if space == "model" else "MW"
                    ax.set_ylabel(f"{CHANNELS[channel]}\n{unit}")
                if channel == 2:
                    ax.set_xlabel("Lead hour")
        title_space = "standardized model space" if space == "model" else "mapped back to residual MW"
        fig.suptitle(f"Forward noising of representative window {window_id}: {title_space}", fontsize=13)
        fig.savefig(output_dir / f"forward_noise_{space}_space.png", dpi=170)
        plt.close(fig)
    return window_id


def interval(scenarios: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - level) / 2.0
    return np.quantile(scenarios, tail, axis=1), np.quantile(scenarios, 1.0 - tail, axis=1)


def select_windows(actual: np.ndarray, forecast: np.ndarray, scenarios: np.ndarray) -> list[tuple[str, int]]:
    low90, high90 = interval(scenarios, 0.90)
    point_covered = (actual >= low90) & (actual <= high90)
    window_coverage = np.mean(point_covered, axis=(1, 2))
    load_coverage = np.mean(point_covered[:, 2, :], axis=1)
    load_mean = np.mean(scenarios[:, :, 2, :], axis=1)
    load_rmse = np.sqrt(np.mean((load_mean - actual[:, 2, :]) ** 2, axis=1))
    median_target = np.median(window_coverage)
    median_order = np.argsort(np.abs(window_coverage - median_target))
    worst_order = np.lexsort((-load_rmse, load_coverage))
    high_load_order = np.argsort(np.max(actual[:, 2, :], axis=1))[::-1]
    net_load = actual[:, 2, :] - actual[:, 0, :] - actual[:, 1, :]
    ramp6 = np.max(np.abs(net_load[:, 6:] - net_load[:, :-6]), axis=1)
    ramp_order = np.argsort(ramp6)[::-1]

    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for label, order in (
        ("Median 90% coverage", median_order),
        ("Worst load coverage", worst_order),
        ("Highest actual load", high_load_order),
        ("Largest 6h net-load ramp", ramp_order),
    ):
        chosen = next(int(x) for x in order if int(x) not in used)
        used.add(chosen)
        selected.append((label, chosen))
    return selected


def save_envelope_plot(
    actual: np.ndarray,
    forecast: np.ndarray,
    scenarios: np.ndarray,
    selected: list[tuple[str, int]],
    output: Path,
    test_start: datetime,
    show_members: bool = False,
) -> list[dict]:
    low90, high90 = interval(scenarios, 0.90)
    median = np.median(scenarios, axis=1)
    hours = np.arange(actual.shape[-1])
    records = []
    fig, axes = plt.subplots(len(selected), 3, figsize=(18, 13), sharex=True, constrained_layout=True)
    for row, (criterion, window_id) in enumerate(selected):
        start = test_start + timedelta(hours=window_id)
        row_record = {"criterion": criterion, "window_id": window_id, "window_start": start.isoformat()}
        for channel in range(3):
            ax = axes[row, channel]
            if show_members:
                for member in range(scenarios.shape[1]):
                    ax.plot(hours, scenarios[window_id, member, channel], color="0.55", alpha=0.045, lw=0.45)
            ax.fill_between(hours, low90[window_id, channel], high90[window_id, channel], color=COLORS[channel], alpha=0.20, label="90% envelope")
            ax.plot(hours, median[window_id, channel], color=COLORS[channel], lw=1.2, label="scenario median")
            ax.plot(hours, actual[window_id, channel], color="black", lw=1.15, label="actual")
            ax.plot(hours, forecast[window_id, channel], color="0.4", ls="--", lw=0.9, label="forecast")
            covered = np.mean((actual[window_id, channel] >= low90[window_id, channel]) & (actual[window_id, channel] <= high90[window_id, channel]))
            row_record[f"{CHANNELS[channel].lower()}_coverage90"] = float(covered)
            ax.set_title(f"{CHANNELS[channel]} | coverage={covered * 100:.1f}%")
            ax.grid(alpha=0.18)
            if channel == 0:
                ax.set_ylabel(f"{criterion}\nwindow {window_id}\nMW")
            if row == len(selected) - 1:
                ax.set_xlabel("Lead hour")
            if row == 0 and channel == 2:
                ax.legend(frameon=False, fontsize=8, ncol=2)
        records.append(row_record)
    suffix = "with all individual members" if show_members else "central 90% envelope"
    fig.suptitle(f"V4-RS posterior scenarios: {suffix}", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return records


def save_coverage_by_lead(actual: np.ndarray, scenarios: np.ndarray, output: Path) -> dict:
    low90, high90 = interval(scenarios, 0.90)
    covered = (actual >= low90) & (actual <= high90)
    hourly = np.mean(covered, axis=0)
    kernel = np.ones(24) / 24.0
    summary = {}
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for channel, ax in enumerate(axes):
        rolling = np.convolve(hourly[channel], kernel, mode="valid")
        x_roll = np.arange(23, actual.shape[-1])
        ax.plot(np.arange(actual.shape[-1]), hourly[channel] * 100, color="0.7", lw=0.7, label="hourly")
        ax.plot(x_roll, rolling * 100, color=COLORS[channel], lw=2, label="24h rolling mean")
        ax.axhline(90, color="0.3", ls="--", lw=1, label="nominal 90%")
        ax.set_ylabel(f"{CHANNELS[channel]}\ncoverage (%)")
        ax.set_ylim(40, 101)
        ax.grid(alpha=0.2)
        summary[CHANNELS[channel].lower()] = {
            "minimum_hourly_coverage": float(np.min(hourly[channel])),
            "minimum_24h_coverage": float(np.min(rolling)),
            "minimum_24h_start_lead": int(np.argmin(rolling)),
        }
        if channel == 0:
            ax.legend(frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Lead hour within 168h window")
    fig.suptitle("90% envelope coverage by forecast lead")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def save_coverage_by_error(actual: np.ndarray, forecast: np.ndarray, scenarios: np.ndarray, output: Path) -> dict:
    low90, high90 = interval(scenarios, 0.90)
    covered = (actual >= low90) & (actual <= high90)
    width = high90 - low90
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for channel, ax in enumerate(axes):
        error = np.abs(actual[:, channel, :] - forecast[:, channel, :]).ravel()
        cov = covered[:, channel, :].ravel()
        wid = width[:, channel, :].ravel()
        edges = np.quantile(error, np.linspace(0, 1, 11))
        edges = np.maximum.accumulate(edges)
        centers, coverage_values, width_values, counts = [], [], [], []
        for idx in range(10):
            if idx == 9:
                mask = (error >= edges[idx]) & (error <= edges[idx + 1])
            else:
                mask = (error >= edges[idx]) & (error < edges[idx + 1])
            centers.append((idx + 0.5) * 10)
            coverage_values.append(float(np.mean(cov[mask])) if np.any(mask) else np.nan)
            width_values.append(float(np.mean(wid[mask])) if np.any(mask) else np.nan)
            counts.append(int(np.sum(mask)))
        ax.plot(centers, np.asarray(coverage_values) * 100, marker="o", color=COLORS[channel], lw=2)
        ax.axhline(90, color="0.3", ls="--", lw=1)
        ax.set_title(CHANNELS[channel])
        ax.set_xlabel("Absolute forecast-error decile (%)")
        ax.set_ylabel("90% envelope coverage (%)")
        ax.set_ylim(0, 101)
        ax.grid(alpha=0.2)
        summary[CHANNELS[channel].lower()] = {
            "coverage_by_error_decile": coverage_values,
            "mean_width_mw_by_error_decile": width_values,
            "counts": counts,
            "error_edges_mw": edges.tolist(),
        }
    fig.suptitle("Does the envelope widen where forecast error is large?")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def save_window_coverage_distribution(actual: np.ndarray, scenarios: np.ndarray, output: Path) -> dict:
    low90, high90 = interval(scenarios, 0.90)
    covered = (actual >= low90) & (actual <= high90)
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    bins = np.linspace(0, 1, 21)
    for channel, ax in enumerate(axes):
        window_cov = np.mean(covered[:, channel, :], axis=1)
        ax.hist(window_cov, bins=bins, color=COLORS[channel], alpha=0.75)
        ax.axvline(0.90, color="0.25", ls="--", lw=1.2)
        ax.axvline(np.median(window_cov), color=COLORS[channel], lw=2)
        ax.set_title(f"{CHANNELS[channel]} | median={np.median(window_cov) * 100:.1f}%")
        ax.set_xlabel("Per-window 90% coverage")
        ax.set_ylabel("Number of windows")
        ax.grid(axis="y", alpha=0.2)
        summary[CHANNELS[channel].lower()] = {
            "p10": float(np.quantile(window_cov, 0.10)),
            "median": float(np.median(window_cov)),
            "p90": float(np.quantile(window_cov, 0.90)),
            "fraction_below_80pct": float(np.mean(window_cov < 0.80)),
        }
    fig.suptitle("Distribution of scenario-envelope coverage across 577 windows")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def save_rank_histogram(actual: np.ndarray, scenarios: np.ndarray, output: Path) -> dict:
    members = scenarios.shape[1]
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for channel, ax in enumerate(axes):
        values = scenarios[:, :, channel, :]
        ranks = np.sum(values < actual[:, None, channel, :], axis=1).ravel()
        counts = np.bincount(ranks, minlength=members + 1)
        expected = len(ranks) / (members + 1)
        ax.bar(np.arange(members + 1), counts / expected, color=COLORS[channel], width=0.85)
        ax.axhline(1.0, color="0.25", ls="--", lw=1)
        ax.set_title(CHANNELS[channel])
        ax.set_xlabel("Rank of actual among 50 members")
        ax.set_ylabel("Count / uniform expectation")
        ax.grid(axis="y", alpha=0.2)
        summary[CHANNELS[channel].lower()] = {
            "below_all_fraction": float(counts[0] / len(ranks)),
            "above_all_fraction": float(counts[-1] / len(ranks)),
            "edge_fraction": float((counts[0] + counts[-1]) / len(ranks)),
        }
    fig.suptitle("Rank histograms: U-shape indicates underdispersion")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return summary


def ensemble_summary(actual: np.ndarray, forecast: np.ndarray, scenarios: np.ndarray) -> list[dict]:
    records = []
    ensemble_mean = np.mean(scenarios, axis=1)
    for channel, name in enumerate(CHANNELS):
        record = {"channel": name.lower()}
        for level in (0.80, 0.90, 0.95):
            low, high = interval(scenarios[:, :, channel : channel + 1, :], level)
            a = actual[:, channel : channel + 1, :]
            record[f"coverage_{int(level * 100)}"] = float(np.mean((a >= low) & (a <= high)))
            record[f"mean_width_mw_{int(level * 100)}"] = float(np.mean(high - low))
        mean_error = ensemble_mean[:, channel, :] - actual[:, channel, :]
        within_spread = np.sqrt(np.mean(np.var(scenarios[:, :, channel, :], axis=1)))
        mean_rmse = np.sqrt(np.mean(mean_error * mean_error))
        record.update(
            {
                "ensemble_mean_bias_mw": float(np.mean(mean_error)),
                "ensemble_mean_rmse_mw": float(mean_rmse),
                "within_ensemble_rms_spread_mw": float(within_spread),
                "spread_skill_ratio": float(within_spread / mean_rmse),
                "forecast_mae_mw": float(np.mean(np.abs(forecast[:, channel, :] - actual[:, channel, :]))),
                "actual_min_mw": float(np.min(actual[:, channel, :])),
                "actual_max_mw": float(np.max(actual[:, channel, :])),
                "scenario_min_mw": float(np.min(scenarios[:, :, channel, :])),
                "scenario_p001_mw": float(np.quantile(scenarios[:, :, channel, :], 0.001)),
                "scenario_p999_mw": float(np.quantile(scenarios[:, :, channel, :], 0.999)),
                "scenario_max_mw": float(np.max(scenarios[:, :, channel, :])),
                "scenario_negative_fraction": float(np.mean(scenarios[:, :, channel, :] < 0.0)),
            }
        )
        records.append(record)
    return records


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-start", default="2025-12-01T00:00:00")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args.run_dir)
    denorm = load_json(args.run_dir / "denormalization_used.json")
    standardizer = load_json(args.train_dir / "residual_standardization.json")
    generation_config = (args.run_dir / "generation_config_used.yaml").read_text(encoding="utf-8")
    if not standardizer.get("enabled"):
        raise ValueError("This diagnostic expects residual standardization to be enabled")
    mean = np.asarray(standardizer["mean"], dtype=np.float64)
    std = np.asarray(standardizer["std"], dtype=np.float64)
    scales = np.asarray(denorm["scales"], dtype=np.float64)
    num_steps = 500
    beta_start = 0.0001
    beta_end = 0.04
    for line in generation_config.splitlines():
        stripped = line.strip()
        if stripped.startswith("num_steps:"):
            num_steps = int(stripped.split(":", 1)[1])
        elif stripped.startswith("beta_start:"):
            beta_start = float(stripped.split(":", 1)[1])
        elif stripped.startswith("beta_end:"):
            beta_end = float(stripped.split(":", 1)[1])
    _, alpha_hat = schedule(num_steps, beta_start, beta_end)

    save_snr_plot(alpha_hat, args.output_dir / "snr_schedule.png")
    representative_window = save_forward_noise_plots(
        arrays["actual_norm"], arrays["forecast_norm"], mean, std, scales, alpha_hat, args.output_dir, args.seed
    )
    selected = select_windows(arrays["actual"], arrays["forecast"], arrays["scenarios"])
    selected_records = save_envelope_plot(
        arrays["actual"], arrays["forecast"], arrays["scenarios"], selected,
        args.output_dir / "scenario_envelopes.png", datetime.fromisoformat(args.test_start), show_members=False
    )
    save_envelope_plot(
        arrays["actual"], arrays["forecast"], arrays["scenarios"], selected,
        args.output_dir / "scenario_members_with_outliers.png", datetime.fromisoformat(args.test_start), show_members=True
    )
    coverage_lead = save_coverage_by_lead(
        arrays["actual"], arrays["scenarios"], args.output_dir / "coverage_by_lead_hour.png"
    )
    coverage_error = save_coverage_by_error(
        arrays["actual"], arrays["forecast"], arrays["scenarios"], args.output_dir / "coverage_by_error_magnitude.png"
    )
    window_distribution = save_window_coverage_distribution(
        arrays["actual"], arrays["scenarios"], args.output_dir / "window_coverage_distribution.png"
    )
    rank_summary = save_rank_histogram(
        arrays["actual"], arrays["scenarios"], args.output_dir / "rank_histograms.png"
    )
    summary = ensemble_summary(arrays["actual"], arrays["forecast"], arrays["scenarios"])
    write_csv(args.output_dir / "channel_envelope_summary.csv", summary)
    write_csv(args.output_dir / "selected_window_envelopes.csv", selected_records)
    diagnostics = {
        "run_dir": str(args.run_dir),
        "train_dir": str(args.train_dir),
        "shape": {name: list(value.shape) for name, value in arrays.items()},
        "schedule": {
            "num_steps": num_steps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "alpha_hat_last": float(alpha_hat[-1]),
        },
        "forward_noise_representative_window": representative_window,
        "selected_windows": selected_records,
        "channel_summary": summary,
        "coverage_by_lead": coverage_lead,
        "coverage_by_error_magnitude": coverage_error,
        "window_coverage_distribution": window_distribution,
        "rank_histogram": rank_summary,
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "representative_window": representative_window,
        "selected_windows": selected_records,
        "channel_summary": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
