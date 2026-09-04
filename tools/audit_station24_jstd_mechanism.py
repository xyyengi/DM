#!/usr/bin/env python3
"""Audit what a trained JSTD tail changes, where it changes it, and why.

This is an offline, zero-training diagnostic.  Validation actual/residual values
are used only to score the learned masks and corrections; they are never passed
to the JSTD condition encoder.  The probe evaluates the trained denoiser at
fixed diffusion timesteps with deterministic noise and forces route=1 so that
the tail mechanism can be inspected independently of its issue-level gate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_station24 import move_batch, select_checkpoint_state
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import get_station_dataloader, load_station_static_data
from station_graph_prior import load_generation_graphs
from station_jstd_targets import (
    build_station_jstd_target_arrays,
    fit_station_jstd_event_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-state", choices=("raw", "ema"), default="raw")
    parser.add_argument("--timesteps", default="50,150,300,450")
    parser.add_argument("--false-positive-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    return float((comparison > 0).mean() + 0.5 * (comparison == 0).mean())


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int((labels > 0.5).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores)
    ordered = labels[order] > 0.5
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, ordered.size + 1)
    return float((cumulative[ordered] / ranks[ordered]).sum() / positives)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def _weighted_curve(value: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("s,st->t", weights, value)


def _event_localization(mask: np.ndarray, support: np.ndarray) -> dict[str, float]:
    """Score a [S,L] soft mask against station-time event support."""

    support = support.astype(bool)
    inside = mask[support]
    outside = mask[~support]
    total = float(mask.sum())
    inside_mass = float(inside.sum()) if inside.size else 0.0
    return {
        "mask_inside_mean": float(inside.mean()) if inside.size else float("nan"),
        "mask_outside_mean": float(outside.mean()) if outside.size else float("nan"),
        "mask_inside_outside_ratio": _safe_ratio(
            float(inside.mean()) if inside.size else 0.0,
            float(outside.mean()) if outside.size else 0.0,
        ),
        "mask_mass_event_fraction": _safe_ratio(inside_mass, total),
    }


def _correction_localization(
    correction: np.ndarray, support: np.ndarray
) -> dict[str, float]:
    support = support.astype(bool)
    energy = np.abs(correction)
    total = float(energy.sum())
    inside = float(energy[support].sum()) if support.any() else 0.0
    return {
        "correction_abs_mean": float(energy.mean()),
        "correction_event_energy_fraction": _safe_ratio(inside, total),
        "correction_outside_event_fraction": _safe_ratio(total - inside, total),
    }


def _plot_issue(
    output: Path,
    issue: int,
    label: str,
    forecast: np.ndarray,
    actual: np.ndarray,
    slow_mask: np.ndarray,
    fast_mask: np.ndarray,
    slow_delta_mw: np.ndarray,
    fast_delta_mw: np.ndarray,
    event_time: np.ndarray,
    wind_weights_mw: np.ndarray,
    solar_weights_mw: np.ndarray,
) -> None:
    wind_forecast = _weighted_curve(forecast, wind_weights_mw)
    wind_actual = _weighted_curve(actual, wind_weights_mw)
    solar_forecast = _weighted_curve(forecast, solar_weights_mw)
    solar_actual = _weighted_curve(actual, solar_weights_mw)
    station_weight = wind_weights_mw + solar_weights_mw
    station_weight = station_weight / max(float(station_weight.sum()), 1e-12)
    slow_mask_curve = _weighted_curve(slow_mask, station_weight)
    fast_mask_curve = _weighted_curve(fast_mask, station_weight)
    slow_system_delta = slow_delta_mw.sum(axis=0)
    fast_system_delta = fast_delta_mw.sum(axis=0)

    lead = np.arange(forecast.shape[-1])
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(lead, wind_forecast, "--", color="#009688", label="wind forecast")
    axes[0].plot(lead, wind_actual, color="#111827", label="wind actual")
    axes[0].plot(lead, solar_forecast, "--", color="#f59e0b", label="solar forecast")
    axes[0].plot(lead, solar_actual, color="#9a3412", label="solar actual")
    axes[0].set_ylabel("Aggregated MW")
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(lead, slow_mask_curve, color="#2563eb", label="slow mask")
    axes[1].plot(lead, fast_mask_curve, color="#dc2626", label="fast mask")
    axes[1].fill_between(lead, 0, event_time.astype(float), color="#fde68a", alpha=0.35,
                         label="true event support (audit only)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Mask strength")
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(lead, slow_system_delta, color="#2563eb", label="slow x0-equivalent")
    axes[2].plot(lead, fast_system_delta, color="#dc2626", label="fast x0-equivalent")
    axes[2].plot(
        lead,
        slow_system_delta + fast_system_delta,
        color="#111827",
        linewidth=1.5,
        label="combined",
    )
    axes[2].axhline(0.0, color="#9ca3af", linewidth=0.8)
    axes[2].set_ylabel("System correction MW")
    axes[2].set_xlabel("Lead hour")
    axes[2].legend(ncol=3, fontsize=8)
    fig.suptitle(f"JSTD mechanism audit: issue={issue}, {label}")
    fig.tight_layout()
    fig.savefig(output / f"jstd_mechanism_issue_{issue:02d}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    result_dir = Path(args.result_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    timesteps = sorted({int(value) for value in args.timesteps.split(",")})
    if not timesteps:
        raise ValueError("at least one diffusion timestep is required")
    if args.false_positive_count < 0:
        raise ValueError("false-positive-count must be non-negative")

    config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "checkpoints" / "model_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not bool(config["model"].get("use_jstd_tail", False)):
        raise ValueError("candidate run is not a JSTD model")
    num_steps = int(config["model"]["num_steps"])
    if min(timesteps) < 0 or max(timesteps) >= num_steps:
        raise ValueError(f"timesteps must be in [0,{num_steps - 1}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is required; use --allow-cpu only for a deliberate CPU audit")
    _set_seed(args.seed)
    static = load_station_static_data(args.data_path)
    primary, secondary, graph_manifest = load_generation_graphs(
        args.data_path, run_dir, config["model"], checkpoint
    )
    model = Station24DiffusionModel(
        config["model"],
        static["station_features"],
        primary,
        static["station_capacities"],
        secondary,
    ).to(device)
    state, state_key = select_checkpoint_state(checkpoint, args.checkpoint_state)
    model.load_state_dict(state, strict=True)
    model.eval()

    thresholds = fit_station_jstd_event_thresholds(args.data_path, config["model"])
    targets = build_station_jstd_target_arrays(args.data_path, "val", thresholds)
    loader, dataset = get_station_dataloader(
        args.data_path,
        "val",
        checkpoint["residual_scale"],
        batch_size=1,
        seed=args.seed,
        num_workers=0,
        condition_config=config["model"],
        state_thresholds=checkpoint.get("state_thresholds"),
        jstd_targets=targets,
    )
    if len(dataset) != len(targets.event_active):
        raise ValueError("validation dataset and JSTD targets disagree")

    issue_probability = np.load(result_dir / "tail_expert_probability.npy").astype(float)
    if issue_probability.shape != targets.event_active.shape:
        raise ValueError("saved issue probabilities do not match validation targets")
    event_issues = np.flatnonzero(targets.event_active > 0.5).tolist()
    non_event = np.flatnonzero(targets.event_active <= 0.5)
    false_positive_issues = non_event[
        np.argsort(-issue_probability[non_event])[: args.false_positive_count]
    ].tolist()
    selected = sorted(set(event_issues + false_positive_issues))
    catalog_by_issue = {int(item["sample_index"]): item for item in targets.catalog}

    station_features = static["station_features"].detach().cpu().numpy().astype(
        np.float32, copy=False
    )
    capacities = static["station_capacities"].detach().cpu().numpy().astype(
        np.float32, copy=False
    )
    wind_weights_mw = capacities * station_features[:, 0]
    solar_weights_mw = capacities * station_features[:, 1]
    rows: list[dict[str, object]] = []
    arrays: dict[str, list[np.ndarray]] = {
        "slow_mask": [],
        "fast_mask": [],
        "slow_epsilon_correction": [],
        "fast_epsilon_correction": [],
        "slow_x0_correction_normalized": [],
        "fast_x0_correction_normalized": [],
    }
    plot_inputs: dict[int, dict[str, np.ndarray]] = {}

    for issue, raw_batch in enumerate(loader):
        if issue not in selected:
            continue
        batch = move_batch(raw_batch, device)
        clean = batch["residual_target"]
        per_issue = {name: [] for name in arrays}
        support = batch["jstd_event_station_support"][0].detach().cpu().numpy() > 0.5
        event_time = batch["jstd_event_time_support"][0].detach().cpu().numpy() > 0.5
        generator = torch.Generator(device=device).manual_seed(args.seed + issue * 1009)
        base_noise = torch.randn(clean.shape, generator=generator, device=device)
        issue_logit = model.tail_risk_logits(batch)
        issue_prob_probe = float(torch.sigmoid(issue_logit)[0].detach().cpu())
        for step in timesteps:
            timestep = torch.full((1,), step, device=device, dtype=torch.long)
            noisy, _ = model.diffusion.add_noise(clean, timestep, noise=base_noise)
            with torch.inference_mode():
                _, audit = model.denoiser(
                    noisy,
                    timestep,
                    batch["forecast"],
                    batch["calendar"],
                    batch["lead"],
                    forecast_ramps=batch.get("forecast_ramps"),
                    forecast_revision=batch.get("forecast_revision"),
                    revision_mask=batch.get("revision_mask"),
                    recent_error=batch.get("recent_error"),
                    recent_error_mask=batch.get("recent_error_mask"),
                    node_state=batch.get("node_state"),
                    forecast_condition_strength=1.0,
                    tail_expert_route=1.0,
                    return_jstd_audit=True,
                )
            alpha_hat = model.diffusion.alpha_hat[timestep].view(1, 1, 1)
            x0_factor = -torch.sqrt((1.0 - alpha_hat) / alpha_hat.clamp(min=1e-8))
            slow_x0 = x0_factor * audit.slow_correction
            fast_x0 = x0_factor * audit.fast_correction
            values = {
                "slow_mask": audit.slow_mask[0].detach().cpu().numpy(),
                "fast_mask": audit.fast_mask[0].detach().cpu().numpy(),
                "slow_epsilon_correction": audit.slow_correction[0].detach().cpu().numpy(),
                "fast_epsilon_correction": audit.fast_correction[0].detach().cpu().numpy(),
                "slow_x0_correction_normalized": slow_x0[0].detach().cpu().numpy(),
                "fast_x0_correction_normalized": fast_x0[0].detach().cpu().numpy(),
            }
            for name, value in values.items():
                per_issue[name].append(value.astype(np.float32))

            slow = values["slow_x0_correction_normalized"]
            fast = values["fast_x0_correction_normalized"]
            combined_abs = float(np.abs(slow + fast).sum())
            separate_abs = float(np.abs(slow).sum() + np.abs(fast).sum())
            row: dict[str, object] = {
                "issue": issue,
                "selection": "event" if issue in event_issues else "high_gate_false_positive",
                "event_source": catalog_by_issue.get(issue, {}).get("source", "none"),
                "event_direction": catalog_by_issue.get(issue, {}).get("direction", "none"),
                "event_onset": catalog_by_issue.get(issue, {}).get("lead_onset", -1),
                "event_duration_h": catalog_by_issue.get(issue, {}).get("actual_duration_hours", 0),
                "saved_issue_probability": float(issue_probability[issue]),
                "probe_issue_probability": issue_prob_probe,
                "diffusion_timestep": step,
                "slow_fast_retained_fraction": _safe_ratio(combined_abs, separate_abs),
                "slow_fast_cancellation_fraction": 1.0
                - _safe_ratio(combined_abs, separate_abs),
            }
            row.update({f"slow_{k}": v for k, v in _event_localization(values["slow_mask"], support).items()})
            row.update({f"fast_{k}": v for k, v in _event_localization(values["fast_mask"], support).items()})
            row.update({f"slow_{k}": v for k, v in _correction_localization(slow, support).items()})
            row.update({f"fast_{k}": v for k, v in _correction_localization(fast, support).items()})
            rows.append(row)

        for name in arrays:
            arrays[name].append(np.stack(per_issue[name], axis=0))
        middle = len(timesteps) // 2
        scale = batch["residual_scale"][0].detach().cpu().numpy()
        plot_inputs[issue] = {
            "forecast": batch["forecast"][0].detach().cpu().numpy(),
            "actual": batch["actual"][0].detach().cpu().numpy(),
            "slow_mask": per_issue["slow_mask"][middle],
            "fast_mask": per_issue["fast_mask"][middle],
            "slow_delta_mw": per_issue["slow_x0_correction_normalized"][middle]
            * scale
            * capacities[:, None],
            "fast_delta_mw": per_issue["fast_x0_correction_normalized"][middle]
            * scale
            * capacities[:, None],
            "event_time": event_time,
        }

    if not rows:
        raise RuntimeError("no audit issues were selected")
    issue_array = np.asarray(selected, dtype=np.int64)
    np.save(output / "selected_issue_indices.npy", issue_array)
    np.save(output / "diffusion_timesteps.npy", np.asarray(timesteps, dtype=np.int64))
    for name, values in arrays.items():
        np.save(output / f"{name}.npy", np.stack(values, axis=0))

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "jstd_mechanism_by_issue_timestep.csv", index=False)
    event_rows = frame[frame["selection"] == "event"]
    summary = {
        "method": "fixed_noise_forced_route_jstd_mechanism_audit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_state": args.checkpoint_state,
        "checkpoint_state_key": state_key,
        "data_split": "val",
        "future_actual_or_residual_used_as_generation_condition": False,
        "validation_actual_used_for_offline_scoring_only": True,
        "diffusion_timesteps": timesteps,
        "selected_event_issues": event_issues,
        "selected_high_gate_false_positive_issues": false_positive_issues,
        "event_count": len(event_issues),
        "gate_auc": _binary_auc(targets.event_active, issue_probability),
        "gate_average_precision": _average_precision(targets.event_active, issue_probability),
        "gate_brier": float(np.mean((issue_probability - targets.event_active) ** 2)),
        "event_slow_mask_inside_outside_ratio_mean": float(
            event_rows["slow_mask_inside_outside_ratio"].replace([np.inf, -np.inf], np.nan).mean()
        ),
        "event_fast_mask_inside_outside_ratio_mean": float(
            event_rows["fast_mask_inside_outside_ratio"].replace([np.inf, -np.inf], np.nan).mean()
        ),
        "event_slow_correction_outside_fraction_mean": float(
            event_rows["slow_correction_outside_event_fraction"].mean()
        ),
        "event_fast_correction_outside_fraction_mean": float(
            event_rows["fast_correction_outside_event_fraction"].mean()
        ),
        "event_slow_fast_cancellation_fraction_mean": float(
            event_rows["slow_fast_cancellation_fraction"].mean()
        ),
        "graph_manifest": graph_manifest,
    }
    (output / "jstd_mechanism_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    for issue in selected:
        item = catalog_by_issue.get(issue)
        label = (
            f"{item['source']} {item['direction']}, onset={item['lead_onset']}, "
            f"duration={item['actual_duration_hours']}h"
            if item is not None
            else f"high-gate non-event, p={issue_probability[issue]:.3f}"
        )
        _plot_issue(
            output,
            issue,
            label,
            **plot_inputs[issue],
            wind_weights_mw=wind_weights_mw,
            solar_weights_mw=solar_weights_mw,
        )

    report_lines = [
        "# JSTD-Tail V1 mechanism audit",
        "",
        "This is an offline diagnostic. Validation actual/residual values were used only "
        "to score learned outputs and were not provided as generation conditions.",
        "",
        "## Issue gate",
        "",
        f"- AUROC: {summary['gate_auc']:.4f}",
        f"- Average precision: {summary['gate_average_precision']:.4f}",
        f"- Brier score: {summary['gate_brier']:.4f}",
        "",
        "## Forced-route internal localization",
        "",
        f"- Mean slow-mask inside/outside ratio: {summary['event_slow_mask_inside_outside_ratio_mean']:.4f}",
        f"- Mean fast-mask inside/outside ratio: {summary['event_fast_mask_inside_outside_ratio_mean']:.4f}",
        f"- Mean slow correction outside-event fraction: {summary['event_slow_correction_outside_fraction_mean']:.4f}",
        f"- Mean fast correction outside-event fraction: {summary['event_fast_correction_outside_fraction_mean']:.4f}",
        f"- Mean slow/fast cancellation fraction: {summary['event_slow_fast_cancellation_fraction_mean']:.4f}",
        "",
        "See `jstd_mechanism_by_issue_timestep.csv` and the per-issue PNG files for details.",
    ]
    (output / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"JSTD_MECHANISM_AUDIT_COMPLETE output={output}")


if __name__ == "__main__":
    main()
