#!/usr/bin/env python3
"""Summarize the zero-training Station-24 S5-ESPC experiment.

This script only reads frozen validation member pools and already-produced
evaluation artifacts. It never trains, samples, or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--data-path", type=Path, default=Path("diffusion_input_station"))
    return p.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(label: str, path: Path) -> dict[str, float | str]:
    m = load_json(path / "metrics.json")
    return {
        "variant": label,
        "wind_crps": m["station_average"]["wind"]["crps"],
        "solar_daylight_crps": m["station_average"]["solar_daylight"]["crps"],
        "renewable_crps_mw": m["aggregate_mw"]["renewable"]["crps"],
        "energy_score": m["joint"]["energy_score_pu"],
        "coverage_90": m["aggregate_mw"]["renewable"]["coverage_90"],
        "interval_width_90_mw": m["aggregate_mw"]["renewable"]["width_90"],
        "spatial_corr_rmse": m["joint"]["spatial_corr_rmse_all_pairs"],
        "wind_solar_corr_rmse": m["joint"]["spatial_corr_rmse_wind_solar"],
    }


def weighted_renewable(path: Path, capacities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
    truth = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
    return np.asarray(samples, dtype=np.float32) @ capacities, np.asarray(truth, dtype=np.float32) @ capacities


def interval_score(samples: np.ndarray, truth: np.ndarray) -> float:
    lo, hi = np.quantile(samples, [0.05, 0.95], axis=1)
    return float(np.mean((hi - lo) + 20 * (lo - truth) * (truth < lo) + 20 * (truth - hi) * (truth > hi)))


def event_root(path: Path) -> Path:
    direct = path / "event_evaluation"
    if direct.exists():
        return direct
    # The historical Full V2 result stored evaluation beside, rather than below,
    # the mixture directory. Keep that historical layout read-only.
    for candidate in sorted(path.parent.glob("postprocess*/continuous_event_evaluation")):
        if (candidate / "continuous_event_three_standard_summary.csv").exists():
            return candidate
    raise FileNotFoundError(f"no continuous-event evaluation found for {path}")


def event_summary(label: str, path: Path) -> list[dict]:
    df = pd.read_csv(event_root(path) / "continuous_event_three_standard_summary.csv")
    rows: list[dict] = []
    for standard in ("loose", "primary", "strict"):
        candidate = ~df["variant"].str.lower().str.startswith("raw")
        q = df[candidate & (df.scope == "independent_physical") &
               (df.member_group == "tail") & (df.standard == standard)]
        if q.empty:
            continue
        r = q.iloc[0]
        rows.append({
            "variant": label, "standard": standard,
            "event_count": int(r.event_count), "events_with_any_hit": int(r.events_with_any_hit),
            "mean_member_hit_rate": float(r.mean_member_hit_rate),
            "median_onset_error_h": float(r.median_onset_error_h),
            "median_duration_error_h": float(r.median_duration_error_h),
            "median_depth_ratio": float(r.median_depth_ratio),
        })
    return rows


def per_event_primary(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(event_root(path) / "continuous_event_per_event.csv")
    candidate = ~df["variant"].str.lower().str.startswith("raw")
    q = df[candidate & (df.scope == "independent_physical") &
           (df.member_group == "tail") & (df.standard == "primary")].copy()
    q.insert(0, "method", label)
    return q[["method", "event_id", "hit_count", "member_hit_rate", "any_hit",
              "median_onset_error_h", "median_duration_error_h", "median_depth_ratio"]]


def ramp_rows(label: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(event_root(path) / "fast_ramp_1_3_6h_summary.csv")
    q = df[~df.variant.str.lower().str.startswith("raw")].copy()
    q.insert(0, "method", label)
    q["std_truth_ratio"] = q.generated_ramp_std / q.actual_ramp_std
    return q[["method", "source", "lag_h", "median_ramp_mae", "ramp_90_coverage",
              "generated_ramp_std", "actual_ramp_std", "std_truth_ratio"]]


def mask_metrics(label: str, path: Path, raw: Path, capacities: np.ndarray) -> dict:
    masks = np.load(path / "localized_tail_mask.npy", mmap_mode="r")
    local = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")[:, 400:500]
    bg = np.load(raw / "actual_scenarios_normalized.npy", mmap_mode="r")[:, 400:500]
    outside = np.broadcast_to(np.asarray(masks)[:, :, :, None] == 0, local.shape)
    max_outside = float(np.max(np.abs(np.asarray(local)[outside] - np.asarray(bg)[outside])))
    starts = np.load(path / "localized_tail_start.npy", mmap_mode="r")
    stops = np.load(path / "localized_tail_stop.npy", mmap_mode="r")
    correction = (np.asarray(local, dtype=np.float32) - np.asarray(bg, dtype=np.float32)) @ capacities
    induced_boundary_steps = []
    for issue in range(starts.shape[0]):
        for member in range(starts.shape[1]):
            for time in (int(starts[issue, member]), int(stops[issue, member])):
                if 0 < time < correction.shape[-1]:
                    induced_boundary_steps.append(abs(float(correction[issue, member, time] - correction[issue, member, time - 1])))
    return {
        "variant": label,
        "localized_time_fraction": float(np.mean(np.asarray(masks) > 0)),
        "strict_raw_outside_max_abs_error": max_outside,
        "mean_boundary_induced_step_mw": float(np.mean(induced_boundary_steps)),
        "max_boundary_induced_step_mw": float(np.max(induced_boundary_steps)),
    }


def plot_event_envelopes(output: Path, issue: int, source: str, title: str,
                         paths: dict[str, Path], capacities: np.ndarray) -> None:
    idx = np.arange(12) if source == "wind" else np.arange(12, 24)
    w = capacities[idx]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    truth = np.load(next(iter(paths.values())) / "actual_data_normalized.npy", mmap_mode="r")
    truth_agg = np.asarray(truth[issue])[:, idx] @ w
    axes[0].plot(truth_agg, color="black", lw=2.3, label="Truth (evaluation only)")
    colors = {"Original S5": "#3366cc", "Event+adaptive": "#109618", "S5-ESPC": "#dc3912"}
    for label, path in paths.items():
        samples = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
        agg = np.asarray(samples[issue, 400:500])[:, :, idx] @ w
        q05, med, q95 = np.quantile(agg, [0.05, 0.5, 0.95], axis=0)
        axes[0].plot(med, lw=1.5, color=colors[label], label=label)
        axes[0].fill_between(np.arange(168), q05, q95, color=colors[label], alpha=0.10)
        mask = np.load(path / "localized_tail_mask.npy", mmap_mode="r")
        axes[1].plot(np.mean(np.asarray(mask[issue]) > 0, axis=0), color=colors[label], lw=1.5, label=label)
    axes[0].set_ylabel(f"Capacity-weighted {source} (MW)")
    axes[0].set_title(title)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set_ylabel("Members localized")
    axes[1].set_xlabel("Hour in 168 h window")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    a = parse_args()
    root = a.root
    stations = pd.read_csv(a.data_path / "station_metadata.csv").sort_values("channel_index")
    capacities = stations.capacity_mw.to_numpy(np.float32)
    paths = {
        "Raw 500": a.raw,
        "Full V2 mixture": a.full,
        "Original S5": a.original,
        "S5 + event window": root / "s5_event_fixed",
        "S5 + event + adaptive": root / "s5_event_adaptive",
        "S5-ESPC": root / "s5_espc",
    }

    ordinary = pd.DataFrame([metric_row(k, v) for k, v in paths.items()])
    for i, row in ordinary.iterrows():
        samples, truth = weighted_renewable(paths[row.variant], capacities)
        ordinary.loc[i, "interval_score_90_mw"] = interval_score(samples, truth)
    ordinary.to_csv(root / "S5_ESPC_ABLATION.csv", index=False)

    event_paths = {k: v for k, v in paths.items() if k != "Raw 500"}
    events = pd.DataFrame(sum((event_summary(k, v) for k, v in event_paths.items()), []))
    events.to_csv(root / "S5_ESPC_EVENT_SUMMARY.csv", index=False)
    per_event = pd.concat([per_event_primary(k, v) for k, v in event_paths.items()], ignore_index=True)
    per_event.to_csv(root / "S5_ESPC_PER_EVENT_PRIMARY.csv", index=False)
    ramps = pd.concat([ramp_rows(k, v) for k, v in event_paths.items()], ignore_index=True)
    ramps.to_csv(root / "S5_ESPC_RAMP_SUMMARY.csv", index=False)

    mask_paths = {"Original S5": a.original, "S5 + event window": root / "s5_event_fixed",
                  "S5 + event + adaptive": root / "s5_event_adaptive", "S5-ESPC": root / "s5_espc"}
    masks = pd.DataFrame([mask_metrics(k, v, a.raw, capacities) for k, v in mask_paths.items()])
    records = pd.read_csv(root / "window_selection_records.csv")
    reference = records[records.variant == "s5_event_adaptive"].sort_values(["issue", "member"])
    original_start = np.load(a.original / "localized_tail_start.npy").reshape(-1)
    original_stop = np.load(a.original / "localized_tail_stop.npy").reshape(-1)
    ref_start = reference.structural_reference_start.to_numpy(int)
    ref_stop = reference.structural_reference_stop.to_numpy(int)
    overlap = np.maximum(0, np.minimum(original_stop, ref_stop) - np.maximum(original_start, ref_start))
    original_overlap = overlap / np.maximum(ref_stop - ref_start, 1)
    for i, row in masks.iterrows():
        key = {"S5 + event window": "s5_event_fixed", "S5 + event + adaptive": "s5_event_adaptive",
               "S5-ESPC": "s5_espc"}.get(row.variant)
        if key:
            q = records[records.variant == key]
            masks.loc[i, "mean_duration_h"] = q.duration_h.mean()
            masks.loc[i, "boundary_feasible_rate"] = q.boundary_feasible.mean()
            masks.loc[i, "fallback_rate"] = q.fallback.mean()
            masks.loc[i, "mean_structural_reference_overlap"] = q.structural_reference_overlap_fraction.mean()
        else:
            masks.loc[i, "mean_duration_h"] = np.mean(np.load(a.original / "localized_tail_stop.npy") - np.load(a.original / "localized_tail_start.npy"))
            masks.loc[i, "fallback_rate"] = 0.0
            masks.loc[i, "mean_structural_reference_overlap"] = float(np.mean(original_overlap))
            masks.loc[i, "zero_structural_reference_overlap_rate"] = float(np.mean(original_overlap == 0))
            masks.loc[i, "full_structural_reference_overlap_rate"] = float(np.mean(original_overlap >= 1))
    masks.to_csv(root / "S5_ESPC_SCOPE_AND_BOUNDARY.csv", index=False)

    pairing = pd.read_csv(root / "pairing_sensitivity.csv")
    pairing_events = []
    for seed in (1729, 271828, 314159):
        p = root / "pairing_sensitivity" / f"random_pairing_{seed}"
        x = event_summary(f"random_pairing_{seed}", p)
        pairing_events.extend([r for r in x if r["standard"] == "primary"])
    pd.DataFrame(pairing_events).to_csv(root / "S5_PAIRING_EVENT_SENSITIVITY.csv", index=False)

    # Locate representative issues from the frozen event catalog; truth is used only in these post-hoc plots.
    match = pd.read_csv(event_root(root / "s5_event_adaptive") / "continuous_event_member_matches.csv")
    success_issue = int(match[match.event_id == "wind_negative_000"].issue.iloc[0])
    failure_issue = int(match[match.event_id == "wind_positive_002"].issue.iloc[0])
    plot_paths = {"Original S5": a.original, "Event+adaptive": root / "s5_event_adaptive", "S5-ESPC": root / "s5_espc"}
    plot_event_envelopes(root / "FIG_SUCCESS_WIND_NEGATIVE.png", success_issue, "wind",
                         "Representative success: negative-wind event (post-hoc evaluation)", plot_paths, capacities)
    plot_event_envelopes(root / "FIG_FAILURE_WIND_POSITIVE.png", failure_issue, "wind",
                         "Representative failure: positive-wind event (post-hoc evaluation)", plot_paths, capacities)

    # Compact comparison figure.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    event_primary = events[events.standard == "primary"]
    ax[0].barh(event_primary.variant, event_primary.mean_member_hit_rate, color="#4472c4")
    ax[0].set_title("Primary event member-hit rate")
    ax[0].set_xlabel("Rate (4 events only)")
    base = float(ordinary[ordinary.variant == "Original S5"].renewable_crps_mw.iloc[0])
    ax[1].barh(ordinary.variant, ordinary.renewable_crps_mw - base, color="#70ad47")
    ax[1].axvline(0, color="black", lw=.8)
    ax[1].set_title("Renewable CRPS difference vs Original S5")
    ax[1].set_xlabel("MW; lower is better")
    fig.tight_layout()
    fig.savefig(root / "FIG_ABLATION_OVERVIEW.png", dpi=180)
    plt.close(fig)

    manifest = load_json(root / "EXPERIMENT_MANIFEST.json")
    summary = {
        "status": "PASS",
        "training": "NOT RUN",
        "generation": "NOT RUN (existing member pools reused)",
        "test_split_accessed": False,
        "s5_reproduction": load_json(root / "S5_REPRODUCTION.json"),
        "fallback_rate_espc": float(records[records.variant == "s5_espc"].fallback.mean()),
        "elapsed_core_seconds": manifest["elapsed_seconds"],
        "recommendation": "Do not replace Original S5 with full S5-ESPC; event-aware adaptive length is diagnostic-only pending independent evidence.",
    }
    (root / "FINAL_DECISION.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"S5_ESPC_SUMMARY_COMPLETE output={root}")


if __name__ == "__main__":
    main()
