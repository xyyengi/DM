#!/usr/bin/env python3
"""Summarize one frozen generation seed across CRN and three S5 permutations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.run_station24_s5_espc import weighted_source


ARMS = ("crn_identity", "permutation_1", "permutation_2", "permutation_3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--generation-seed", required=True, type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval_score(path: Path, capacities: np.ndarray) -> float:
    samples = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
    truth = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
    ensemble = np.asarray(samples, np.float32) @ capacities
    target = np.asarray(truth, np.float32) @ capacities
    lower, upper = np.quantile(ensemble, (0.05, 0.95), axis=1)
    score = (
        upper - lower
        + 20 * (lower - target) * (target < lower)
        + 20 * (target - upper) * (target > upper)
    )
    return float(np.mean(score))


def ordinary_row(name: str, path: Path, capacities: np.ndarray) -> dict:
    metrics = load_json(path / "metrics.json")
    joint = metrics["joint"]
    renewable = metrics["aggregate_mw"]["renewable"]
    return {
        "arm": name,
        "wind_crps": metrics["station_average"]["wind"]["crps"],
        "solar_daylight_crps": metrics["station_average"]["solar_daylight"]["crps"],
        "renewable_crps_mw": renewable["crps"],
        "energy_score_pu": joint["energy_score_pu"],
        "coverage_90": renewable["coverage_90"],
        "width_90_mw": renewable["width_90"],
        "interval_score_90_mw": interval_score(path, capacities),
        "variogram_score": joint["adjacency_variogram_score"],
        "spatial_corr_rmse": joint["spatial_corr_rmse_all_pairs"],
        "wind_solar_corr_rmse": joint["spatial_corr_rmse_wind_solar"],
    }


def pair_delta_row(name: str, root: Path, stations: pd.DataFrame) -> dict:
    raw = np.load(root / "raw_n500" / "actual_scenarios_normalized.npy", mmap_mode="r")
    tail = np.asarray(np.load(root / "tail_n100" / "actual_scenarios_normalized.npy", mmap_mode="r"), np.float32)
    variant = root / "couplings" / name
    indices = np.load(variant / "paired_raw_background_index.npy")
    background = np.stack([np.asarray(raw[i, indices[i]], np.float32) for i in range(raw.shape[0])])
    delta = tail - background
    wind = weighted_source(delta, stations, "wind")
    solar = weighted_source(delta, stations, "solar")
    return {
        "arm": name,
        "delta_variance_all": float(np.var(delta)),
        "delta_rms_all": float(np.sqrt(np.mean(delta.square()))),
        "joint_capacity_energy_mean": float(np.mean(np.abs(wind) + np.abs(solar))),
        "joint_capacity_energy_q95": float(np.quantile(np.abs(wind) + np.abs(solar), 0.95)),
    }


def main() -> None:
    args = parse_args()
    output = args.root / "summary"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    stations = pd.read_csv(args.data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    capacities = stations.capacity_mw.to_numpy(np.float32)
    paths = {"raw_n500": args.root / "raw_n500", "full_v2_mixture": args.root / "couplings" / "full_v2_mixture"}
    paths.update({arm: args.root / "couplings" / arm for arm in ARMS})
    ordinary = pd.DataFrame([ordinary_row(name, path, capacities) for name, path in paths.items()])
    deltas = pd.DataFrame([pair_delta_row(arm, args.root, stations) for arm in ARMS])
    retention_rows, event_rows = [], []
    for arm in ARMS:
        audit = args.root / f"attribution_{arm}"
        summary = load_json(audit / "s5_member_attribution_summary.json")
        retention_rows.append({"arm": arm, **summary})
        event = pd.read_csv(audit / "s5_event_member_attribution.csv")
        event.insert(0, "arm", arm)
        event_rows.append(event)
    retention = pd.DataFrame(retention_rows)
    events = pd.concat(event_rows, ignore_index=True)
    ordinary.to_csv(output / "probability_metrics.csv", index=False)
    deltas.to_csv(output / "pair_delta_metrics.csv", index=False)
    retention.to_csv(output / "event_retention_summary.csv", index=False)
    events.to_csv(output / "per_event_member_attribution.csv", index=False)

    permutations = deltas[deltas.arm.str.startswith("permutation_")]
    crn_variance = float(deltas.loc[deltas.arm.eq("crn_identity"), "delta_variance_all"].iloc[0])
    permutation_range = [float(permutations.delta_variance_all.min()), float(permutations.delta_variance_all.max())]
    crn_retention = float(retention.loc[retention.arm.eq("crn_identity"), "mean_event_retention_rate"].iloc[0])
    permutation_retention = retention[retention.arm.str.startswith("permutation_")].mean_event_retention_rate
    decision = {
        "generation_seed": args.generation_seed,
        "independent_physical_event_count": int(events.event_id.nunique()),
        "crn_delta_variance": crn_variance,
        "permutation_delta_variance_range": permutation_range,
        "crn_lower_delta_variance_than_all_permutations": bool(crn_variance < permutation_range[0]),
        "crn_event_retention": crn_retention,
        "permutation_event_retention_range": [float(permutation_retention.min()), float(permutation_retention.max())],
        "single_generation_seed_only": True,
        "generalization_claim_permitted": False,
        "pairing_major_bottleneck_proven": False,
    }
    (output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(deltas.arm, deltas.delta_variance_all)
    axes[0].set_title("Raw–Tail pair-delta variance")
    axes[1].bar(retention.arm, retention.mean_event_retention_rate)
    axes[1].set_title("Full-Tail hit retention (4 events)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "CRN_SINGLE_SEED_OVERVIEW.png", dpi=180)
    plt.close(fig)

    report = [
        "# Station-24 Original S5 单seed CRN配对归因报告", "",
        f"Generation seed: `{args.generation_seed}`。本实验无训练、未访问Test。", "",
        "所有S5配对臂复用完全相同的Raw500与Tail100边缘成员；仅Raw背景与Tail的配对关系不同。", "",
        "## 三类结论必须分开", "",
        f"1. CRN差值方差：`{crn_variance:.8g}`；三次固定置换范围：`{permutation_range}`。",
        f"2. CRN事件命中保留率：`{crn_retention:.6g}`；置换范围：`{decision['permutation_event_retention_range']}`。",
        "3. 最终概率质量见 `probability_metrics.csv`；降低差值方差不自动等于事件保留或概率质量改善。", "",
        "## 证据边界", "",
        "这里只有一个generation seed和少量独立物理事件。本报告只能做成员配对机制归因，不能证明泛化优势，也不会自动触发新S5开发。", "",
        "逐事件结果见 `per_event_member_attribution.csv`，CRPS/Energy/coverage/width/Interval Score/Variogram/相关结构见 `probability_metrics.csv`。", "",
    ]
    (output / "CRN_SINGLE_SEED_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"S5_CRN_SINGLE_SEED_SUMMARY_COMPLETE output={output}")


if __name__ == "__main__":
    main()
