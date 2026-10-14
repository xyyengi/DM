#!/usr/bin/env python3
"""Summarize the frozen three-seed Original-S5 full-path CRN experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.run_station24_s5_espc import (
    candidate_rows_for_member,
    fit_train_statistics,
    weighted_source,
)


SEEDS = (271828, 314159, 161803)
ARMS = ("crn_identity", "independent_permutation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("diffusion_input_station"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    prefix = np.pad(np.cumsum(values, axis=-2), ((0, 0), (0, 0), (1, 0), (0, 0)))
    return (prefix[:, :, width:] - prefix[:, :, :-width]) / width


def interval_score(path: Path, capacities: np.ndarray) -> float:
    samples = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
    truth = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
    ensemble = np.asarray(samples, np.float32) @ capacities
    target = np.asarray(truth, np.float32) @ capacities
    lower, upper = np.quantile(ensemble, (0.05, 0.95), axis=1)
    score = ((upper - lower) + 20 * (lower - target) * (target < lower)
             + 20 * (target - upper) * (target > upper))
    return float(np.mean(score))


def ordinary_row(seed: int, arm: str, path: Path, capacities: np.ndarray) -> dict:
    metrics = load_json(path / "metrics.json")
    joint = metrics["joint"]
    renewable = metrics["aggregate_mw"]["renewable"]
    return {
        "seed": seed, "arm": arm,
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
        "wind_acf_error_lag1": joint["wind_acf_abs_error_lag1"],
        "wind_acf_error_lag6": joint["wind_acf_abs_error_lag6"],
        "wind_acf_error_lag24": joint["wind_acf_abs_error_lag24"],
        "wind_acf_error_lag48": joint["wind_acf_abs_error_lag48"],
        "solar_acf_error_lag1": joint["solar_acf_abs_error_lag1"],
        "solar_acf_error_lag6": joint["solar_acf_abs_error_lag6"],
        "solar_acf_error_lag24": joint["solar_acf_abs_error_lag24"],
        "solar_acf_error_lag48": joint["solar_acf_abs_error_lag48"],
    }


def paired_delta_row(seed: int, arm: str, seed_root: Path, stations: pd.DataFrame) -> dict:
    raw_path = seed_root / "raw_n500"
    tail_path = seed_root / "tail_n100"
    variant = seed_root / "couplings" / arm
    raw = np.load(raw_path / "actual_scenarios_normalized.npy", mmap_mode="r")
    tail = np.asarray(np.load(tail_path / "actual_scenarios_normalized.npy", mmap_mode="r")[:, :100], np.float32)
    indices = np.load(variant / "paired_raw_background_index.npy")
    background = np.stack([np.asarray(raw[i, indices[i]], np.float32) for i in range(raw.shape[0])])
    delta = tail - background
    wind = weighted_source(delta, stations, "wind")
    solar = weighted_source(delta, stations, "solar")
    energy = np.abs(wind) + np.abs(solar)
    result = {
        "seed": seed, "arm": arm,
        "delta_variance_all": float(np.var(delta)),
        "delta_rms_all": float(np.sqrt(np.mean(delta.square()))),
        "joint_capacity_energy_mean": float(np.mean(energy)),
        "joint_capacity_energy_q95": float(np.quantile(energy, 0.95)),
    }
    for width in (12, 24):
        low = moving_average(delta, width)
        result[f"low_frequency_{width}h_rms"] = float(np.sqrt(np.mean(low.square())))
    return result


def structural_overlap(seed_root: Path, arm: str, data_path: Path,
                       stations: pd.DataFrame, stats: dict) -> dict:
    tail = np.load(seed_root / "tail_n100" / "actual_scenarios_normalized.npy", mmap_mode="r")
    raw = np.load(seed_root / "raw_n500" / "actual_scenarios_normalized.npy", mmap_mode="r")
    daylight = np.load(seed_root / "raw_n500" / "station_daylight_mask.npy", mmap_mode="r")
    variant = seed_root / "couplings" / arm
    indices = np.load(variant / "paired_raw_background_index.npy")
    starts = np.load(variant / "localized_tail_start.npy")
    stops = np.load(variant / "localized_tail_stop.npy")
    energy = np.load(variant / "pair_correction_energy.npy", mmap_mode="r")
    lengths = list(stats["duration_candidates_h"])
    overlaps, ious, exact_zero = [], [], []
    records = []
    for issue in range(tail.shape[0]):
        for member in range(100):
            rows = candidate_rows_for_member(
                energy=np.asarray(energy[issue, member]),
                background=np.asarray(raw[issue, indices[issue, member]]),
                tail=np.asarray(tail[issue, member]),
                daylight=np.asarray(daylight[issue]),
                stations=stations,
                lengths=lengths,
                taper=3,
                stats=stats,
            )
            reference = max(rows, key=lambda row: row["structure"])
            left, right = int(starts[issue, member]), int(stops[issue, member])
            intersection = max(0, min(right, reference["stop"]) - max(left, reference["start"]))
            reference_fraction = intersection / max(reference["stop"] - reference["start"], 1)
            union = max(right, reference["stop"]) - min(left, reference["start"])
            iou = intersection / max(union, 1)
            overlaps.append(reference_fraction); ious.append(iou); exact_zero.append(intersection == 0)
            records.append({
                "seed": int(seed_root.name.split("_")[-1]), "arm": arm,
                "issue": issue, "tail_member": member,
                "selected_start": left, "selected_stop": right,
                "structure_start": int(reference["start"]),
                "structure_stop": int(reference["stop"]),
                "structure_reference_overlap": reference_fraction,
                "window_iou": iou,
            })
    return {
        "summary": {
            "mean_structure_reference_overlap": float(np.mean(overlaps)),
            "mean_window_iou": float(np.mean(ious)),
            "zero_overlap_rate": float(np.mean(exact_zero)),
        },
        "records": records,
    }


def cross_seed_summary(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    numeric = [column for column in frame.columns if column not in ("seed", group)]
    rows = []
    for key, subset in frame.groupby(group, sort=False):
        for metric in numeric:
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            rows.append({
                group: key, "metric": metric, "mean": values.mean(),
                "std": values.std(ddof=1), "min": values.min(), "max": values.max(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output = args.root / "cross_seed_summary"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    stations = pd.read_csv(args.data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    capacities = stations.capacity_mw.to_numpy(np.float32)
    stats = fit_train_statistics(args.data_path, stations)
    ordinary, deltas, overlaps, overlap_records, events = [], [], [], [], []
    attribution_summaries = []
    for seed in SEEDS:
        seed_root = args.root / f"seed_{seed}"
        for arm in ARMS:
            variant = seed_root / "couplings" / arm
            ordinary.append(ordinary_row(seed, arm, variant, capacities))
            deltas.append(paired_delta_row(seed, arm, seed_root, stations))
            structure = structural_overlap(seed_root, arm, args.data_path, stations, stats)
            overlaps.append({"seed": seed, "arm": arm, **structure["summary"]})
            overlap_records.extend(structure["records"])
            audit = seed_root / f"attribution_{arm}"
            event = pd.read_csv(audit / "s5_event_member_attribution.csv")
            event.insert(0, "arm", arm); event.insert(0, "seed", seed)
            events.append(event)
            attribution_summaries.append({
                "seed": seed, "arm": arm,
                **load_json(audit / "s5_member_attribution_summary.json"),
            })

    ordinary_df = pd.DataFrame(ordinary); delta_df = pd.DataFrame(deltas)
    overlap_df = pd.DataFrame(overlaps); events_df = pd.concat(events, ignore_index=True)
    attribution_df = pd.DataFrame(attribution_summaries)
    ordinary_df.to_csv(output / "per_seed_probability_metrics.csv", index=False)
    delta_df.to_csv(output / "per_seed_pair_delta_metrics.csv", index=False)
    overlap_df.to_csv(output / "per_seed_window_structure_overlap.csv", index=False)
    pd.DataFrame(overlap_records).to_csv(output / "per_member_window_structure_overlap.csv", index=False)
    events_df.to_csv(output / "per_event_seed_attribution.csv", index=False)
    attribution_df.to_csv(output / "per_seed_event_retention.csv", index=False)
    cross_seed_summary(ordinary_df, "arm").to_csv(output / "probability_cross_seed_summary.csv", index=False)
    cross_seed_summary(delta_df, "arm").to_csv(output / "pair_delta_cross_seed_summary.csv", index=False)
    cross_seed_summary(overlap_df, "arm").to_csv(output / "overlap_cross_seed_summary.csv", index=False)

    paired_delta = delta_df.pivot(index="seed", columns="arm", values="delta_variance_all")
    delta_lower_all = bool((paired_delta.crn_identity < paired_delta.independent_permutation).all())
    retention = attribution_df.pivot(index="seed", columns="arm", values="mean_event_retention_rate")
    retention_better_seeds = int((retention.crn_identity > retention.independent_permutation).sum())
    event_mean = events_df.groupby(["arm", "event_id"], as_index=False).retention_rate_of_full_hits.mean()
    event_pivot = event_mean.pivot(index="event_id", columns="arm", values="retention_rate_of_full_hits")
    better_events = int((event_pivot.crn_identity > event_pivot.independent_permutation).sum())
    equal_events = int((event_pivot.crn_identity == event_pivot.independent_permutation).sum())

    lower_better = ["renewable_crps_mw", "energy_score_pu", "interval_score_90_mw",
                    "variogram_score", "spatial_corr_rmse", "wind_solar_corr_rmse"]
    ordinary_pivot = ordinary_df.pivot(index="seed", columns="arm", values=lower_better)
    body_directions = {}
    for metric in lower_better:
        comparison = ordinary_pivot[metric]
        body_directions[metric] = int(
            (comparison.crn_identity <= comparison.independent_permutation).sum()
        )
    body_not_systematically_worse = all(value >= 2 for value in body_directions.values())

    evidence_pairing_bottleneck = bool(
        delta_lower_all and retention_better_seeds >= 2 and better_events >= 3
        and body_not_systematically_worse
    )
    decision = {
        "delta_randomness_reduced_all_three_seeds": delta_lower_all,
        "event_retention_better_seed_count": retention_better_seeds,
        "event_retention_better_event_count": better_events,
        "event_retention_equal_event_count": equal_events,
        "ordinary_lower_or_equal_seed_count_by_metric": body_directions,
        "body_quality_not_systematically_worse": body_not_systematically_worse,
        "evidence_pairing_is_major_s5_bottleneck": evidence_pairing_bottleneck,
        "independent_physical_event_count": 4,
        "generalization_claim_permitted": False,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for arm, color in zip(ARMS, ("#3366cc", "#dc3912")):
        q = delta_df[delta_df.arm == arm]
        axes[0].plot(q.seed.astype(str), q.delta_variance_all, "o-", label=arm, color=color)
        q = overlap_df[overlap_df.arm == arm]
        axes[1].plot(q.seed.astype(str), q.mean_structure_reference_overlap, "o-", label=arm, color=color)
        q = attribution_df[attribution_df.arm == arm]
        axes[2].plot(q.seed.astype(str), q.mean_event_retention_rate, "o-", label=arm, color=color)
    axes[0].set_title("Pair-delta variance"); axes[1].set_title("Window/structure overlap")
    axes[2].set_title("Full-Tail hit retention (4 events)")
    for axis in axes:
        axis.set_xlabel("Generation seed"); axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "CRN_COUPLING_OVERVIEW.png", dpi=180); plt.close(fig)

    conclusion = (
        "现有三seed描述性证据支持成员coupling是Original S5的主要瓶颈。"
        if evidence_pairing_bottleneck else
        "现有三seed证据不足以把成员coupling认定为Original S5的主要瓶颈。"
    )
    next_step = (
        "唯一下一步：在更大的独立物理事件集上冻结CRN pairing并前瞻复核；不要继续调整S5窗口。"
        if evidence_pairing_bottleneck else
        "唯一下一步：停止S5窗口扩展，转向Lightweight Tail的drop–recovery局部轨迹监督。"
    )
    report = [
        "# Station-24 Original S5 全路径 CRN 配对归因报告", "",
        "本实验无训练、未访问Test，Raw/Tail边缘成员在两coupling臂间完全相同。", "",
        "## 三个相互独立的结论", "",
        f"1. **差值随机性**：三seed全部下降 = `{delta_lower_all}`。",
        f"2. **事件保留**：CRN更好的seed数 = `{retention_better_seeds}/3`；"
        f"更好的独立事件数 = `{better_events}/4`（相同 `{equal_events}/4`）。",
        f"3. **最终概率质量**：各指标CRN不差于置换臂的seed数 = `{body_directions}`。", "",
        "## 方法论结论", "", conclusion, "",
        "由于只有4个独立物理事件，本报告不宣称泛化优势或统计显著性。成员数和重叠issue不作为独立事件样本。", "",
        "## 唯一下一步建议", "", next_step, "",
        "详细逐seed、逐事件及跨seed均值/标准差/范围见本目录CSV与 `decision.json`。", "",
    ]
    (output / "CRN_ATTRIBUTION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"S5_CRN_SUMMARY_COMPLETE output={output}")


if __name__ == "__main__":
    main()
