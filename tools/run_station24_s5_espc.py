#!/usr/bin/env python3
"""Zero-training S5-ESPC ablation on frozen Station-24 Raw/Full-V2 members.

Window selection is causal with respect to validation targets: it uses only
generated Raw/Tail trajectories, astronomical daylight, and train-fitted
duration/level/ramp statistics. Validation actuals are loaded only after all
localized members and selection records have been finalized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from station_dataset import build_station_daylight_mask
from station_evaluation import evaluate_station_scenarios, save_evaluation
from station_jstd_targets import build_station_jstd_target_arrays, fit_station_jstd_event_thresholds
from tools.localize_station24_independent_joint_tail import (
    MEMBER_ARRAYS,
    STATIC_ARRAYS,
    _assert_same,
    _load,
    build_tapered_masks,
    joint_correction_energy,
    localize_members,
    stratified_train_durations,
    strongest_windows,
)


VARIANTS = ("s5_event_fixed", "s5_event_adaptive", "s5_espc")
PAIRING_SEEDS = (1729, 271828, 314159)
LAGS = (1, 3, 6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body-results", required=True)
    p.add_argument("--tail-results", required=True)
    p.add_argument("--original-s5-results", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--data-path", default="diffusion_input_station")
    p.add_argument("--body-members", type=int, default=400)
    p.add_argument("--tail-members", type=int, default=100)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--edge-taper-hours", type=int, default=3)
    p.add_argument("--energy-score-member-limit", type=int, default=80)
    return p.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_source(values: np.ndarray, stations: pd.DataFrame, source: str) -> np.ndarray:
    idx = stations.index[stations.data_type.eq(source)].to_numpy(int)
    capacity = stations.capacity_mw.to_numpy(float)[idx]
    weight = capacity / capacity.sum()
    return np.einsum("...ts,s->...t", values[..., idx], weight)


def daylight_source(
    values: np.ndarray, daylight: np.ndarray, stations: pd.DataFrame, source: str
) -> tuple[np.ndarray, np.ndarray]:
    if source != "solar":
        result = weighted_source(values, stations, source)
        return result, np.ones(result.shape[-1], dtype=bool) if result.ndim == 1 else np.ones(result.shape, dtype=bool)
    idx = stations.index[stations.data_type.eq("solar")].to_numpy(int)
    capacity = stations.capacity_mw.to_numpy(float)[idx]
    weight = capacity / capacity.sum()
    mask = np.asarray(daylight[..., idx], float)
    available = np.einsum("...ts,s->...t", mask, weight)
    numerator = np.einsum("...ts,...ts,s->...t", values[..., idx], mask, weight)
    result = np.divide(numerator, available, out=np.zeros_like(numerator), where=available > 1e-8)
    return result, available > 0.5


def fit_train_statistics(data_path: Path, stations: pd.DataFrame) -> dict[str, object]:
    train = np.asarray(np.load(data_path / "train_actual.npy", mmap_mode="r"), float)
    fill_valid = np.asarray(np.load(data_path / "train_fill_mask.npy", mmap_mode="r")) == 0
    daylight, daylight_audit = build_station_daylight_mask(data_path, "train")
    stats: dict[str, object] = {"fit_split": "train", "daylight": daylight_audit, "sources": {}}
    for source in ("wind", "solar"):
        series, dayvalid = daylight_source(train, daylight, stations, source)
        idx = stations.index[stations.data_type.eq(source)].to_numpy(int)
        valid = fill_valid[..., idx].all(axis=-1)
        if source == "solar":
            valid &= dayvalid
        source_stats = {
            "low_q10": float(np.quantile(series[valid], 0.10)),
            "ramp_abs_q90": {},
            "ramp_abs_q99": {},
        }
        for lag in LAGS:
            delta = series[:, lag:] - series[:, :-lag]
            pair = valid[:, lag:] & valid[:, :-lag]
            values = np.abs(delta[pair])
            source_stats["ramp_abs_q90"][str(lag)] = float(np.quantile(values, 0.90))
            source_stats["ramp_abs_q99"][str(lag)] = float(np.quantile(values, 0.99))
        stats["sources"][source] = source_stats
    thresholds = fit_station_jstd_event_thresholds(data_path)
    targets = build_station_jstd_target_arrays(data_path, "train", thresholds)
    durations = np.asarray([r["actual_duration_hours"] for r in targets.audit["segment_catalog"]], float)
    q = np.quantile(durations, (0.25, 0.50, 0.75), method="nearest")
    stats["duration_candidates_h"] = sorted(set(int(x) for x in np.clip(np.rint(q), 1, 168)))
    stats["duration_catalog_count"] = int(durations.size)
    stats["duration_quantiles"] = {"q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2])}
    return stats


def saturate(value: float) -> float:
    value = max(float(value), 0.0)
    return value / (1.0 + value)


def segment_structure(
    wind: np.ndarray,
    solar: np.ndarray,
    solar_valid: np.ndarray,
    start: int,
    stop: int,
    stats: dict[str, object],
) -> tuple[float, float, float, float]:
    w = wind[start:stop]
    s = solar[start:stop]
    sv = solar_valid[start:stop]
    wind_low = float(np.mean(w <= stats["sources"]["wind"]["low_q10"]))
    solar_low = float(np.mean(s[sv] <= stats["sources"]["solar"]["low_q10"])) if np.any(sv) else 0.0
    low = max(wind_low, solar_low)
    ramp_ratio = 0.0
    for source, values, valid in (("wind", w, np.ones(w.shape, bool)), ("solar", s, sv)):
        for lag in LAGS:
            if values.size <= lag:
                continue
            pair = valid[lag:] & valid[:-lag]
            if np.any(pair):
                scale = max(float(stats["sources"][source]["ramp_abs_q90"][str(lag)]), 1e-12)
                ramp_ratio = max(ramp_ratio, float(np.max(np.abs(values[lag:] - values[:-lag])[pair])) / scale)
    ramp = saturate(ramp_ratio)
    recovery_ratio = 0.0
    for source, values, valid in (("wind", w, np.ones(w.shape, bool)), ("solar", s, sv)):
        if values.size < 3:
            continue
        clean = np.where(valid, values, np.nan)
        for trough in range(1, len(clean) - 1):
            if not np.isfinite(clean[trough]):
                continue
            before = clean[:trough]
            after = clean[trough + 1:]
            if not np.any(np.isfinite(before)) or not np.any(np.isfinite(after)):
                continue
            drop = max(float(np.nanmax(before) - clean[trough]), 0.0)
            rise = max(float(np.nanmax(after) - clean[trough]), 0.0)
            scale = max(float(stats["sources"][source]["ramp_abs_q90"]["3"]), 1e-12)
            recovery_ratio = max(recovery_ratio, min(drop / scale, rise / scale))
    recovery = saturate(recovery_ratio)
    return max(low, ramp, recovery), low, ramp, recovery


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = (np.arange(len(values), dtype=float) + 1.0) / len(values)
    return ranks


def boundary_feasible(
    background_wind: np.ndarray,
    background_solar: np.ndarray,
    local_wind: np.ndarray,
    local_solar: np.ndarray,
    start: int,
    stop: int,
    stats: dict[str, object],
) -> tuple[bool, float]:
    worst_excess = 0.0
    for source, raw, local in (
        ("wind", background_wind, local_wind),
        ("solar", background_solar, local_solar),
    ):
        for lag in LAGS:
            limit = float(stats["sources"][source]["ramp_abs_q99"][str(lag)])
            raw_delta = np.abs(raw[lag:] - raw[:-lag])
            local_delta = np.abs(local[lag:] - local[:-lag])
            times = np.arange(lag, len(raw))
            near = ((times >= start - lag) & (times <= start + 3 + lag)) | ((times >= stop - 3 - lag) & (times <= stop + lag))
            if not np.any(near):
                continue
            allowed = np.maximum(limit, raw_delta[near])
            excess = local_delta[near] - allowed
            worst_excess = max(worst_excess, float(np.max(excess, initial=0.0)))
    return worst_excess <= 1e-10, worst_excess


def candidate_rows_for_member(
    energy: np.ndarray,
    background: np.ndarray,
    tail: np.ndarray,
    daylight: np.ndarray,
    stations: pd.DataFrame,
    lengths: list[int],
    taper: int,
    stats: dict[str, object],
) -> list[dict[str, object]]:
    bw = weighted_source(background, stations, "wind")
    tw = weighted_source(tail, stations, "wind")
    bs, solar_valid = daylight_source(background, daylight, stations, "solar")
    ts, _ = daylight_source(tail, daylight, stations, "solar")
    rows = []
    for length in lengths:
        length = min(max(int(length), 1), len(energy))
        for start in range(0, len(energy) - length + 1):
            stop = start + length
            structure, low, ramp, recovery = segment_structure(tw, ts, solar_valid, start, stop, stats)
            mask = build_tapered_masks(np.array([[start]]), np.array([[stop]]), len(energy), taper)[0, 0]
            lw = bw + mask * (tw - bw)
            ls = bs + mask * (ts - bs)
            feasible, excess = boundary_feasible(bw, bs, lw, ls, start, stop, stats)
            rows.append({
                "start": start, "stop": stop, "duration_h": length,
                "energy_density": float(np.mean(energy[start:stop])),
                "structure": structure, "low": low, "ramp": ramp, "recovery": recovery,
                "boundary_feasible": feasible, "boundary_excess": excess,
            })
    e_rank = percentile_ranks(np.asarray([r["energy_density"] for r in rows]))
    s_rank = percentile_ranks(np.asarray([r["structure"] for r in rows]))
    for row, er, sr in zip(rows, e_rank, s_rank):
        row["energy_rank"] = float(er)
        row["structure_rank"] = float(sr)
        row["combined_score"] = float(math.sqrt(er * sr))
    return rows


def select_windows(
    energy: np.ndarray,
    background: np.ndarray,
    tail: np.ndarray,
    daylight: np.ndarray,
    stations: pd.DataFrame,
    fixed_durations: np.ndarray,
    adaptive_lengths: list[int],
    taper: int,
    stats: dict[str, object],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    issue_count, member_count, lead_count = energy.shape
    selected = {name: np.zeros((issue_count, member_count, 2), np.int16) for name in VARIANTS}
    records = []
    for issue in range(issue_count):
        for member in range(member_count):
            common = dict(
                energy=energy[issue, member], background=background[issue, member],
                tail=tail[issue, member], daylight=daylight[issue], stations=stations,
                taper=taper, stats=stats,
            )
            fixed_rows = candidate_rows_for_member(lengths=[int(fixed_durations[issue, member])], **common)
            adaptive_rows = candidate_rows_for_member(lengths=adaptive_lengths, **common)
            choices = {
                "s5_event_fixed": (max(fixed_rows, key=lambda r: r["combined_score"]), False),
                "s5_event_adaptive": (max(adaptive_rows, key=lambda r: r["combined_score"]), False),
            }
            feasible = [r for r in adaptive_rows if r["boundary_feasible"]]
            fallback = not feasible
            choices["s5_espc"] = (max(feasible or adaptive_rows, key=lambda r: r["combined_score"]), fallback)
            structural_best = max(adaptive_rows, key=lambda r: r["structure"])
            for variant, (row, used_fallback) in choices.items():
                selected[variant][issue, member] = (row["start"], row["stop"])
                records.append({
                    "issue": issue, "member": member, "variant": variant,
                    **row, "fallback": used_fallback,
                    "structural_reference_start": structural_best["start"],
                    "structural_reference_stop": structural_best["stop"],
                    "structural_reference_overlap_fraction": (
                        max(0, min(row["stop"], structural_best["stop"]) - max(row["start"], structural_best["start"]))
                        / max(structural_best["duration_h"], 1)
                    ),
                })
    masks = {}
    for variant, windows in selected.items():
        masks[variant] = build_tapered_masks(windows[..., 0], windows[..., 1], lead_count, taper)
    return masks, pd.DataFrame(records)


def interval_bundle(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    lower, upper = np.quantile(samples, (0.05, 0.95), axis=1)
    width = upper - lower
    score = width + 20.0 * (lower - truth) * (truth < lower) + 20.0 * (truth - upper) * (truth > upper)
    return {
        "coverage_90": float(np.mean((truth >= lower) & (truth <= upper))),
        "interval_width_90": float(np.mean(width)),
        "interval_score_90": float(np.mean(score)),
    }


def save_variant(
    output: Path,
    variant: str,
    masks: np.ndarray,
    body_dir: Path,
    tail_dir: Path,
    body_members: int,
    tail_members: int,
    stations: pd.DataFrame,
    data_path: Path,
    metadata: dict[str, object],
    energy_limit: int,
    background_indices: np.ndarray | None = None,
) -> None:
    output.mkdir(parents=True)
    b, k = body_members, tail_members
    indices = np.arange(b, b + k) if background_indices is None else background_indices
    for name in MEMBER_ARRAYS:
        body = _load(body_dir, name)
        tail = _load(tail_dir, name)
        localized = localize_members(np.asarray(body[:, indices]), np.asarray(tail[:, :k]), masks)
        np.save(output / name, np.concatenate((np.asarray(body[:, :b]), localized), axis=1))
    for name in STATIC_ARRAYS:
        np.save(output / name, np.asarray(_load(body_dir, name)))
    np.save(output / "localized_tail_mask.npy", masks)
    route = np.zeros((masks.shape[0], b + k), np.float32); route[:, b:] = 1
    np.save(output / "tail_expert_route.npy", route)
    np.save(output / "tail_expert_probability.npy", np.full(masks.shape[0], k / (b + k), np.float32))
    samples = _load(output, "actual_scenarios_normalized.npy")
    actual = _load(output, "actual_data_normalized.npy")
    summary, station_frame, lead_frame = evaluate_station_scenarios(
        samples, _load(output, "actual_scenarios_raw_normalized.npy"), actual,
        _load(output, "forecast_data_normalized.npy"), stations,
        np.load(data_path / "station_adjacency.npy"),
        daylight_mask=_load(output, "station_daylight_mask.npy"),
        interval_levels=(0.80, 0.90, 0.95, 0.99),
        energy_score_member_limit=energy_limit,
    )
    capacity = stations.capacity_mw.to_numpy(float)
    renewable = np.einsum("nkts,s->nkt", samples, capacity)
    renewable_truth = np.einsum("nts,s->nt", actual, capacity)
    summary["espc_interval"] = {"renewable_mw": interval_bundle(renewable, renewable_truth)}
    run = {**metadata, "condition_variant": variant, "n_samples": b + k,
           "evaluation_member_count": b + k, "body_members": b, "tail_members": k,
           "tail_fraction": k / (b + k), "test_used": False}
    summary["run"] = run
    save_evaluation(output, summary, station_frame, lead_frame)
    (output / "generation_metadata.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")


def reproduce_original(
    body_dir: Path, tail_dir: Path, original: Path, data_path: Path,
    stations: pd.DataFrame, b: int, k: int, seed: int, taper: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    body = _load(body_dir, MEMBER_ARRAYS[0]); tail = _load(tail_dir, MEMBER_ARRAYS[0])
    background = np.asarray(body[:, b:b + k]); tail100 = np.asarray(tail[:, :k])
    durations, audit = stratified_train_durations(data_path, k, body.shape[0], seed)
    energy = joint_correction_energy(tail100, background, stations)
    starts, stops = strongest_windows(energy, durations)
    masks = build_tapered_masks(starts, stops, body.shape[2], taper)
    existing_mask = np.load(original / "localized_tail_mask.npy", mmap_mode="r")
    existing_scenarios = np.load(original / MEMBER_ARRAYS[0], mmap_mode="r")
    reconstructed = np.concatenate((np.asarray(body[:, :b]), localize_members(background, tail100, masks)), axis=1)
    report = {
        "status": "PASS" if np.array_equal(masks, existing_mask) and np.array_equal(reconstructed, existing_scenarios) else "FAIL",
        "mask_exact": bool(np.array_equal(masks, existing_mask)),
        "scenario_exact": bool(np.array_equal(reconstructed, existing_scenarios)),
        "mask_max_abs_error": float(np.max(np.abs(masks - existing_mask))),
        "scenario_max_abs_error": float(np.max(np.abs(reconstructed - existing_scenarios))),
        "duration_audit": audit,
    }
    return energy, durations, report


def main() -> None:
    args = parse_args()
    body_dir, tail_dir, original = map(Path, (args.body_results, args.tail_results, args.original_s5_results))
    root, data_path = Path(args.output_root), Path(args.data_path)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    started = time.perf_counter()
    body_meta = json.loads((body_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    tail_meta = json.loads((tail_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    for meta in (body_meta, tail_meta):
        if meta.get("split") != "val" or meta.get("future_actual_used_as_generation_condition", False):
            raise ValueError("only causal validation members are allowed")
    for name in STATIC_ARRAYS:
        _assert_same(_load(body_dir, name), _load(tail_dir, name), name)
    stations = pd.read_csv(data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    b, k = args.body_members, args.tail_members
    energy, fixed_durations, reproduction = reproduce_original(
        body_dir, tail_dir, original, data_path, stations, b, k, args.seed, args.edge_taper_hours
    )
    (root / "S5_REPRODUCTION.json").write_text(json.dumps(reproduction, indent=2), encoding="utf-8")
    if reproduction["status"] != "PASS":
        raise RuntimeError("original S5 reproduction failed; ESPC comparison is blocked")
    train_stats = fit_train_statistics(data_path, stations)
    (root / "TRAIN_FITTED_ESPC_PARAMETERS.json").write_text(json.dumps(train_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    body = _load(body_dir, MEMBER_ARRAYS[0]); tail = _load(tail_dir, MEMBER_ARRAYS[0])
    background = np.asarray(body[:, b:b + k]); tail100 = np.asarray(tail[:, :k])
    daylight = np.asarray(_load(body_dir, "station_daylight_mask.npy"), bool)
    masks, records = select_windows(
        energy, background, tail100, daylight, stations, fixed_durations,
        list(train_stats["duration_candidates_h"]), args.edge_taper_hours, train_stats,
    )
    records.to_csv(root / "window_selection_records.csv", index=False)
    metadata = {
        "family": "zero_training_s5_espc_v1", "split": "val", "generation_seed": args.seed,
        "source_body_results": str(body_dir), "source_tail_results": str(tail_dir),
        "future_actual_used_for_selection": False, "validation_event_labels_used_for_selection": False,
        "train_fitted_parameters": "TRAIN_FITTED_ESPC_PARAMETERS.json",
        "edge_taper_hours": args.edge_taper_hours,
        "adaptive_lengths_h": train_stats["duration_candidates_h"],
        "selection": "equal geometric mean of within-member energy and event-structure percentile ranks",
        "structure": "max(sustained-low, 1/3/6h-ramp, optional drop-recovery)",
        "boundary_constraint": "localized boundary ramp <= max(train q99, paired Raw boundary ramp)",
        "fallback": "unconstrained adaptive event-aware winner when no feasible candidate exists",
        "test_used": False,
    }
    for variant in VARIANTS:
        save_variant(root / variant, variant, masks[variant], body_dir, tail_dir, b, k,
                     stations, data_path, metadata, args.energy_score_member_limit)
        issue_count, member_count = masks[variant].shape[:2]
        selected = records[records.variant.eq(variant)].sort_values(["issue", "member"])
        np.save(
            root / variant / "localized_tail_start.npy",
            selected["start"].to_numpy(np.int16).reshape(issue_count, member_count),
        )
        np.save(
            root / variant / "localized_tail_stop.npy",
            selected["stop"].to_numpy(np.int16).reshape(issue_count, member_count),
        )

    pairing_rows = []
    for pairing_seed in PAIRING_SEEDS:
        rng = np.random.default_rng(pairing_seed)
        indices = np.empty((body.shape[0], k), int)
        permuted_masks = np.empty_like(masks["s5_event_fixed"])
        for issue in range(body.shape[0]):
            indices[issue] = b + rng.permutation(k)
            paired_background = np.asarray(body[issue, indices[issue]])
            paired_energy = joint_correction_energy(tail100[issue:issue + 1], paired_background[None], stations)[0]
            starts, stops = strongest_windows(paired_energy[None], fixed_durations[issue:issue + 1])
            permuted_masks[issue] = build_tapered_masks(starts, stops, body.shape[2], args.edge_taper_hours)[0]
        # Variable issue-wise indices require direct construction for the main scenario array.
        paired = np.stack([np.asarray(body[i, indices[i]]) for i in range(body.shape[0])])
        localized = localize_members(paired, tail100, permuted_masks)
        samples = np.concatenate((np.asarray(body[:, :b]), localized), axis=1)
        actual = _load(body_dir, "actual_data_normalized.npy")
        capacity = stations.capacity_mw.to_numpy(float)
        renewable = np.einsum("nkts,s->nkt", samples, capacity)
        truth = np.einsum("nts,s->nt", actual, capacity)
        bundle = interval_bundle(renewable, truth)
        pairing_rows.append({
            "pairing": f"random_seed_{pairing_seed}",
            "mean_selected_fraction": float(np.mean(permuted_masks > 0)),
            **bundle,
        })
    pd.DataFrame(pairing_rows).to_csv(root / "pairing_sensitivity.csv", index=False)
    manifest = {
        "status": "PASS", "formal_training": "NOT RUN", "optimizer_steps": 0,
        "new_generation": "NOT RUN", "checkpoint_modified": False,
        "original_s5_reproduced": True, "variants": list(VARIANTS),
        "pairing_seeds": list(PAIRING_SEEDS), "elapsed_seconds": time.perf_counter() - started,
        "source_hashes": {
            "body_metadata": sha256(body_dir / "generation_metadata.json"),
            "tail_metadata": sha256(tail_dir / "generation_metadata.json"),
            "original_s5_metadata": sha256(original / "generation_metadata.json"),
        },
    }
    (root / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"S5_ESPC_VARIANTS_COMPLETE root={root}")


if __name__ == "__main__":
    main()
