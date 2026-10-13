#!/usr/bin/env python3
"""Zero-training temporal localization of an existing independent joint tail.

This is a diagnostic transformation, not a newly trained forecasting model.
For every generated tail member, the strongest contiguous correction window is
selected using only the generated tail/body pair.  Window durations are a
deterministic stratified sample from the train-only continuous-event catalogue.
Outside the selected window the member is exactly the frozen Raw background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from station_evaluation import evaluate_station_scenarios, save_evaluation
from station_jstd_targets import (
    build_station_jstd_target_arrays,
    fit_station_jstd_event_thresholds,
)


MEMBER_ARRAYS = (
    "actual_scenarios_normalized.npy",
    "actual_scenarios_raw_normalized.npy",
    "generated_residual_normalized.npy",
    "generated_residual_standardized.npy",
    "generated_stochastic_residual_standardized.npy",
)
STATIC_ARRAYS = (
    "actual_data_normalized.npy",
    "forecast_data_normalized.npy",
    "station_daylight_mask.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-results", required=True)
    parser.add_argument("--tail-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--body-members", type=int, default=400)
    parser.add_argument("--tail-members", type=int, default=100)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--edge-taper-hours", type=int, default=3)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--energy-score-member-limit", type=int, default=80)
    parser.add_argument("--condition-variant", default="independent_joint_tail_v2_self_localized")
    return parser.parse_args()


def _load(folder: Path, name: str) -> np.ndarray:
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"missing required array: {path}")
    return np.load(path, mmap_mode="r")


def _assert_same(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"body/tail issuance mismatch: {name}")


def stratified_train_durations(
    data_path: str | Path,
    member_count: int,
    issue_count: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return issue-wise shuffled empirical duration quantiles fitted on train."""

    thresholds = fit_station_jstd_event_thresholds(data_path)
    targets = build_station_jstd_target_arrays(data_path, "train", thresholds)
    durations = np.asarray(
        [row["actual_duration_hours"] for row in targets.audit["segment_catalog"]],
        dtype=np.float64,
    )
    if durations.size == 0:
        raise ValueError("train-only continuous-event catalogue is empty")
    quantiles = (np.arange(member_count, dtype=np.float64) + 0.5) / member_count
    try:
        base = np.quantile(durations, quantiles, method="nearest")
    except TypeError:  # NumPy < 1.22
        base = np.quantile(durations, quantiles, interpolation="nearest")
    base = np.clip(np.rint(base), 1, 168).astype(np.int16)
    rng = np.random.default_rng(seed)
    result = np.empty((issue_count, member_count), dtype=np.int16)
    for issue in range(issue_count):
        result[issue] = base[rng.permutation(member_count)]
    audit = {
        "fit_split": "train",
        "future_actual_used_as_generation_condition": False,
        "catalog_event_count": int(durations.size),
        "catalog_duration_min_h": int(durations.min()),
        "catalog_duration_median_h": float(np.median(durations)),
        "catalog_duration_max_h": int(durations.max()),
        "sampling": "deterministic_stratified_empirical_quantiles_then_issue_shuffle",
    }
    return result, audit


def joint_correction_energy(
    tail: np.ndarray,
    background: np.ndarray,
    stations: pd.DataFrame,
) -> np.ndarray:
    """Capacity-weighted wind+solar correction energy [issue, member, lead]."""

    delta = np.asarray(tail, dtype=np.float64) - np.asarray(background, dtype=np.float64)
    capacity = stations.capacity_mw.to_numpy(dtype=np.float64)
    components = []
    for source in ("wind", "solar"):
        index = stations.index[stations.data_type.eq(source)].to_numpy(dtype=int)
        weight = capacity[index] / capacity[index].sum()
        components.append(np.abs(np.einsum("imts,s->imt", delta[..., index], weight)))
    return components[0] + components[1]


def strongest_windows(energy: np.ndarray, durations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the maximum-energy rolling interval for every issue/member."""

    if energy.ndim != 3 or durations.shape != energy.shape[:2]:
        raise ValueError("energy/duration shapes do not align")
    issue_count, member_count, lead_count = energy.shape
    starts = np.empty((issue_count, member_count), dtype=np.int16)
    stops = np.empty_like(starts)
    prefix = np.pad(np.cumsum(energy, axis=-1), ((0, 0), (0, 0), (1, 0)))
    for issue in range(issue_count):
        for member in range(member_count):
            duration = int(np.clip(durations[issue, member], 1, lead_count))
            sums = prefix[issue, member, duration:] - prefix[issue, member, :-duration]
            start = int(np.argmax(sums))
            starts[issue, member] = start
            stops[issue, member] = start + duration
    return starts, stops


def build_tapered_masks(
    starts: np.ndarray,
    stops: np.ndarray,
    lead_count: int,
    taper_hours: int,
) -> np.ndarray:
    """Construct compact-support masks with ramps inside the selected interval."""

    if starts.shape != stops.shape or taper_hours < 0:
        raise ValueError("invalid starts/stops/taper")
    masks = np.zeros(starts.shape + (lead_count,), dtype=np.float32)
    for index in np.ndindex(starts.shape):
        start, stop = int(starts[index]), int(stops[index])
        if not 0 <= start < stop <= lead_count:
            raise ValueError("window outside lead range")
        duration = stop - start
        width = min(int(taper_hours), duration // 2)
        window = np.ones(duration, dtype=np.float32)
        if width > 0:
            ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(width) + 1) / (width + 1))
            window[:width] = ramp
            window[-width:] = ramp[::-1]
        masks[index + (slice(start, stop),)] = window
    return masks


def localize_members(background: np.ndarray, tail: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if background.shape != tail.shape or masks.shape != background.shape[:3]:
        raise ValueError("localization shapes do not align")
    return np.asarray(background) + masks[..., None] * (np.asarray(tail) - np.asarray(background))


def main() -> None:
    args = parse_args()
    body_dir, tail_dir, output = Path(args.body_results), Path(args.tail_results), Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    body_meta = json.loads((body_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    tail_meta = json.loads((tail_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    for meta in (body_meta, tail_meta):
        if meta.get("future_actual_used_as_generation_condition", False):
            raise ValueError("noncausal/oracle source cannot enter localization")
        if meta.get("split") != "val":
            raise ValueError("localization diagnostic is validation-only")
    for name in STATIC_ARRAYS:
        _assert_same(_load(body_dir, name), _load(tail_dir, name), name)

    scenario_body = _load(body_dir, MEMBER_ARRAYS[0])
    scenario_tail = _load(tail_dir, MEMBER_ARRAYS[0])
    b, k = int(args.body_members), int(args.tail_members)
    if scenario_body.shape[1] < b + k or scenario_tail.shape[1] < k:
        raise ValueError("sources do not contain enough disjoint body/background and tail members")
    issue_count, _, lead_count, _ = scenario_body.shape
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    background_slice = slice(b, b + k)
    background = np.asarray(scenario_body[:, background_slice])
    tail = np.asarray(scenario_tail[:, :k])
    durations, duration_audit = stratified_train_durations(args.data_path, k, issue_count, args.seed)
    if not 0.5 <= float(args.duration_scale) <= 3.0:
        raise ValueError("duration-scale must be in [0.5,3.0]")
    durations = np.clip(
        np.rint(durations.astype(np.float64) * float(args.duration_scale)), 1, lead_count
    ).astype(np.int16)
    energy = joint_correction_energy(tail, background, stations)
    starts, stops = strongest_windows(energy, durations)
    masks = build_tapered_masks(starts, stops, lead_count, int(args.edge_taper_hours))

    output.mkdir(parents=True)
    for name in MEMBER_ARRAYS:
        body_values = _load(body_dir, name)
        tail_values = _load(tail_dir, name)
        if body_values.shape[1] < b + k or tail_values.shape[1] < k:
            raise ValueError(f"insufficient members for {name}")
        localized = localize_members(
            np.asarray(body_values[:, background_slice]),
            np.asarray(tail_values[:, :k]),
            masks,
        )
        np.save(output / name, np.concatenate((np.asarray(body_values[:, :b]), localized), axis=1))
    for name in STATIC_ARRAYS:
        np.save(output / name, np.asarray(_load(body_dir, name)))
    np.save(output / "localized_tail_mask.npy", masks)
    np.save(output / "localized_tail_start.npy", starts)
    np.save(output / "localized_tail_stop.npy", stops)
    np.save(output / "localized_tail_duration.npy", durations)
    total = b + k
    route = np.zeros((issue_count, total), dtype=np.float32)
    route[:, b:] = 1.0
    np.save(output / "tail_expert_route.npy", route)
    np.save(output / "tail_expert_probability.npy", np.full(issue_count, k / total, dtype=np.float32))

    samples = _load(output, "actual_scenarios_normalized.npy")
    raw_samples = _load(output, "actual_scenarios_raw_normalized.npy")
    actual = _load(output, "actual_data_normalized.npy")
    forecast = _load(output, "forecast_data_normalized.npy")
    daylight = _load(output, "station_daylight_mask.npy")
    adjacency = np.load(Path(args.data_path) / "station_adjacency.npy")
    summary, station_frame, lead_frame = evaluate_station_scenarios(
        samples, raw_samples, actual, forecast, stations, adjacency,
        daylight_mask=daylight,
        interval_levels=(0.80, 0.90, 0.95, 0.99),
        energy_score_member_limit=int(args.energy_score_member_limit),
    )
    run = {
        **{key: body_meta[key] for key in (
            "split", "generation_seed", "physical_projection", "architecture",
            "spatial_mode", "spatial_mix_levels", "parallel_spatial_fusion_levels",
            "parallel_spatial_adjacency_mode",
        ) if key in body_meta},
        **{key: tail_meta[key] for key in (
            "parameter_count", "checkpoint_validation_objective",
            "checkpoint_validation_mse", "checkpoint_validation_objective_type",
            "checkpoint_epoch",
        ) if key in tail_meta},
        "condition_variant": args.condition_variant,
        "family": "zero_training_self_localized_independent_joint_tail_diagnostic_v1",
        "n_samples": total,
        "evaluation_member_count": total,
        "body_members": b,
        "tail_members": k,
        "tail_fraction": k / total,
        "body_results": str(body_dir),
        "tail_results": str(tail_dir),
        "background_member_range": [b, b + k],
        "localization_signal": "generated_joint_wind_solar_tail_minus_raw_correction_energy",
        "localization_window": "maximum_energy_contiguous_train_duration_window",
        "edge_taper_hours": int(args.edge_taper_hours),
        "duration_scale": float(args.duration_scale),
        "duration_audit": duration_audit,
        "future_actual_used_as_generation_condition": False,
        "event_labels_used_as_generation_condition": False,
        "reportable_as_causal_forecast": True,
        "diagnostic_not_trained_model": True,
        "scientific_scope": "tests whether global tail action causes excess width; does not prove forecast-to-event identifiability",
    }
    summary["run"] = run
    save_evaluation(output, summary, station_frame, lead_frame)
    (output / "generation_metadata.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SELF_LOCALIZED_JOINT_TAIL_COMPLETE output={output} body={b} tail={k}")


if __name__ == "__main__":
    main()


