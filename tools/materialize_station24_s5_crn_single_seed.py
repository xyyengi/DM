#!/usr/bin/env python3
"""Build frozen Full-V2, identity-CRN and three permutation S5 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from station_evaluation import evaluate_station_scenarios, save_evaluation
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", required=True, type=Path)
    parser.add_argument("--tail-results", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--generation-seed", required=True, type=int)
    parser.add_argument("--duration-seed", required=True, type=int)
    parser.add_argument("--permutation-seeds", required=True, nargs=3, type=int)
    parser.add_argument("--body-members", default=400, type=int)
    parser.add_argument("--background-members", default=100, type=int)
    parser.add_argument("--tail-members", default=100, type=int)
    parser.add_argument("--edge-taper-hours", default=3, type=int)
    parser.add_argument("--energy-score-member-limit", default=80, type=int)
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    return json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))


def build_pairings(
    issues: int, body_members: int, background_members: int, seeds: list[int]
) -> dict[str, np.ndarray]:
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("three distinct permutation seeds are required")
    identity = np.broadcast_to(
        np.arange(body_members, body_members + background_members),
        (issues, background_members),
    ).copy()
    pairings = {"crn_identity": identity}
    for number, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        pairings[f"permutation_{number}"] = np.stack(
            [
                body_members + rng.permutation(background_members)
                for _ in range(issues)
            ]
        )
    return pairings


def save_arm(
    output: Path,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    stations: pd.DataFrame,
    adjacency: np.ndarray,
    metadata: dict,
    member_limit: int,
) -> None:
    output.mkdir(parents=True)
    for name, values in arrays.items():
        np.save(output / name, values)
    for name, values in static.items():
        np.save(output / name, values)
    samples = arrays["actual_scenarios_normalized.npy"]
    raw_samples = arrays["actual_scenarios_raw_normalized.npy"]
    summary, station_frame, lead_frame = evaluate_station_scenarios(
        samples,
        raw_samples,
        static["actual_data_normalized.npy"],
        static["forecast_data_normalized.npy"],
        stations,
        adjacency,
        daylight_mask=static["station_daylight_mask.npy"],
        interval_levels=(0.80, 0.90, 0.95, 0.99),
        energy_score_member_limit=member_limit,
    )
    summary["run"] = metadata
    save_evaluation(output, summary, station_frame, lead_frame)
    (output / "generation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    issues, members = samples.shape[:2]
    route = np.zeros((issues, members), dtype=np.float32)
    route[:, -int(metadata["tail_members"]):] = 1.0
    np.save(output / "tail_expert_route.npy", route)
    np.save(
        output / "tail_expert_probability.npy",
        np.full(issues, metadata["tail_members"] / members, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if len(set(args.permutation_seeds)) != 3:
        raise ValueError("three distinct permutation seeds are required")
    b, r, k = args.body_members, args.background_members, args.tail_members
    if (b, r, k) != (400, 100, 100):
        raise ValueError("frozen budget is 400 Raw body + 100 Raw background + 100 Tail")

    raw_meta, tail_meta = load_metadata(args.raw_results), load_metadata(args.tail_results)
    for label, meta, expected_members, expected_offset in (
        ("Raw", raw_meta, b + r, 0), ("Tail", tail_meta, k, b)
    ):
        if meta.get("split") != "val" or bool(meta.get("test_used", False)):
            raise ValueError(f"{label} source is not locked validation output")
        if meta.get("ancestral_noise_mode") != "keyed_full_path":
            raise ValueError(f"{label} source is not keyed full-path CRN")
        if not bool(meta.get("ancestral_noise_controls_initial_and_all_reverse_steps")):
            raise ValueError(f"{label} lacks full-path noise audit")
        if int(meta.get("generation_seed", -1)) != args.generation_seed:
            raise ValueError(f"{label} generation seed mismatch")
        if int(meta.get("ancestral_noise_seed", -1)) != args.generation_seed:
            raise ValueError(f"{label} ancestral seed mismatch")
        if int(meta.get("n_samples", -1)) != expected_members:
            raise ValueError(f"{label} member count mismatch")
        if int(meta.get("ancestral_member_offset", -1)) != expected_offset:
            raise ValueError(f"{label} member offset mismatch")
        if int(meta.get("ancestral_member_block_size", -1)) != 100:
            raise ValueError(f"{label} canonical CRN block mismatch")

    for name in STATIC_ARRAYS:
        _assert_same(_load(args.raw_results, name), _load(args.tail_results, name), name)
    raw_scenarios = _load(args.raw_results, MEMBER_ARRAYS[0])
    tail_scenarios = _load(args.tail_results, MEMBER_ARRAYS[0])
    if raw_scenarios.shape[1] != b + r or tail_scenarios.shape[1] != k:
        raise ValueError("sources must be exactly Raw500 and Tail100")
    issues, _, leads, _ = raw_scenarios.shape
    stations = pd.read_csv(args.data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    adjacency = np.load(args.data_path / "station_adjacency.npy")
    static = {name: np.asarray(_load(args.raw_results, name)) for name in STATIC_ARRAYS}
    durations, duration_audit = stratified_train_durations(
        args.data_path, k, issues, args.duration_seed
    )
    pairings = build_pairings(issues, b, r, list(args.permutation_seeds))
    for name, indices in pairings.items():
        if any(np.unique(row).size != r for row in indices):
            raise RuntimeError(f"invalid permutation in {name}")

    args.output_root.mkdir(parents=True)
    np.save(args.output_root / "duration_assignment.npy", durations)
    for name, indices in pairings.items():
        np.save(args.output_root / f"{name}_background_index.npy", indices)

    # Only the two arrays consumed by formal evaluation are duplicated per arm;
    # immutable residual representations remain in their source directories.
    evaluation_arrays = MEMBER_ARRAYS[:2]
    full_arrays = {
        name: np.concatenate(
            (np.asarray(_load(args.raw_results, name)[:, :b]),
             np.asarray(_load(args.tail_results, name)[:, :k])), axis=1
        )
        for name in evaluation_arrays
    }
    common = {
        **{key: raw_meta[key] for key in (
            "split", "generation_seed", "physical_projection", "architecture",
            "spatial_mode", "spatial_mix_levels", "parallel_spatial_fusion_levels",
            "parallel_spatial_adjacency_mode",
        ) if key in raw_meta},
        **{key: tail_meta[key] for key in (
            "parameter_count", "checkpoint_validation_objective",
            "checkpoint_validation_mse", "checkpoint_validation_objective_type",
            "checkpoint_epoch",
        ) if key in tail_meta},
        "family": "station24_s5_crn_single_seed_v1",
        "split": "val",
        "generation_seed": args.generation_seed,
        "duration_seed": args.duration_seed,
        "permutation_seeds": args.permutation_seeds,
        "body_members": b,
        "background_members": r,
        "tail_members": k,
        "n_samples": b + k,
        "raw_results": str(args.raw_results),
        "tail_results": str(args.tail_results),
        "ancestral_noise_mode": "keyed_full_path",
        "ancestral_noise_scheme": raw_meta["ancestral_noise_scheme"],
        "duration_audit": duration_audit,
        "original_s5_rule": "maximum_joint_correction_energy_single_window",
        "edge_taper_hours": args.edge_taper_hours,
        "same_raw_and_tail_marginals_across_pairing_arms": True,
        "future_actual_used_for_pairing_or_selection": False,
        "event_labels_used_for_pairing_or_selection": False,
        "test_used": False,
        "training": "NOT APPLICABLE",
    }
    save_arm(
        args.output_root / "full_v2_mixture", full_arrays, static, stations,
        adjacency, {**common, "condition_variant": "full_v2_crn_mixture",
                    "coupling": "full_tail_not_localized"},
        args.energy_score_member_limit,
    )

    tail100 = np.asarray(tail_scenarios[:, :k])
    arm_manifest = {}
    for name, indices in pairings.items():
        background = np.stack(
            [np.asarray(raw_scenarios[i, indices[i]]) for i in range(issues)]
        )
        energy = joint_correction_energy(tail100, background, stations)
        starts, stops = strongest_windows(energy, durations)
        masks = build_tapered_masks(starts, stops, leads, args.edge_taper_hours)
        arrays = {}
        for array_name in evaluation_arrays:
            raw_values = _load(args.raw_results, array_name)
            tail_values = _load(args.tail_results, array_name)
            paired = np.stack(
                [np.asarray(raw_values[i, indices[i]]) for i in range(issues)]
            )
            localized = localize_members(paired, np.asarray(tail_values[:, :k]), masks)
            arrays[array_name] = np.concatenate(
                (np.asarray(raw_values[:, :b]), localized), axis=1
            )
        seed_value = None if name == "crn_identity" else args.permutation_seeds[int(name[-1]) - 1]
        metadata = {
            **common,
            "condition_variant": f"original_s5_{name}",
            "coupling": "full_path_crn_identity" if name == "crn_identity" else "fixed_issue_wise_random_permutation",
            "pairing_seed": seed_value,
        }
        output = args.output_root / name
        save_arm(
            output, arrays, static, stations, adjacency, metadata,
            args.energy_score_member_limit,
        )
        np.save(output / "localized_tail_mask.npy", masks)
        np.save(output / "localized_tail_start.npy", starts)
        np.save(output / "localized_tail_stop.npy", stops)
        np.save(output / "localized_tail_duration.npy", durations)
        np.save(output / "paired_raw_background_index.npy", indices)
        np.save(output / "pair_correction_energy.npy", energy.astype(np.float32))
        arm_manifest[name] = {"path": str(output), "pairing_seed": seed_value}
        print(f"S5_CRN_COUPLING_COMPLETE arm={name} output={output}")

    manifest = {**common, "arms": arm_manifest, "full_v2_mixture": str(args.output_root / "full_v2_mixture")}
    (args.output_root / "coupling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"S5_CRN_SINGLE_SEED_MATERIALIZATION_COMPLETE root={args.output_root}")


if __name__ == "__main__":
    main()
