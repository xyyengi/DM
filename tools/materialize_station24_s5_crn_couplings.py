#!/usr/bin/env python3
"""Materialize frozen Original-S5 identity-CRN and permuted couplings.

Both S5 arms reuse exactly the same generated Raw500 and Full-V2 Tail100
marginals.  The only changed factor is which Raw background member is paired
with each fixed Tail member.  No validation truth enters pairing or windows.
"""

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
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--tail-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("diffusion_input_station"))
    parser.add_argument("--generation-seed", type=int, required=True)
    parser.add_argument("--duration-seed", type=int, default=424242)
    parser.add_argument("--pairing-seed", type=int, default=424242)
    parser.add_argument("--body-members", type=int, default=400)
    parser.add_argument("--tail-members", type=int, default=100)
    parser.add_argument("--edge-taper-hours", type=int, default=3)
    parser.add_argument("--energy-score-member-limit", type=int, default=80)
    return parser.parse_args()


def metadata(path: Path) -> dict[str, object]:
    return json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))


def evaluate_and_save(
    output: Path,
    arrays: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    stations: pd.DataFrame,
    adjacency: np.ndarray,
    run: dict[str, object],
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
        energy_score_member_limit=int(member_limit),
    )
    summary["run"] = run
    save_evaluation(output, summary, station_frame, lead_frame)
    (output / "generation_metadata.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    raw_meta, tail_meta = metadata(args.raw_results), metadata(args.tail_results)
    for label, meta, expected_members, expected_offset in (
        ("Raw", raw_meta, 500, 0),
        ("Tail", tail_meta, 100, 400),
    ):
        if meta.get("split") != "val" or bool(meta.get("test_used", False)):
            raise ValueError(f"{label} source is not locked validation output")
        if meta.get("ancestral_noise_mode") != "keyed_full_path":
            raise ValueError(f"{label} source is not full-path keyed CRN")
        if not bool(meta.get("ancestral_noise_controls_initial_and_all_reverse_steps")):
            raise ValueError(f"{label} source lacks full-path noise control audit")
        if int(meta.get("generation_seed", -1)) != int(args.generation_seed):
            raise ValueError(f"{label} generation seed mismatch")
        if int(meta.get("ancestral_noise_seed", -1)) != int(args.generation_seed):
            raise ValueError(f"{label} ancestral noise seed mismatch")
        if int(meta.get("n_samples", -1)) != expected_members:
            raise ValueError(f"{label} member count mismatch")
        if int(meta.get("ancestral_member_offset", -1)) != expected_offset:
            raise ValueError(f"{label} member offset mismatch")
        if int(meta.get("ancestral_member_block_size", -1)) != 100:
            raise ValueError(f"{label} CRN member block must be exactly 100")

    for name in STATIC_ARRAYS:
        _assert_same(_load(args.raw_results, name), _load(args.tail_results, name), name)
    raw_scenarios = _load(args.raw_results, MEMBER_ARRAYS[0])
    tail_scenarios = _load(args.tail_results, MEMBER_ARRAYS[0])
    b, k = int(args.body_members), int(args.tail_members)
    if raw_scenarios.shape[1] != b + k or tail_scenarios.shape[1] != k:
        raise ValueError("CRN sources must contain exactly Raw500 and Tail100")
    issues, _, leads, _ = raw_scenarios.shape
    stations = pd.read_csv(args.data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    adjacency = np.load(args.data_path / "station_adjacency.npy")
    static = {name: np.asarray(_load(args.raw_results, name)) for name in STATIC_ARRAYS}
    durations, duration_audit = stratified_train_durations(
        args.data_path, k, issues, int(args.duration_seed)
    )

    identity = np.broadcast_to(np.arange(b, b + k), (issues, k)).copy()
    rng = np.random.default_rng(int(args.pairing_seed))
    permuted = np.stack([b + rng.permutation(k) for _ in range(issues)])
    if any(np.unique(row).size != k for row in permuted):
        raise RuntimeError("invalid issue-wise pairing permutation")

    args.output_root.mkdir(parents=True)
    np.save(args.output_root / "duration_assignment.npy", durations)
    np.save(args.output_root / "crn_identity_background_index.npy", identity)
    np.save(args.output_root / "independent_permutation_background_index.npy", permuted)

    # Post-hoc S5 evaluation needs the projected and raw actual members.  The
    # three residual representations remain available in the immutable Raw/Tail
    # source directories and are not duplicated into every coupling arm.
    evaluation_member_arrays = MEMBER_ARRAYS[:2]
    full_arrays = {
        name: np.concatenate(
            (np.asarray(_load(args.raw_results, name)[:, :b]),
             np.asarray(_load(args.tail_results, name)[:, :k])),
            axis=1,
        )
        for name in evaluation_member_arrays
    }
    common = {
        "family": "station24_s5_full_path_crn_attribution_v1",
        "split": "val",
        "generation_seed": int(args.generation_seed),
        "duration_seed": int(args.duration_seed),
        "pairing_seed": int(args.pairing_seed),
        "body_members": b,
        "tail_members": k,
        "n_samples": b + k,
        "raw_results": str(args.raw_results),
        "tail_results": str(args.tail_results),
        "ancestral_noise_mode": "keyed_full_path",
        "ancestral_noise_scheme": raw_meta["ancestral_noise_scheme"],
        "crn_raw_background_member_offset": 400,
        "crn_tail_member_offset": 400,
        "duration_audit": duration_audit,
        "original_s5_rule": "maximum_joint_correction_energy_single_window",
        "edge_taper_hours": int(args.edge_taper_hours),
        "future_actual_used_for_pairing_or_selection": False,
        "event_labels_used_for_pairing_or_selection": False,
        "test_used": False,
        "training": "NOT APPLICABLE",
    }
    full_run = {**common, "condition_variant": "full_v2_crn_mixture",
                "coupling": "not_localized_full_tail_reference"}
    evaluate_and_save(
        args.output_root / "full_v2_mixture",
        full_arrays,
        static,
        stations,
        adjacency,
        full_run,
        args.energy_score_member_limit,
    )
    full_route = np.zeros((issues, b + k), dtype=np.float32)
    full_route[:, b:] = 1.0
    np.save(args.output_root / "full_v2_mixture" / "tail_expert_route.npy", full_route)
    np.save(
        args.output_root / "full_v2_mixture" / "tail_expert_probability.npy",
        np.full(issues, k / (b + k), dtype=np.float32),
    )

    tail100 = np.asarray(tail_scenarios[:, :k])
    for variant, indices in (
        ("crn_identity", identity),
        ("independent_permutation", permuted),
    ):
        background = np.stack(
            [np.asarray(raw_scenarios[issue, indices[issue]]) for issue in range(issues)]
        )
        energy = joint_correction_energy(tail100, background, stations)
        starts, stops = strongest_windows(energy, durations)
        masks = build_tapered_masks(
            starts, stops, leads, int(args.edge_taper_hours)
        )
        arrays: dict[str, np.ndarray] = {}
        for name in evaluation_member_arrays:
            raw_values = _load(args.raw_results, name)
            tail_values = _load(args.tail_results, name)
            paired_background = np.stack(
                [np.asarray(raw_values[issue, indices[issue]]) for issue in range(issues)]
            )
            localized = localize_members(
                paired_background, np.asarray(tail_values[:, :k]), masks
            )
            arrays[name] = np.concatenate(
                (np.asarray(raw_values[:, :b]), localized), axis=1
            )
        run = {
            **common,
            "condition_variant": f"original_s5_{variant}",
            "coupling": (
                "identity_full_path_crn_raw401_500_to_tail1_100"
                if variant == "crn_identity"
                else "issue_wise_fixed_random_permutation_of_same_raw_background_pool"
            ),
            "same_raw_and_tail_marginal_members_across_coupling_arms": True,
        }
        output = args.output_root / variant
        evaluate_and_save(
            output, arrays, static, stations, adjacency, run,
            args.energy_score_member_limit,
        )
        np.save(output / "localized_tail_mask.npy", masks)
        np.save(output / "localized_tail_start.npy", starts)
        np.save(output / "localized_tail_stop.npy", stops)
        np.save(output / "localized_tail_duration.npy", durations)
        np.save(output / "paired_raw_background_index.npy", indices)
        np.save(output / "pair_correction_energy.npy", energy.astype(np.float32))
        route = np.zeros((issues, b + k), dtype=np.float32)
        route[:, b:] = 1.0
        np.save(output / "tail_expert_route.npy", route)
        np.save(
            output / "tail_expert_probability.npy",
            np.full(issues, k / (b + k), dtype=np.float32),
        )
        print(f"S5_CRN_COUPLING_COMPLETE variant={variant} output={output}")

    (args.output_root / "coupling_manifest.json").write_text(
        json.dumps(common, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"S5_CRN_MATERIALIZATION_COMPLETE root={args.output_root}")


if __name__ == "__main__":
    main()
