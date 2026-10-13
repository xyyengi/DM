#!/usr/bin/env python3
"""Materialize minimal S5 random-pairing variants for event sensitivity audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.localize_station24_independent_joint_tail import (
    _load, build_tapered_masks, joint_correction_energy, localize_members,
    stratified_train_durations, strongest_windows,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body-results", required=True)
    p.add_argument("--tail-results", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--data-path", default="diffusion_input_station")
    p.add_argument("--pairing-seeds", default="1729,271828,314159")
    p.add_argument("--body-members", type=int, default=400)
    p.add_argument("--tail-members", type=int, default=100)
    p.add_argument("--generation-seed", type=int, default=424242)
    p.add_argument("--edge-taper-hours", type=int, default=3)
    args = p.parse_args()
    body_dir, tail_dir, root = Path(args.body_results), Path(args.tail_results), Path(args.output_root)
    body = _load(body_dir, "actual_scenarios_normalized.npy")
    tail = _load(tail_dir, "actual_scenarios_normalized.npy")
    b, k = args.body_members, args.tail_members
    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    durations, _ = stratified_train_durations(args.data_path, k, body.shape[0], args.generation_seed)
    for pairing_seed in [int(x) for x in args.pairing_seeds.split(",")]:
        output = root / f"random_pairing_{pairing_seed}"
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        rng = np.random.default_rng(pairing_seed)
        indices = np.stack([b + rng.permutation(k) for _ in range(body.shape[0])])
        background = np.stack([np.asarray(body[i, indices[i]]) for i in range(body.shape[0])])
        tail100 = np.asarray(tail[:, :k])
        energy = joint_correction_energy(tail100, background, stations)
        starts, stops = strongest_windows(energy, durations)
        masks = build_tapered_masks(starts, stops, body.shape[2], args.edge_taper_hours)
        localized = localize_members(background, tail100, masks)
        np.save(output / "actual_scenarios_normalized.npy", np.concatenate((np.asarray(body[:, :b]), localized), axis=1))
        for name in ("actual_data_normalized.npy", "forecast_data_normalized.npy"):
            np.save(output / name, np.asarray(_load(body_dir, name)))
        route = np.zeros((body.shape[0], b + k), np.float32); route[:, b:] = 1
        np.save(output / "tail_expert_route.npy", route)
        np.save(output / "localized_tail_mask.npy", masks)
        np.save(output / "localized_tail_start.npy", starts)
        np.save(output / "localized_tail_stop.npy", stops)
        meta = {
            "family": "s5_random_pairing_sensitivity_v1", "split": "val",
            "generation_seed": args.generation_seed, "pairing_seed": pairing_seed,
            "pairing": "issue-wise fixed-seed permutation of Raw background members 401-500",
            "future_actual_used_for_selection": False, "test_used": False,
        }
        (output / "generation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"S5_RANDOM_PAIRING_MATERIALIZED seed={pairing_seed} output={output}")


if __name__ == "__main__":
    main()
