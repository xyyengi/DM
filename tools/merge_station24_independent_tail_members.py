"""Build one fixed-budget body/tail ensemble from two frozen result folders.

The body and tail are sampled separately but the output contains exactly the
requested total number of members.  This is deliberately an evaluation/output
operation, not a hidden post-processing correction of either model.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-results", required=True)
    parser.add_argument("--tail-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--energy-score-member-limit", type=int, default=80)
    parser.add_argument("--body-member-limit", type=int, default=None)
    parser.add_argument("--tail-member-limit", type=int, default=None)
    parser.add_argument(
        "--condition-variant",
        default="independent_joint_tail_v1_mixture",
    )
    parser.add_argument(
        "--family",
        default="fixed_quota_independent_joint_tail_mixture",
    )
    return parser.parse_args()


def _load(path: Path, name: str) -> np.ndarray:
    item = path / name
    if not item.is_file():
        raise FileNotFoundError(f"missing required result array: {item}")
    return np.load(item, mmap_mode="r")


def _assert_identical(left: np.ndarray, right: np.ndarray, name: str) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(
            f"body/tail results do not share identical issuance data: {name}"
        )


def main() -> None:
    args = parse_args()
    body = Path(args.body_results)
    tail = Path(args.tail_results)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if body.resolve() == tail.resolve():
        raise ValueError("body and tail result directories must be distinct")

    metadata = [json.loads((folder / "generation_metadata.json").read_text(encoding="utf-8"))
                for folder in (body, tail)]
    for entry in metadata:
        if entry.get("future_actual_used_as_generation_condition", False) or entry.get("reportable_as_causal_forecast", True) is not True:
            raise ValueError("oracle/noncausal results cannot enter a causal mixture")
    for key in ("split", "generation_seed", "physical_projection", "architecture", "spatial_mode"):
        if key not in metadata[0] or metadata[0][key] != metadata[1].get(key):
            raise ValueError(f"body/tail protocol mismatch: {key}")
    if metadata[0]["split"] != "val":
        raise ValueError("independent tail experiment is validation-only")
    counts = None
    for name in MEMBER_ARRAYS:
        left, right = _load(body, name), _load(tail, name)
        if left.ndim != 4 or right.ndim != 4 or left.shape[0] != right.shape[0] or left.shape[2:] != right.shape[2:]:
            raise ValueError(f"invalid member shapes: {name}")
        current = (left.shape[1], right.shape[1])
        if counts is not None and current != counts:
            raise ValueError("inconsistent source member counts")
        counts = current
        for limit, count in zip((args.body_member_limit, args.tail_member_limit), counts):
            if limit is not None and not 0 < limit <= count:
                raise ValueError("member limit outside source ensemble")
    for name in STATIC_ARRAYS:
        _assert_identical(_load(body, name), _load(tail, name), name)
    for entry, count in zip(metadata, counts):
        if entry.get("n_samples") != count:
            raise ValueError("metadata and array member counts differ")
    output.mkdir(parents=True)
    body_members = None
    tail_members = None
    for name in MEMBER_ARRAYS:
        body_value = _load(body, name)
        tail_value = _load(tail, name)
        if body_value.ndim != 4 or tail_value.ndim != 4:
            raise ValueError(f"{name} must be [issue,member,lead,station]")
        if body_value.shape[0] != tail_value.shape[0] or body_value.shape[2:] != tail_value.shape[2:]:
            raise ValueError(f"body/tail member shape mismatch for {name}")
        for limit, value in ((args.body_member_limit, body_value), (args.tail_member_limit, tail_value)):
            if limit is not None and not 0 < limit <= value.shape[1]:
                raise ValueError("member limit outside source ensemble")
        body_value = body_value[:, :args.body_member_limit]
        tail_value = tail_value[:, :args.tail_member_limit]
        if body_members is not None and (body_members != body_value.shape[1] or tail_members != tail_value.shape[1]):
            raise ValueError("inconsistent member counts across arrays")
        np.save(output / name, np.concatenate((body_value, tail_value), axis=1))
        body_members = int(body_value.shape[1])
        tail_members = int(tail_value.shape[1])

    for name in STATIC_ARRAYS:
        body_value = _load(body, name)
        tail_value = _load(tail, name)
        _assert_identical(body_value, tail_value, name)
        np.save(output / name, body_value)

    if body_members is None or tail_members is None:
        raise RuntimeError("no member arrays were merged")
    total_members = body_members + tail_members
    if total_members <= 1:
        raise ValueError("merged ensemble needs at least two members")
    issue_count = int(_load(output, "actual_data_normalized.npy").shape[0])
    tail_route = np.zeros((issue_count, total_members), dtype=np.float32)
    tail_route[:, body_members:] = 1.0
    np.save(output / "tail_expert_route.npy", tail_route)
    np.save(output / "tail_expert_probability.npy", np.full(
        (issue_count,), tail_members / total_members, dtype=np.float32
    ))

    data_path = Path(args.data_path)
    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    adjacency = np.load(data_path / "station_adjacency.npy")
    samples = _load(output, "actual_scenarios_normalized.npy")
    raw_samples = _load(output, "actual_scenarios_raw_normalized.npy")
    actual = _load(output, "actual_data_normalized.npy")
    forecast = _load(output, "forecast_data_normalized.npy")
    daylight = _load(output, "station_daylight_mask.npy")
    summary, station_frame, lead_frame = evaluate_station_scenarios(
        samples, raw_samples, actual, forecast, stations, adjacency,
        daylight_mask=daylight,
        interval_levels=(0.80, 0.90, 0.95, 0.99),
        energy_score_member_limit=int(args.energy_score_member_limit),
    )
    summary["run"] = {
        **{key: metadata[0][key] for key in (
            "split", "generation_seed", "physical_projection", "architecture",
            "spatial_mode", "spatial_mix_levels", "parallel_spatial_fusion_levels",
            "parallel_spatial_adjacency_mode") if key in metadata[0]},
        **{key: metadata[1][key] for key in (
            "parameter_count", "checkpoint_validation_objective",
            "checkpoint_validation_mse", "checkpoint_validation_objective_type",
            "checkpoint_epoch") if key in metadata[1]},
        "condition_variant": str(args.condition_variant),
        "n_samples": total_members,
        "evaluation_member_count": total_members,
        "test_used": False,
        "source_runs": metadata,
        "future_actual_used_as_generation_condition": False,
        "reportable_as_causal_forecast": True,

        "family": str(args.family),
        "body_results": str(body),
        "tail_results": str(tail),
        "body_members": body_members,
        "tail_members": tail_members,
        "tail_fraction": tail_members / total_members,
        "total_members": total_members,
        "member_order": "body_then_independent_tail",
        "generation_condition": "unchanged_causal_raw_conditions",
        "event_labels_used_as_generation_condition": False,
    }
    save_evaluation(output, summary, station_frame, lead_frame)
    (output / "generation_metadata.json").write_text(
        json.dumps(summary["run"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "INDEPENDENT_TAIL_MERGE_COMPLETE "
        f"output={output} body={body_members} tail={tail_members} total={total_members}"
    )


if __name__ == "__main__":
    main()
