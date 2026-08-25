"""Audit parameter isolation and temporal specialization of a tail localizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--candidate-result", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-issues", type=int, default=5)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), -1, values
    )


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_checkpoint)
    run_dir = Path(args.candidate_run)
    result_dir = Path(args.candidate_result)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    candidate = torch.load(
        run_dir / "checkpoints" / "model_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    if source.get("condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("source checkpoint is not the Raw body-tail model")
    if candidate.get("condition_variant") != (
        "geo_history_actual_body_tail_time_localized"
    ):
        raise ValueError("candidate checkpoint variant mismatch")
    initialization = json.loads(
        (run_dir / "body_tail_initialization.json").read_text(encoding="utf-8")
    )
    if initialization.get("checkpoint_state_source") != "raw":
        raise ValueError("candidate was not initialized from Raw parameters")

    source_state = source["model_state_dict"]
    candidate_raw = candidate["model_state_dict"]
    candidate_ema = candidate["ema_model_state_dict"]
    time_keys = set(candidate.get("tail_time_trainable_parameter_names", []))
    if not time_keys:
        raise ValueError("candidate checkpoint has no tail-time trainable parameters")
    serialized_time_keys = time_keys | {
        name.replace("denoiser.", "diffusion.denoiser.", 1)
        for name in time_keys
    }
    changed_raw: list[str] = []
    changed_ema: list[str] = []
    checked = 0
    for name, tensor in source_state.items():
        if name in serialized_time_keys:
            continue
        if name not in candidate_raw or name not in candidate_ema:
            raise ValueError(f"candidate state lacks inherited key {name}")
        checked += 1
        if not torch.equal(tensor, candidate_raw[name]):
            changed_raw.append(name)
        if not torch.equal(tensor, candidate_ema[name]):
            changed_ema.append(name)
    if changed_raw or changed_ema:
        raise ValueError(
            "frozen inherited parameters changed: "
            f"raw={changed_raw[:5]} ema={changed_ema[:5]}"
        )

    scenarios = np.load(result_dir / "actual_scenarios_normalized.npy")
    actual = np.load(result_dir / "actual_data_normalized.npy")
    forecast = np.load(result_dir / "forecast_data_normalized.npy")
    route = np.load(result_dir / "tail_expert_route.npy").astype(bool)
    probability = np.load(result_dir / "tail_event_time_probability.npy")
    starts = np.load(result_dir / "tail_event_start.npy").astype(int)
    metadata = json.loads(
        (result_dir / "generation_metadata.json").read_text(encoding="utf-8")
    )
    issues, members, hours, station_count = scenarios.shape
    if actual.shape != (issues, hours, station_count):
        raise ValueError("actual array shape mismatch")
    if route.shape != (issues, members) or starts.shape != route.shape:
        raise ValueError("tail route/start shape mismatch")
    if probability.shape != (issues, hours):
        raise ValueError("tail time probability shape mismatch")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("tail time probabilities do not sum to one")
    if np.any(starts[~route] != -1):
        raise ValueError("body members must use event start -1")
    if np.any((starts[route] < 0) | (starts[route] >= hours)):
        raise ValueError("tail event starts are outside the 168 h horizon")
    if metadata.get("test_used") is not False or metadata.get("split") != "val":
        raise ValueError("formal tail-time audit requires locked validation results")
    if metadata.get("condition_feature_audit", {}).get(
        "future_actual_used_as_condition"
    ) is not False:
        raise ValueError("future actual leaked into generation conditions")

    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values(
        "channel_index"
    )
    wind = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacity = stations.capacity_mw.to_numpy(float)[wind]
    aggregate_scenarios = np.einsum(
        "imts,s->imt", scenarios[..., wind], capacity
    )
    aggregate_actual = np.einsum("its,s->it", actual[..., wind], capacity)
    aggregate_forecast = np.einsum("its,s->it", forecast[..., wind], capacity)
    width = 6
    mismatch = rolling_mean(aggregate_forecast - aggregate_actual, width)
    event_start = mismatch.argmax(axis=1)
    event_score = mismatch[np.arange(issues), event_start]
    top = np.argsort(event_score)[::-1][: int(args.top_issues)]
    radius = int(metadata["tail_time_mask_radius_hours"])
    rows: list[dict[str, float | int]] = []
    all_tail_body_offsets: list[float] = []
    for issue in range(issues):
        if route[issue].any() and (~route[issue]).any():
            all_tail_body_offsets.append(
                float(
                    aggregate_scenarios[issue, route[issue]].mean()
                    - aggregate_scenarios[issue, ~route[issue]].mean()
                )
            )
    for rank, issue in enumerate(top, start=1):
        start = int(event_start[issue])
        stop = start + width
        center = 0.5 * (start + stop - 1)
        context_left = max(0, start - radius)
        context_right = min(hours, stop + radius)
        routed_starts = starts[issue, route[issue]]
        overlap = (routed_starts >= start - radius) & (
            routed_starts <= stop - 1 + radius
        )
        tail_mean = aggregate_scenarios[issue, route[issue]].mean(axis=0)
        body_mean = aggregate_scenarios[issue, ~route[issue]].mean(axis=0)
        delta = tail_mean - body_mean
        event_effect = float(np.mean(np.abs(delta[start:stop])))
        outside = np.ones(hours, dtype=bool)
        outside[context_left:context_right] = False
        outside_effect = float(np.mean(np.abs(delta[outside])))
        rows.append(
            {
                "rank": rank,
                "issue_index": int(issue),
                "event_start": start,
                "event_end": stop - 1,
                "forecast_miss_6h_mw": float(event_score[issue]),
                "time_probability_core_mass": float(
                    probability[issue, start:stop].sum()
                ),
                "time_probability_context_mass": float(
                    probability[issue, context_left:context_right].sum()
                ),
                "time_probability_argmax": int(probability[issue].argmax()),
                "argmax_abs_offset_h": float(
                    abs(int(probability[issue].argmax()) - center)
                ),
                "routed_tail_members": int(route[issue].sum()),
                "sampled_masks_overlapping_event": int(overlap.sum()),
                "sampled_mask_overlap_fraction": float(overlap.mean()),
                "event_tail_body_abs_effect_mw": event_effect,
                "outside_tail_body_abs_effect_mw": outside_effect,
                "localization_ratio": event_effect / max(outside_effect, 1e-9),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "top5_tail_time_localization.csv", index=False)
    entropy = -np.sum(probability * np.log(probability.clip(min=1e-12)), axis=1)
    summary = {
        "method": "frozen_raw_body_tail_with_member_level_time_sampling_v1",
        "inherited_parameter_tensors_checked": checked,
        "inherited_raw_parameter_changes": len(changed_raw),
        "inherited_ema_parameter_changes": len(changed_ema),
        "trainable_tail_time_parameter_names": sorted(time_keys),
        "issues": issues,
        "members": members,
        "tail_member_fraction": float(route.mean()),
        "time_distribution_entropy_mean": float(entropy.mean()),
        "time_distribution_effective_hours_mean": float(np.exp(entropy).mean()),
        "full_horizon_tail_minus_body_mean_mw": float(
            np.mean(all_tail_body_offsets)
        ),
        "top5_localization_ratio_mean": float(frame.localization_ratio.mean()),
        "top5_events_with_sampled_overlap": int(
            (frame.sampled_masks_overlapping_event > 0).sum()
        ),
        "future_actual_used_as_condition": False,
        "test_used": False,
    }
    (output_dir / "tail_time_localization_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Tail time localization audit",
        "",
        f"- inherited tensors checked: {checked}",
        "- inherited Raw/EMA parameter changes: 0 / 0",
        f"- routed tail-member fraction: {summary['tail_member_fraction']:.4%}",
        f"- effective location hours: {summary['time_distribution_effective_hours_mean']:.2f}",
        f"- full-horizon tail-minus-body mean: {summary['full_horizon_tail_minus_body_mean_mw']:.2f} MW",
        f"- Top5 mean localization ratio: {summary['top5_localization_ratio_mean']:.3f}",
        f"- Top5 events with at least one overlapping sampled mask: {summary['top5_events_with_sampled_overlap']}/5",
        "- validation only; future actual is not a generation condition",
    ]
    (output_dir / "tail_time_localization_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"TAIL_TIME_LOCALIZATION_AUDIT_COMPLETE output={output_dir}")


if __name__ == "__main__":
    main()
