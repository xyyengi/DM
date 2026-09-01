#!/usr/bin/env python3
"""Evaluate a frozen Raw body-tail routing-ratio sweep on sustained wind drops."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOURS = 168


@dataclass(frozen=True)
class Event:
    event_id: str
    issue: int
    issue_date: str
    onset: int
    stop: int
    peak_start: int
    physical_time: pd.Timestamp
    depth_mw: float
    mean_shortfall_mw: float
    severity_normalized: float

    @property
    def duration(self) -> int:
        return self.stop - self.onset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="label=validation_result_directory; provide baseline, tail15, tail20, tail30",
    )
    parser.add_argument("--top-events", type=int, default=5)
    parser.add_argument("--time-tolerance-hours", type=int, default=12)
    parser.add_argument("--shape-time-tolerance-hours", type=int, default=24)
    parser.add_argument("--depth-ratio", type=float, default=0.75)
    return parser.parse_args()


def parse_results(values: list[str]) -> dict[str, Path]:
    results = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"result must be label=path, got {value!r}")
        label, raw_path = value.split("=", 1)
        if label in results:
            raise ValueError(f"duplicate result label {label}")
        results[label] = Path(raw_path)
    required = {"baseline", "tail15", "tail20", "tail30"}
    if set(results) != required:
        raise ValueError(f"expected result labels {sorted(required)}, got {sorted(results)}")
    return results


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    cumulative = np.cumsum(values, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros(values.shape[:-1] + (1,), dtype=np.float64), cumulative], axis=-1
    )
    return (cumulative[..., width:] - cumulative[..., :-width]) / float(width)


def contiguous_runs(mask: np.ndarray, merge_gap: int = 1) -> list[tuple[int, int]]:
    points = np.flatnonzero(mask)
    if not len(points):
        return []
    groups = np.split(points, np.flatnonzero(np.diff(points) > merge_gap + 1) + 1)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def wind_arrays(
    values: np.ndarray, wind: np.ndarray, capacities: np.ndarray, normalized: bool
) -> np.ndarray:
    weights = capacities / capacities.sum() if normalized else capacities
    return np.einsum("...ts,s->...t", np.asarray(values)[..., wind], weights)


def event_replay_specification(run_dir: Path) -> dict[str, object]:
    replay = json.loads((run_dir / "event_replay.json").read_text(encoding="utf-8"))
    if replay.get("method") == "train_unified_wind_event_replay_v1":
        replay = replay["deep_replay"]
    if int(replay["event_window_hours"]) != 6:
        raise ValueError("Raw body-tail sustained-event audit expects the trained 6h event definition")
    return replay


def extract_independent_events(
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    forecast_normalized: np.ndarray,
    actual_normalized: np.ndarray,
    issues: pd.DataFrame,
    replay: dict[str, object],
    *,
    deduplicate: bool = True,
    event_id_prefix: str = "val_drop",
) -> list[Event]:
    window = int(replay["event_window_hours"])
    threshold = float(replay["severity_thresholds"][0])
    merge_gap = int(replay.get("merge_gap_hours", 24))
    mismatch_norm = forecast_normalized - actual_normalized
    mismatch_mw = forecast_mw - actual_mw
    score = rolling_mean(mismatch_norm, window)
    target_column = "target_start" if "target_start" in issues else "issue_date"
    target_starts = pd.to_datetime(issues[target_column])
    candidates = []
    for issue in range(len(score)):
        peak_start = int(np.argmax(score[issue]))
        severity = float(score[issue, peak_start])
        if severity < threshold:
            continue
        midpoint = min(peak_start + window // 2, HOURS - 1)
        base_mask = mismatch_norm[issue] >= 0.5 * threshold
        runs = contiguous_runs(base_mask, merge_gap=1)
        containing = [run for run in runs if run[0] <= midpoint < run[1]]
        if containing:
            onset, stop = containing[0]
        else:
            onset, stop = peak_start, peak_start + window
        physical_time = target_starts.iloc[issue] + pd.Timedelta(hours=peak_start)
        candidates.append(
            {
                "issue": issue,
                "peak_start": peak_start,
                "severity": severity,
                "physical_time": physical_time,
                "onset": onset,
                "stop": stop,
            }
        )
    candidates.sort(key=lambda row: row["physical_time"])
    if deduplicate:
        clusters: list[list[dict[str, object]]] = []
        for candidate in candidates:
            if (
                not clusters
                or candidate["physical_time"] - clusters[-1][-1]["physical_time"]
                > pd.Timedelta(hours=merge_gap)
            ):
                clusters.append([candidate])
            else:
                clusters[-1].append(candidate)
        selected = [
            max(cluster, key=lambda row: float(row["severity"])) for cluster in clusters
        ]
    else:
        selected = candidates
    selected.sort(key=lambda row: float(row["severity"]), reverse=True)
    events = []
    for rank, row in enumerate(selected, start=1):
        issue = int(row["issue"])
        onset, stop = int(row["onset"]), int(row["stop"])
        shortfall = mismatch_mw[issue, onset:stop]
        events.append(
            Event(
                event_id=f"{event_id_prefix}_{rank:02d}",
                issue=issue,
                issue_date=str(issues.iloc[issue].issue_date),
                onset=onset,
                stop=stop,
                peak_start=int(row["peak_start"]),
                physical_time=pd.Timestamp(row["physical_time"]),
                depth_mw=float(np.max(shortfall)),
                mean_shortfall_mw=float(np.mean(shortfall)),
                severity_normalized=float(row["severity"]),
            )
        )
    if not events:
        raise ValueError("no independent validation sustained drops exceed the train-only threshold")
    return events


def candidate_segments(
    forecast_norm: np.ndarray,
    scenario_norm: np.ndarray,
    forecast_mw: np.ndarray,
    scenario_mw: np.ndarray,
    threshold: float,
) -> list[dict[str, float]]:
    mismatch_norm = forecast_norm - scenario_norm
    mismatch_mw = forecast_mw - scenario_mw
    segments = []
    for onset, stop in contiguous_runs(mismatch_norm >= 0.5 * threshold, merge_gap=1):
        if stop - onset < 2:
            continue
        local = mismatch_mw[onset:stop]
        segments.append(
            {
                "onset": onset,
                "stop": stop,
                "duration": stop - onset,
                "depth_mw": float(np.max(local)),
                "mean_shortfall_mw": float(np.mean(local)),
            }
        )
    return segments


def interval_overlap(left: tuple[int, int], right: tuple[int, int]) -> tuple[float, float]:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    recall = intersection / max(left[1] - left[0], 1)
    union = max(left[1], right[1]) - min(left[0], right[0])
    return float(recall), float(intersection / max(union, 1))


def resample_shape(values: np.ndarray, bins: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.repeat(values, bins)
    return np.interp(np.linspace(0, len(values) - 1, bins), np.arange(len(values)), values)


def best_member_match(
    event: Event,
    forecast_norm: np.ndarray,
    scenario_norm: np.ndarray,
    forecast_mw: np.ndarray,
    scenario_mw: np.ndarray,
    actual_shortfall_mw: np.ndarray,
    threshold: float,
    time_tolerance: int,
    shape_time_tolerance: int,
    depth_ratio_required: float,
) -> dict[str, object]:
    segments = candidate_segments(
        forecast_norm, scenario_norm, forecast_mw, scenario_mw, threshold
    )
    if not segments:
        return {
            "has_candidate": False,
            "coverage_hit": False,
            "morphology_reasonable": False,
            "onset_abs_error_h": np.nan,
            "duration_abs_error_h": np.nan,
            "depth_abs_error_mw": np.nan,
            "depth_ratio": np.nan,
            "true_interval_recall": 0.0,
            "interval_iou": 0.0,
            "shape_correlation": np.nan,
            "target_hour_coverage": 0.0,
        }
    target_shape = actual_shortfall_mw[event.onset : event.stop]
    scored = []
    for segment in segments:
        onset_error = abs(int(segment["onset"]) - event.onset)
        duration_error = abs(int(segment["duration"]) - event.duration)
        depth_error = abs(float(segment["depth_mw"]) - event.depth_mw)
        recall, iou = interval_overlap(
            (event.onset, event.stop), (int(segment["onset"]), int(segment["stop"]))
        )
        candidate_shape = (
            forecast_mw[int(segment["onset"]) : int(segment["stop"])]
            - scenario_mw[int(segment["onset"]) : int(segment["stop"])]
        )
        left = resample_shape(target_shape)
        right = resample_shape(candidate_shape)
        correlation = (
            float(np.corrcoef(left, right)[0, 1])
            if np.std(left) > 1e-8 and np.std(right) > 1e-8
            else 0.0
        )
        score = (
            onset_error / max(time_tolerance, 1)
            + duration_error / max(event.duration, 1)
            + depth_error / max(event.depth_mw, 1.0)
            + (1.0 - iou)
        )
        scored.append((score, segment, onset_error, duration_error, depth_error, recall, iou, correlation))
    _, segment, onset_error, duration_error, depth_error, recall, iou, correlation = min(
        scored, key=lambda value: value[0]
    )
    depth_ratio = float(segment["depth_mw"]) / max(event.depth_mw, 1.0)
    candidate_shortfall_at_target = forecast_mw[event.onset : event.stop] - scenario_mw[
        event.onset : event.stop
    ]
    target_hour_coverage = float(
        np.mean(candidate_shortfall_at_target >= depth_ratio_required * target_shape)
    )
    coverage_hit = (
        onset_error <= time_tolerance
        and recall >= 0.5
        and depth_ratio >= depth_ratio_required
    )
    morphology_reasonable = (
        onset_error <= shape_time_tolerance
        and 0.5 <= float(segment["duration"]) / max(event.duration, 1) <= 2.0
        and 0.5 <= depth_ratio <= 1.5
        and correlation >= 0.5
    )
    return {
        "has_candidate": True,
        "coverage_hit": bool(coverage_hit),
        "morphology_reasonable": bool(morphology_reasonable),
        "onset_abs_error_h": float(onset_error),
        "duration_abs_error_h": float(duration_error),
        "depth_abs_error_mw": float(depth_error),
        "depth_ratio": depth_ratio,
        "true_interval_recall": recall,
        "interval_iou": iou,
        "shape_correlation": correlation,
        "target_hour_coverage": target_hour_coverage,
    }


def evaluate_result_events(
    label: str,
    result_dir: Path,
    events: list[Event],
    wind: np.ndarray,
    capacities: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = np.load(result_dir / "actual_scenarios_normalized.npy", mmap_mode="r")
    actual = np.load(result_dir / "actual_data_normalized.npy", mmap_mode="r")
    forecast = np.load(result_dir / "forecast_data_normalized.npy", mmap_mode="r")
    route = np.load(result_dir / "tail_expert_route.npy").astype(bool)
    scenario_mw = wind_arrays(scenarios, wind, capacities, normalized=False)
    scenario_norm = wind_arrays(scenarios, wind, capacities, normalized=True)
    actual_mw = wind_arrays(actual, wind, capacities, normalized=False)
    forecast_mw = wind_arrays(forecast, wind, capacities, normalized=False)
    forecast_norm = wind_arrays(forecast, wind, capacities, normalized=True)
    actual_shortfall = forecast_mw - actual_mw
    member_rows = []
    summary_rows = []
    for event in events:
        issue = event.issue
        group_indices = {
            "all": np.arange(scenarios.shape[1]),
            "body": np.flatnonzero(~route[issue]),
            "tail": np.flatnonzero(route[issue]),
        }
        event_member_rows = []
        for member in range(scenarios.shape[1]):
            match = best_member_match(
                event,
                forecast_norm[issue],
                scenario_norm[issue, member],
                forecast_mw[issue],
                scenario_mw[issue, member],
                actual_shortfall[issue],
                threshold,
                args.time_tolerance_hours,
                args.shape_time_tolerance_hours,
                args.depth_ratio,
            )
            row = {
                "variant": label,
                "event_id": event.event_id,
                "issue_index": issue,
                "issue_date": event.issue_date,
                "member_index": member,
                "member_group": "tail" if route[issue, member] else "body",
                **match,
            }
            member_rows.append(row)
            event_member_rows.append(row)
        member_frame = pd.DataFrame(event_member_rows)
        for group, indices in group_indices.items():
            selected = member_frame[member_frame.member_index.isin(indices)]
            hit = selected[selected.coverage_hit]
            reasonable = selected[selected.morphology_reasonable]
            candidates = selected[selected.has_candidate]
            best = (
                candidates.sort_values(
                    ["onset_abs_error_h", "duration_abs_error_h", "depth_abs_error_mw"]
                ).iloc[0]
                if len(candidates)
                else None
            )
            summary_rows.append(
                {
                    "variant": label,
                    "event_id": event.event_id,
                    "issue_index": issue,
                    "issue_date": event.issue_date,
                    "event_onset": event.onset,
                    "event_duration_h": event.duration,
                    "event_depth_mw": event.depth_mw,
                    "event_severity_normalized": event.severity_normalized,
                    "member_group": group,
                    "member_count": int(len(selected)),
                    "any_coverage_hit": bool(len(hit)),
                    "coverage_hit_count": int(len(hit)),
                    "coverage_hit_rate": float(len(hit) / max(len(selected), 1)),
                    "reasonable_shape_count": int(len(reasonable)),
                    "reasonable_shape_rate": float(len(reasonable) / max(len(selected), 1)),
                    "best_onset_abs_error_h": float(best.onset_abs_error_h) if best is not None else np.nan,
                    "best_duration_abs_error_h": float(best.duration_abs_error_h) if best is not None else np.nan,
                    "best_depth_abs_error_mw": float(best.depth_abs_error_mw) if best is not None else np.nan,
                    "best_true_interval_recall": float(best.true_interval_recall) if best is not None else 0.0,
                    "best_shape_correlation": float(best.shape_correlation) if best is not None else np.nan,
                    "best_target_hour_coverage": float(best.target_hour_coverage) if best is not None else 0.0,
                    "median_candidate_onset_error_h": float(candidates.onset_abs_error_h.median()) if len(candidates) else np.nan,
                    "median_candidate_duration_error_h": float(candidates.duration_abs_error_h.median()) if len(candidates) else np.nan,
                    "median_candidate_depth_error_mw": float(candidates.depth_abs_error_mw.median()) if len(candidates) else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(member_rows)


def quality_row(label: str, result_dir: Path) -> dict[str, object]:
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (result_dir / "generation_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "variant": label,
        "tail_route_probability_override": metadata.get("tail_route_probability_override"),
        "realized_tail_fraction": float(metadata["tail_member_fraction"]),
        "wind_crps": float(metrics["station_average"]["wind"]["crps"]),
        "aggregate_wind_crps_mw": float(metrics["aggregate_mw"]["wind"]["crps"]),
        "wind_coverage_90": float(metrics["station_average"]["wind"]["coverage_90"]),
        "wind_width_90": float(metrics["station_average"]["wind"]["width_90"]),
        "aggregate_wind_coverage_90": float(metrics["aggregate_mw"]["wind"]["coverage_90"]),
        "aggregate_wind_width_90_mw": float(metrics["aggregate_mw"]["wind"]["width_90"]),
        "energy_score_pu": float(metrics["joint"]["energy_score_pu"]),
        "energy_score_member_count": int(metrics["joint"]["energy_score_member_count"]),
        "spatial_corr_rmse_all_pairs": float(metrics["joint"]["spatial_corr_rmse_all_pairs"]),
        "spatial_corr_rmse_wind_wind": float(metrics["joint"]["spatial_corr_rmse_wind_wind"]),
    }


def validate_protocol(results: dict[str, Path]) -> None:
    signatures = set()
    reference_actual = None
    reference_forecast = None
    expected = {"tail15": 0.15, "tail20": 0.20, "tail30": 0.30}
    for label, path in results.items():
        metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
        signatures.add(
            (
                metadata.get("split"), int(metadata.get("n_samples", 0)),
                int(metadata.get("generation_seed", -1)), metadata.get("checkpoint"),
                metadata.get("checkpoint_state_source"),
            )
        )
        if metadata.get("split") != "val" or int(metadata.get("n_samples", 0)) != 500:
            raise ValueError(f"{label} is not a 23-window val n500 result")
        if metadata.get("checkpoint_state_source") != "raw":
            raise ValueError(f"{label} did not use the Raw checkpoint state")
        override = metadata.get("tail_route_probability_override")
        if label == "baseline" and override is not None:
            raise ValueError("baseline must retain the fitted causal route probability")
        if label in expected and abs(float(override) - expected[label]) > 1e-9:
            raise ValueError(f"{label} route override mismatch: {override}")
        actual = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
        forecast = np.load(path / "forecast_data_normalized.npy", mmap_mode="r")
        if actual.shape[0] != 23 or forecast.shape[0] != 23:
            raise ValueError(f"{label} does not contain the required 23 validation windows")
        if reference_actual is None:
            reference_actual = np.asarray(actual)
            reference_forecast = np.asarray(forecast)
        elif not np.array_equal(reference_actual, actual) or not np.array_equal(reference_forecast, forecast):
            raise ValueError("sweep variants do not share identical actual/forecast arrays")
    if len(signatures) != 1:
        raise ValueError(f"sweep protocol mismatch: {signatures}")


def summarize_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group_name), group in frame.groupby(["variant", "member_group"], sort=False):
        rows.append(
            {
                "variant": variant,
                "member_group": group_name,
                "event_count": int(len(group)),
                "events_with_any_hit": int(group.any_coverage_hit.sum()),
                "event_any_hit_rate": float(group.any_coverage_hit.mean()),
                "total_hit_members": int(group.coverage_hit_count.sum()),
                "mean_member_hit_rate": float(group.coverage_hit_rate.mean()),
                "events_with_reasonable_shape": int((group.reasonable_shape_count > 0).sum()),
                "mean_reasonable_shape_rate": float(group.reasonable_shape_rate.mean()),
                "median_best_onset_error_h": float(group.best_onset_abs_error_h.median()),
                "median_best_duration_error_h": float(group.best_duration_abs_error_h.median()),
                "median_best_depth_error_mw": float(group.best_depth_abs_error_mw.median()),
                "median_best_interval_recall": float(group.best_true_interval_recall.median()),
                "median_candidate_onset_error_h": float(group.median_candidate_onset_error_h.median()),
                "median_candidate_duration_error_h": float(group.median_candidate_duration_error_h.median()),
                "median_candidate_depth_error_mw": float(group.median_candidate_depth_error_mw.median()),
            }
        )
    return pd.DataFrame(rows)


def classify_case(quality: pd.DataFrame, event_summary: pd.DataFrame) -> tuple[str, dict[str, object]]:
    baseline_quality = quality.set_index("variant").loc["baseline"]
    candidates = event_summary[event_summary.member_group.eq("all")].set_index("variant")
    baseline_event = candidates.loc["baseline"]
    best_label = str(candidates.mean_member_hit_rate.idxmax())
    best_event = candidates.loc[best_label]
    best_quality = quality.set_index("variant").loc[best_label]
    hit_gain = (
        float(best_event.mean_member_hit_rate) - float(baseline_event.mean_member_hit_rate)
    ) / max(float(baseline_event.mean_member_hit_rate), 1e-12)
    quality_relative = {
        metric: (float(best_quality[metric]) - float(baseline_quality[metric]))
        / max(abs(float(baseline_quality[metric])), 1e-12)
        for metric in (
            "wind_crps", "aggregate_wind_crps_mw", "wind_width_90",
            "aggregate_wind_width_90_mw", "energy_score_pu",
            "spatial_corr_rmse_all_pairs",
        )
    }
    ordinary_stable = (
        quality_relative["wind_crps"] <= 0.03
        and quality_relative["aggregate_wind_crps_mw"] <= 0.03
        and quality_relative["energy_score_pu"] <= 0.03
        and quality_relative["aggregate_wind_width_90_mw"] <= 0.10
        and quality_relative["spatial_corr_rmse_all_pairs"] <= 0.10
    )
    tail = event_summary[
        event_summary.variant.eq(best_label) & event_summary.member_group.eq("tail")
    ].iloc[0]
    morphology_unreasonable = bool(
        float(tail.median_candidate_onset_error_h) > 12
        or float(tail.median_candidate_duration_error_h) > 6
        or float(tail.events_with_reasonable_shape) < 0.5 * float(tail.event_count)
    )
    tail_can_hit = int(tail.events_with_any_hit) > 0
    if tail_can_hit and morphology_unreasonable:
        case = "C"
    elif hit_gain >= 0.25 and ordinary_stable:
        case = "A"
    else:
        case = "B"
    return case, {
        "best_ratio_variant": best_label,
        "all_member_hit_rate_relative_gain": hit_gain,
        "ordinary_quality_stable": ordinary_stable,
        "tail_can_generate_coverage_hits": tail_can_hit,
        "tail_morphology_unreasonable": morphology_unreasonable,
        "quality_relative_changes": quality_relative,
    }


def plot_events(
    events: list[Event],
    results: dict[str, Path],
    wind: np.ndarray,
    capacities: np.ndarray,
    output: Path,
    top_events: int,
) -> None:
    order = ["baseline", "tail15", "tail20", "tail30"]
    for event in events[:top_events]:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        for axis, label in zip(axes, order):
            path = results[label]
            scenarios = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
            actual = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
            forecast = np.load(path / "forecast_data_normalized.npy", mmap_mode="r")
            route = np.load(path / "tail_expert_route.npy").astype(bool)
            issue = event.issue
            scenario_mw = wind_arrays(scenarios[issue], wind, capacities, normalized=False)
            actual_mw = wind_arrays(actual[issue : issue + 1], wind, capacities, normalized=False)[0]
            forecast_mw = wind_arrays(forecast[issue : issue + 1], wind, capacities, normalized=False)[0]
            lead = np.arange(HOURS)
            body = scenario_mw[~route[issue]]
            tail = scenario_mw[route[issue]]
            axis.plot(lead, actual_mw, color="#111827", lw=1.8, label="actual")
            axis.plot(lead, forecast_mw, color="#059669", lw=1.5, ls="--", label="forecast")
            if len(body):
                lower, upper = np.quantile(body, [0.05, 0.95], axis=0)
                axis.fill_between(lead, lower, upper, color="#60a5fa", alpha=0.22, label="body 90%")
                axis.plot(lead, np.quantile(body, 0.05, axis=0), color="#2563eb", lw=1.0, label="body low-tail")
            if len(tail):
                lower, upper = np.quantile(tail, [0.05, 0.95], axis=0)
                axis.fill_between(lead, lower, upper, color="#fb7185", alpha=0.25, label="tail 90%")
                axis.plot(lead, np.quantile(tail, 0.05, axis=0), color="#dc165d", lw=1.2, label="tail low-tail")
            axis.axvspan(event.onset, event.stop - 1, color="#f59e0b", alpha=0.15)
            axis.set_ylabel("Wind MW")
            axis.set_title(f"{label}: body={len(body)}, tail={len(tail)}")
            axis.grid(alpha=0.2)
        axes[0].legend(ncol=6, fontsize=8, loc="upper center")
        axes[-1].set_xlabel("Lead hour")
        fig.suptitle(
            f"{event.event_id} | issue {event.issue_date} | onset={event.onset}, "
            f"duration={event.duration}h, depth={event.depth_mw:.0f} MW",
            y=0.995,
        )
        fig.tight_layout()
        fig.savefig(output / f"{event.event_id}_body_tail_sweep.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in frame.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            values.append(f"{value:.5g}" if isinstance(value, float) and np.isfinite(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    case: str,
    audit: dict[str, object],
    quality: pd.DataFrame,
    event_summary: pd.DataFrame,
    event_details: pd.DataFrame,
    figure_events: list[Event],
) -> None:
    descriptions = {
        "A": "增加tail成员后持续深跌命中明显增加，普通质量基本稳定：当前机制可生成，主要是尾部采样不足。",
        "B": "增加tail成员主要扩大区间，真实深跌命中未明显增加：不能依靠增加tail数量，需要改事件形态建模。",
        "C": "tail能够生成部分深跌，但onset/duration/depth分布不合理：下一步应做事件级时刻—持续时间—深度建模。",
    }
    quality_columns = [
        "variant", "realized_tail_fraction", "wind_crps", "aggregate_wind_crps_mw",
        "wind_coverage_90", "wind_width_90", "aggregate_wind_coverage_90",
        "aggregate_wind_width_90_mw", "energy_score_pu", "spatial_corr_rmse_all_pairs",
    ]
    event_columns = [
        "variant", "member_group", "event_count", "events_with_any_hit",
        "mean_member_hit_rate", "events_with_reasonable_shape",
        "median_best_onset_error_h", "median_best_duration_error_h",
        "median_best_depth_error_mw", "median_candidate_onset_error_h",
    ]
    lines = [
        "# Raw body-tail持续深跌覆盖率sweep诊断",
        "",
        "## 结论",
        "",
        f"**情况{case}。** {descriptions[case]}",
        "",
        f"- 最佳命中比例方案：`{audit['best_ratio_variant']}`",
        f"- 普通质量稳定：`{audit['ordinary_quality_stable']}`",
        f"- tail存在覆盖命中：`{audit['tail_can_generate_coverage_hits']}`",
        f"- tail形态分布不合理：`{audit['tail_morphology_unreasonable']}`",
        "",
        "## 普通场景质量",
        "",
        markdown_table(quality[quality_columns]),
        "",
        "Energy Score沿用生成评价中的固定成员上限，成员数见quality_metrics.csv。",
        "",
        "## 持续深跌事件覆盖",
        "",
        markdown_table(event_summary[event_columns]),
        "",
        "严格覆盖命中要求：onset误差≤12h、真实区间覆盖≥50%、生成深度≥真实深度的75%。",
        "合理形态要求：onset误差≤24h、duration比例0.5–2、depth比例0.5–1.5且归一化形态相关系数≥0.5。",
        "",
        "## 代表性事件图",
        "",
    ]
    independent_count = int(event_details.event_id.nunique())
    if len(figure_events) > independent_count:
        lines.extend(
            [
                f"验证集按24小时去重后只有{independent_count}个独立持续深跌事件。",
                "为满足可视诊断需要，后续图片补充同一物理事件在其他发布窗口下的视图；补充视图不参与事件命中率统计。",
                "",
            ]
        )
    for event in figure_events:
        event_id = event.event_id
        lines.append(f"![{event_id}](figures/{event_id}_body_tail_sweep.png)")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = parse_results(args.result)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "figures").mkdir(parents=True)
    validate_protocol(results)

    data_path = Path(args.data_path)
    run_dir = Path(args.run_dir)
    stations = pd.read_csv(data_path / "station_order.csv").sort_values("channel_index").reset_index(drop=True)
    wind = stations.data_type.eq("wind").to_numpy()
    capacities = stations.loc[wind, "capacity_mw"].to_numpy(float)
    issues = pd.read_csv(data_path / "val_issue_dates.csv")
    baseline = results["baseline"]
    actual = np.load(baseline / "actual_data_normalized.npy")
    forecast = np.load(baseline / "forecast_data_normalized.npy")
    actual_mw = wind_arrays(actual, wind, capacities, normalized=False)
    forecast_mw = wind_arrays(forecast, wind, capacities, normalized=False)
    actual_norm = wind_arrays(actual, wind, capacities, normalized=True)
    forecast_norm = wind_arrays(forecast, wind, capacities, normalized=True)
    replay = event_replay_specification(run_dir)
    events = extract_independent_events(
        forecast_mw, actual_mw, forecast_norm, actual_norm, issues, replay
    )
    issue_views = extract_independent_events(
        forecast_mw,
        actual_mw,
        forecast_norm,
        actual_norm,
        issues,
        replay,
        deduplicate=False,
        event_id_prefix="overlap_view",
    )
    figure_events = list(events[: args.top_events])
    used_issues = {event.issue for event in figure_events}
    for view in issue_views:
        if len(figure_events) >= args.top_events:
            break
        if view.issue not in used_issues:
            figure_events.append(view)
            used_issues.add(view.issue)

    event_frames = []
    member_frames = []
    quality_rows = []
    threshold = float(replay["severity_thresholds"][0])
    for label in ("baseline", "tail15", "tail20", "tail30"):
        quality_rows.append(quality_row(label, results[label]))
        event_frame, member_frame = evaluate_result_events(
            label, results[label], events, wind, capacities, threshold, args
        )
        event_frames.append(event_frame)
        member_frames.append(member_frame)
    event_details = pd.concat(event_frames, ignore_index=True)
    member_details = pd.concat(member_frames, ignore_index=True)
    quality = pd.DataFrame(quality_rows)
    event_summary = summarize_events(event_details)
    case, audit = classify_case(quality, event_summary)

    event_catalog = pd.DataFrame(
        [
            {
                "event_id": event.event_id, "issue_index": event.issue,
                "issue_date": event.issue_date, "physical_time": event.physical_time,
                "onset_hour": event.onset, "duration_hours": event.duration,
                "depth_mw": event.depth_mw, "mean_shortfall_mw": event.mean_shortfall_mw,
                "severity_normalized": event.severity_normalized,
            }
            for event in events
        ]
    )
    event_catalog.to_csv(output / "sustained_drop_event_catalog.csv", index=False)
    event_details.to_csv(output / "per_event_group_metrics.csv", index=False)
    member_details.to_csv(output / "per_member_event_matches.csv", index=False)
    event_summary.to_csv(output / "event_coverage_summary.csv", index=False)
    quality.to_csv(output / "quality_metrics.csv", index=False)
    plot_events(
        figure_events, results, wind, capacities, output / "figures", len(figure_events)
    )
    write_report(
        output / "sustained_drop_tail_sweep_report.md",
        case, audit, quality, event_summary, event_details, figure_events,
    )
    metadata = {
        "case": case,
        "decision": audit,
        "event_count": len(events),
        "figure_count": len(figure_events),
        "supplemental_overlap_view_count": len(figure_events) - min(len(events), args.top_events),
        "event_definition": "independent train-q80 6h forecast-minus-actual drops, variable excursion duration",
        "event_threshold_normalized": threshold,
        "tail_scale_sweep_used": False,
        "tail_scale_sweep_reason": "Raw body-tail has no tail-specific sampling scale or temperature",
        "model_parameters_modified": False,
        "training_used": False,
        "test_used": False,
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SUSTAINED_DROP_TAIL_SWEEP_COMPLETE case={case} output={output}")


if __name__ == "__main__":
    main()
