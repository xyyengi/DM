#!/usr/bin/env python3
"""Offline low/high-frequency audit for the frozen Raw body-tail generator."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_station24_sustained_drop_tail_sweep import (
    Event,
    contiguous_runs,
    event_replay_specification,
    extract_independent_events,
    resample_shape,
    wind_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--widths", default="12,24")
    parser.add_argument("--top-events", type=int, default=5)
    parser.add_argument("--event-context-hours", type=int, default=12)
    return parser.parse_args()


def centered_moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Zero-phase boxcar with reflection padding along the final axis."""

    values = np.asarray(values, dtype=np.float64)
    width = int(width)
    if width < 2 or width > values.shape[-1]:
        raise ValueError("moving-average width must be in [2, series length]")
    left = (width - 1) // 2
    right = width // 2
    padded = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(left, right)], mode="reflect")
    cumulative = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros(padded.shape[:-1] + (1,), dtype=np.float64), cumulative], axis=-1
    )
    output = (cumulative[..., width:] - cumulative[..., :-width]) / float(width)
    if output.shape != values.shape:
        raise RuntimeError("centered moving average changed the series shape")
    return output


def decompose(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    low = centered_moving_average(values, width)
    return low, np.asarray(values, dtype=np.float64) - low


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size < 2:
        return np.nan
    left_flat = np.std(left) < 1e-9
    right_flat = np.std(right) < 1e-9
    if left_flat and right_flat:
        return 1.0 if np.allclose(left, right, atol=1e-8) else 0.0
    if left_flat or right_flat:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def actual_low_descriptor(low_residual: np.ndarray, event: Event) -> dict[str, float]:
    downside = np.maximum(-np.asarray(low_residual, dtype=np.float64), 0.0)
    peak = int(event.onset + np.argmax(downside[event.onset : event.stop]))
    depth = float(downside[peak])
    threshold = 0.5 * depth
    runs = contiguous_runs(downside >= threshold, merge_gap=1)
    containing = [run for run in runs if run[0] <= peak < run[1]]
    onset, stop = containing[0] if containing else (event.onset, event.stop)
    return {
        "onset": int(onset),
        "stop": int(stop),
        "duration": int(stop - onset),
        "minimum_mw": float(np.min(low_residual[onset:stop])),
        "depth_mw": depth,
        "threshold_mw": threshold,
    }


def member_low_descriptor(
    low_residual: np.ndarray,
    actual_low: np.ndarray,
    reference: dict[str, float],
    search_radius: int = 36,
) -> dict[str, float | bool]:
    downside = np.maximum(-np.asarray(low_residual, dtype=np.float64), 0.0)
    threshold = float(reference["threshold_mw"])
    runs = contiguous_runs(downside >= threshold, merge_gap=1)
    left = max(0, int(reference["onset"]) - int(search_radius))
    right = min(len(downside), int(reference["stop"]) + int(search_radius))
    runs = [run for run in runs if run[1] > left and run[0] < right]
    if not runs:
        return {
            "valid": False,
            "onset": np.nan,
            "duration": np.nan,
            "minimum_mw": np.nan,
            "depth_mw": np.nan,
            "shape_correlation": np.nan,
        }
    actual_interval = (int(reference["onset"]), int(reference["stop"]))

    def score(run: tuple[int, int]) -> tuple[float, float, float]:
        overlap = max(0, min(run[1], actual_interval[1]) - max(run[0], actual_interval[0]))
        return (
            -float(overlap),
            abs(float(run[0] - actual_interval[0])),
            abs(float((run[1] - run[0]) - (actual_interval[1] - actual_interval[0]))),
        )

    onset, stop = min(runs, key=score)
    actual_shape = resample_shape(actual_low[actual_interval[0] : actual_interval[1]])
    member_shape = resample_shape(low_residual[onset:stop])
    return {
        "valid": True,
        "onset": int(onset),
        "duration": int(stop - onset),
        "minimum_mw": float(np.min(low_residual[onset:stop])),
        "depth_mw": float(np.max(downside[onset:stop])),
        "shape_correlation": safe_correlation(actual_shape, member_shape),
    }


def descriptor_distribution(
    members: np.ndarray,
    actual_low: np.ndarray,
    reference: dict[str, float],
) -> dict[str, float]:
    descriptors = [member_low_descriptor(member, actual_low, reference) for member in members]
    frame = pd.DataFrame(descriptors)
    valid = frame[frame.valid]
    row: dict[str, float] = {
        "member_count": int(len(frame)),
        "valid_event_fraction": float(len(valid) / max(len(frame), 1)),
    }
    for metric in ("onset", "duration", "minimum_mw", "depth_mw"):
        actual_value = float(reference[metric])
        values = valid[metric].to_numpy(float)
        if len(values):
            q05, median, q95 = np.quantile(values, [0.05, 0.5, 0.95])
            row[f"{metric}_median"] = float(median)
            row[f"{metric}_q05"] = float(q05)
            row[f"{metric}_q95"] = float(q95)
            row[f"{metric}_median_abs_error"] = abs(float(median) - actual_value)
            row[f"{metric}_coverage_90"] = bool(q05 <= actual_value <= q95)
        else:
            for suffix in ("median", "q05", "q95", "median_abs_error"):
                row[f"{metric}_{suffix}"] = np.nan
            row[f"{metric}_coverage_90"] = False
    correlations = valid.shape_correlation.dropna().to_numpy(float)
    row["shape_correlation_median"] = (
        float(np.median(correlations)) if len(correlations) else np.nan
    )
    row["shape_correlation_positive_fraction"] = (
        float(np.mean(correlations >= 0.5)) if len(correlations) else 0.0
    )
    duration_scale = max(float(reference["duration"]), 6.0)
    depth_scale = max(float(reference["depth_mw"]), 1.0)
    correlation = row["shape_correlation_median"]
    correlation_penalty = 1.0 if not np.isfinite(correlation) else (1.0 - correlation) / 2.0
    components = [
        min(float(row["onset_median_abs_error"]) / 12.0, 2.0),
        min(float(row["duration_median_abs_error"]) / duration_scale, 2.0),
        min(float(row["depth_mw_median_abs_error"]) / depth_scale, 2.0),
        min(max(correlation_penalty, 0.0), 1.0),
    ]
    row["low_structure_score"] = float(np.mean(components))
    low_failures = [
        float(row["onset_median_abs_error"]) > 12.0,
        float(row["duration_median_abs_error"]) / duration_scale > 0.5,
        float(row["depth_mw_median_abs_error"]) / depth_scale > 0.3,
        not np.isfinite(correlation) or correlation < 0.5,
    ]
    row["low_failed_dimensions"] = int(sum(int(value) for value in low_failures))
    row["low_problem"] = bool(row["low_failed_dimensions"] >= 2)
    return row


def high_frequency_descriptors(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    output: dict[str, np.ndarray] = {}
    for lag in (1, 3, 6):
        output[f"ramp_{lag}h_mw"] = np.max(
            np.abs(values[:, lag:] - values[:, :-lag]), axis=-1
        )
    output["volatility_std_mw"] = np.std(values, axis=-1)
    return output


def high_distribution(
    members: np.ndarray,
    actual: np.ndarray,
    scale_floor_mw: float,
) -> dict[str, float]:
    actual_metrics = high_frequency_descriptors(actual)
    member_metrics = high_frequency_descriptors(members)
    row: dict[str, float] = {"member_count": int(len(members))}
    failures = 0
    relative_errors = []
    for metric, member_values in member_metrics.items():
        actual_value = float(actual_metrics[metric][0])
        q05, median, q95 = np.quantile(member_values, [0.05, 0.5, 0.95])
        relative_error = abs(float(median) - actual_value) / max(
            abs(actual_value), float(scale_floor_mw)
        )
        row[f"actual_{metric}"] = actual_value
        row[f"{metric}_median"] = float(median)
        row[f"{metric}_q05"] = float(q05)
        row[f"{metric}_q95"] = float(q95)
        row[f"{metric}_coverage_90"] = bool(q05 <= actual_value <= q95)
        row[f"{metric}_relative_median_error"] = float(relative_error)
        relative_errors.append(relative_error)
        failures += int(relative_error > 0.35)
    row["high_structure_score"] = float(np.mean(relative_errors))
    row["high_failed_dimensions"] = int(failures)
    row["high_problem"] = bool(failures >= 2)
    return row


def select_events(
    forecast_mw: np.ndarray,
    actual_mw: np.ndarray,
    forecast_norm: np.ndarray,
    actual_norm: np.ndarray,
    issues: pd.DataFrame,
    replay: dict[str, object],
    top_events: int,
) -> list[tuple[Event, str]]:
    independent = extract_independent_events(
        forecast_mw, actual_mw, forecast_norm, actual_norm, issues, replay
    )
    selected = [(event, "independent") for event in independent[:top_events]]
    used_issues = {event.issue for event, _ in selected}
    views = extract_independent_events(
        forecast_mw,
        actual_mw,
        forecast_norm,
        actual_norm,
        issues,
        replay,
        deduplicate=False,
        event_id_prefix="overlap_view",
    )
    for view in views:
        if len(selected) >= top_events:
            break
        if view.issue not in used_issues:
            selected.append((view, "overlap_view"))
            used_issues.add(view.issue)
    return selected


def plot_event_components(
    event: Event,
    scope: str,
    widths: list[int],
    actual_residual: np.ndarray,
    body_residual: np.ndarray,
    tail_residual: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(len(widths), 2, figsize=(17, 4.5 * len(widths)), sharex=True)
    axes = np.atleast_2d(axes)
    lead = np.arange(actual_residual.shape[-1])
    for row_index, width in enumerate(widths):
        actual_low, actual_high = decompose(actual_residual, width)
        body_low, body_high = decompose(body_residual, width)
        tail_low, tail_high = decompose(tail_residual, width)
        for column, title, actual_component, body_component, tail_component in (
            (0, "low frequency", actual_low, body_low, tail_low),
            (1, "high frequency", actual_high, body_high, tail_high),
        ):
            axis = axes[row_index, column]
            axis.plot(lead, actual_component, color="#111827", lw=1.8, label="actual residual")
            for values, color, name in (
                (body_component, "#2563eb", "body"),
                (tail_component, "#dc165d", "tail"),
            ):
                q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9], axis=0)
                axis.fill_between(lead, q10, q90, color=color, alpha=0.14)
                axis.plot(lead, median, color=color, lw=1.25, label=f"{name} median")
            axis.axvspan(event.onset, event.stop - 1, color="#f59e0b", alpha=0.15)
            axis.axhline(0.0, color="#6b7280", lw=0.7)
            axis.set_title(f"{width}h {title}")
            axis.set_ylabel("Aggregated wind residual MW")
            axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Lead hour")
    axes[-1, 1].set_xlabel("Lead hour")
    fig.suptitle(
        f"{event.event_id} ({scope}) | issue={event.issue_date}, raw onset={event.onset}, "
        f"duration={event.duration}h",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_diagnosis(low: pd.DataFrame, high: pd.DataFrame) -> tuple[str, dict[str, object]]:
    low_independent = low[low.analysis_scope.eq("independent")]
    high_independent = high[high.analysis_scope.eq("independent")]
    group_audit: dict[str, object] = {}
    for group in ("body", "tail"):
        low_group = low_independent[low_independent.member_group.eq(group)]
        high_group = high_independent[high_independent.member_group.eq(group)]
        group_audit[group] = {
            "low_problem_rate": float(low_group.low_problem.mean()),
            "high_problem_rate": float(high_group.high_problem.mean()),
            "median_low_structure_score": float(low_group.low_structure_score.median()),
            "median_high_structure_score": float(high_group.high_structure_score.median()),
            "median_low_shape_correlation": float(low_group.shape_correlation_median.median()),
        }
    low_problem_rate = float(low_independent.low_problem.mean())
    high_problem_rate = float(high_independent.high_problem.mean())
    low_problem = low_problem_rate >= 0.5
    high_problem = high_problem_rate >= 0.5
    if low_problem and high_problem:
        case = "C"
    elif low_problem:
        case = "A"
    elif high_problem:
        case = "B"
    else:
        low_score = float(low_independent.low_structure_score.median())
        high_score = float(high_independent.high_structure_score.median())
        case = "A" if low_score >= high_score else "B"
    return case, {
        "independent_low_problem_rate": low_problem_rate,
        "independent_high_problem_rate": high_problem_rate,
        "group_audit": group_audit,
        "classification_rule": (
            "A if >=50% independent low rows fail >=2/4 low criteria only; "
            "B if the analogous high rule only; C if both"
        ),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in frame.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float) and np.isfinite(value):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    case: str,
    audit: dict[str, object],
    low: pd.DataFrame,
    high: pd.DataFrame,
    events: list[tuple[Event, str]],
) -> None:
    low_summary = (
        low[low.analysis_scope.eq("independent")]
        .groupby(["filter_width_h", "member_group"], as_index=False)
        .agg(
            onset_mae_h=("onset_median_abs_error", "median"),
            duration_mae_h=("duration_median_abs_error", "median"),
            depth_mae_mw=("depth_mw_median_abs_error", "median"),
            shape_corr=("shape_correlation_median", "median"),
            low_score=("low_structure_score", "median"),
            low_problem_rate=("low_problem", "mean"),
        )
    )
    high_columns = [
        "ramp_1h_mw_relative_median_error",
        "ramp_3h_mw_relative_median_error",
        "ramp_6h_mw_relative_median_error",
        "volatility_std_mw_relative_median_error",
    ]
    high_summary = (
        high[high.analysis_scope.eq("independent")]
        .groupby(["filter_width_h", "member_group"], as_index=False)
        .agg({**{column: "median" for column in high_columns}, "high_structure_score": "median", "high_problem": "mean"})
        .rename(columns={"high_problem": "high_problem_rate"})
    )
    descriptions = {
        "A": "主要是low-frequency持续深跌结构没有学好。",
        "B": "主要是high-frequency爬坡和局部波动没有学好。",
        "C": "low-frequency持续结构与high-frequency局部变化都存在明显问题。",
    }
    lines = [
        "# Raw body-tail风电聚合residual低频—高频分解诊断",
        "",
        "## 结论",
        "",
        f"**情况{case}：{descriptions[case]}**",
        "",
        f"- 独立事件low问题率：{audit['independent_low_problem_rate']:.1%}",
        f"- 独立事件high问题率：{audit['independent_high_problem_rate']:.1%}",
        "- 本诊断不训练模型、不修改Raw场景，也不把真实未来或分解结果输入生成器。",
        "- 12h/24h居中反射滑动平均是零相位离线评价，避免人为制造onset滞后。",
        "",
        "## Low-frequency事件结构",
        "",
        markdown_table(low_summary),
        "",
        "## High-frequency局部变化",
        "",
        markdown_table(high_summary),
        "",
        "## 判定规则",
        "",
        "Low每行检查onset>12h、duration相对误差>50%、depth相对误差>30%、shape correlation<0.5；至少两项失败即记为low问题。",
        "High每行检查1/3/6h ramp和波动强度的成员中位相对误差是否>35%；至少两项失败即记为high问题。最终判定只使用3个独立事件，重叠窗口只作稳健性视图。",
        "",
        "## 事件图",
        "",
    ]
    for event, scope in events:
        lines.extend(
            [
                f"### {event.event_id} ({scope})",
                "",
                f"![{event.event_id}](figures/{event.event_id}_frequency_decomposition.png)",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    widths = [int(value) for value in args.widths.split(",") if value.strip()]
    if widths != sorted(set(widths)) or not widths:
        raise ValueError("widths must be unique ascending integers")
    result_dir = Path(args.result_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)

    metadata = json.loads((result_dir / "generation_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("checkpoint_state_source") != "raw":
        raise ValueError("frequency audit requires the Raw checkpoint result")
    if metadata.get("trained_condition_variant") != "geo_history_actual_body_tail_moe":
        raise ValueError("result is not the frozen historical-spatial Raw body-tail model")
    run_dir = Path(metadata["run_dir"])
    replay = event_replay_specification(run_dir)

    stations = pd.read_csv(Path(args.data_path) / "station_order.csv").sort_values("channel_index")
    wind = stations.data_type.eq("wind").to_numpy()
    capacities = stations.loc[wind, "capacity_mw"].to_numpy(float)
    total_wind_capacity = float(capacities.sum())
    issues = pd.read_csv(Path(args.data_path) / "val_issue_dates.csv")
    scenarios = np.load(result_dir / "actual_scenarios_normalized.npy", mmap_mode="r")
    actual = np.load(result_dir / "actual_data_normalized.npy", mmap_mode="r")
    forecast = np.load(result_dir / "forecast_data_normalized.npy", mmap_mode="r")
    route = np.load(result_dir / "tail_expert_route.npy").astype(bool)
    scenario_mw = wind_arrays(scenarios, wind, capacities, normalized=False)
    actual_mw = wind_arrays(actual, wind, capacities, normalized=False)
    forecast_mw = wind_arrays(forecast, wind, capacities, normalized=False)
    actual_norm = wind_arrays(actual, wind, capacities, normalized=True)
    forecast_norm = wind_arrays(forecast, wind, capacities, normalized=True)
    events = select_events(
        forecast_mw,
        actual_mw,
        forecast_norm,
        actual_norm,
        issues,
        replay,
        int(args.top_events),
    )
    actual_residual = actual_mw - forecast_mw
    scenario_residual = scenario_mw - forecast_mw[:, None, :]

    low_rows = []
    high_rows = []
    actual_component_rows = []
    for event, scope in events:
        issue = event.issue
        body = scenario_residual[issue, ~route[issue]]
        tail = scenario_residual[issue, route[issue]]
        for width in widths:
            actual_low, actual_high = decompose(actual_residual[issue], width)
            body_low, body_high = decompose(body, width)
            tail_low, tail_high = decompose(tail, width)
            reference = actual_low_descriptor(actual_low, event)
            actual_component_rows.append(
                {
                    "event_id": event.event_id,
                    "analysis_scope": scope,
                    "filter_width_h": width,
                    **reference,
                }
            )
            context_left = max(0, int(reference["onset"]) - int(args.event_context_hours))
            context_right = min(
                actual_high.shape[-1], int(reference["stop"]) + int(args.event_context_hours)
            )
            for group, low_values, high_values in (
                ("body", body_low, body_high),
                ("tail", tail_low, tail_high),
            ):
                low_rows.append(
                    {
                        "event_id": event.event_id,
                        "analysis_scope": scope,
                        "issue_index": issue,
                        "issue_date": event.issue_date,
                        "filter_width_h": width,
                        "member_group": group,
                        "actual_onset": reference["onset"],
                        "actual_duration": reference["duration"],
                        "actual_minimum_mw": reference["minimum_mw"],
                        "actual_depth_mw": reference["depth_mw"],
                        **descriptor_distribution(low_values, actual_low, reference),
                    }
                )
                high_rows.append(
                    {
                        "event_id": event.event_id,
                        "analysis_scope": scope,
                        "issue_index": issue,
                        "issue_date": event.issue_date,
                        "filter_width_h": width,
                        "member_group": group,
                        **high_distribution(
                            high_values[:, context_left:context_right],
                            actual_high[context_left:context_right],
                            scale_floor_mw=0.02 * total_wind_capacity,
                        ),
                    }
                )
        plot_event_components(
            event,
            scope,
            widths,
            actual_residual[issue],
            body,
            tail,
            figures / f"{event.event_id}_frequency_decomposition.png",
        )

    low_frame = pd.DataFrame(low_rows)
    high_frame = pd.DataFrame(high_rows)
    actual_frame = pd.DataFrame(actual_component_rows)
    case, audit = aggregate_diagnosis(low_frame, high_frame)
    low_frame.to_csv(output / "low_frequency_event_metrics.csv", index=False)
    high_frame.to_csv(output / "high_frequency_ramp_metrics.csv", index=False)
    actual_frame.to_csv(output / "actual_frequency_event_catalog.csv", index=False)
    write_report(
        output / "residual_frequency_decomposition_report.md",
        case,
        audit,
        low_frame,
        high_frame,
        events,
    )
    diagnostic = {
        "case": case,
        "decision": audit,
        "filter": "zero_phase_centered_reflect_boxcar",
        "filter_width_hours": widths,
        "generation_condition_used": False,
        "model_modified": False,
        "training_used": False,
        "test_used": False,
        "result_dir": str(result_dir),
        "event_count_independent": int(sum(scope == "independent" for _, scope in events)),
        "event_count_overlap_views": int(sum(scope == "overlap_view" for _, scope in events)),
        "events": [{**asdict(event), "analysis_scope": scope} for event, scope in events],
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"RESIDUAL_FREQUENCY_DECOMPOSITION_COMPLETE case={case} output={output}")


if __name__ == "__main__":
    main()
