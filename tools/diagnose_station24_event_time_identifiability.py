#!/usr/bin/env python3
"""Leakage-safe event-time identifiability audit for Station24 wind power.

The diagnostic does not train or modify the diffusion model.  It separates two
questions which are otherwise easy to conflate:

1. Candidate upper bound: does the train-only event bank contain a prototype
   that is both close to the validation event time and deep enough?
2. Conditional ranking: can information available at issuance time rank those
   candidates near the correct lead hour?

Validation actual power is loaded only after the train-only candidate bank and
all causal ranking weights have been built.  Test artifacts are never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from station_discrete_event_memory import (
    EVENT_TYPES,
    HOURS,
    build_discrete_event_arrays,
)


WIND_COUNT = 13
PRIMARY_DURATION = 6
EVENT_DURATIONS = (1, 3, 6, 12)
TOLERANCES = (3, 6, 12)
TOP_PEAK_COUNTS = (1, 3, 5, 10)
TOP_CANDIDATE_COUNTS = (1, 3, 5, 10, 20, 40, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=96)
    parser.add_argument("--event-memory-quantile", type=float, default=0.70)
    parser.add_argument("--event-label-quantile", type=float, default=0.90)
    parser.add_argument("--target-stride-hours", type=int, default=1)
    parser.add_argument("--severe-downside-fraction", type=float, default=0.25)
    parser.add_argument("--retrieval-exclusion-days", type=int, default=6)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(values, axis=-1, dtype=np.float64)
    cumsum = np.concatenate(
        [np.zeros(values.shape[:-1] + (1,), dtype=np.float64), cumsum], axis=-1
    )
    return (cumsum[..., width:] - cumsum[..., :-width]) / float(width)


def aggregate_wind(
    values: np.ndarray, wind_indices: np.ndarray, wind_weights: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "...ts,s->...t", np.asarray(values)[..., wind_indices], wind_weights
    )


def previous_indices(issues: pd.DataFrame) -> np.ndarray:
    days = pd.to_datetime(issues["issue_date"]).dt.normalize()
    lookup = {day: index for index, day in enumerate(days)}
    return np.asarray(
        [lookup.get(day - pd.Timedelta(days=1), -1) for day in days],
        dtype=np.int64,
    )


def build_causal_power_context(
    forecast_aggregate: np.ndarray,
    residual_aggregate: np.ndarray,
    issues: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Build forecast revision and elapsed recent error within one split."""

    count = len(forecast_aggregate)
    revision = np.full((count, 144), np.nan, dtype=np.float64)
    recent_error = np.full((count, 24), np.nan, dtype=np.float64)
    previous = previous_indices(issues)
    for issue, previous_issue in enumerate(previous):
        if previous_issue < 0:
            continue
        revision[issue] = (
            forecast_aggregate[issue, :144]
            - forecast_aggregate[previous_issue, 24:]
        )
        recent_error[issue] = residual_aggregate[previous_issue, :24]
    return {
        "revision": revision,
        "recent_error": recent_error,
        "previous_index": previous,
    }


def rank01(values: np.ndarray) -> np.ndarray:
    """Ascending fractional rank, preserving NaN for unavailable features."""

    values = np.asarray(values, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite], kind="mergesort")]
    if len(order) == 1:
        result[order] = 0.0
    else:
        result[order] = np.arange(len(order), dtype=np.float64) / (len(order) - 1)
    return result


def softmax_distance(distance: np.ndarray, temperature: float = 0.20) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    logits = -distance / max(float(temperature), 1e-6)
    logits -= np.nanmax(logits)
    weights = np.exp(np.nan_to_num(logits, nan=-50.0))
    total = float(weights.sum())
    return weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))


def build_causal_candidate_weights(
    memory,
    train_context: dict[str, np.ndarray],
    val_context: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Re-rank the fixed forecast-only pool with causal power information.

    This is deliberately a non-parametric diagnostic, not a claimed optimal
    predictor.  Per-query ranks prevent forecast, revision, and recent-error
    distances from being mixed on incompatible numerical scales.
    """

    query_count, candidate_count = memory.distance.shape
    forecast_weights = np.asarray(memory.prior_weight, dtype=np.float64)
    revision_weights = np.zeros_like(forecast_weights)
    all_power_weights = np.zeros_like(forecast_weights)

    train_revision = train_context["revision"]
    val_revision = val_context["revision"]
    train_recent = train_context["recent_error"]
    val_recent = val_context["recent_error"]

    for query in range(query_count):
        forecast_rank = rank01(memory.distance[query])
        revision_distance = np.full(candidate_count, np.nan)
        recent_distance = np.full(candidate_count, np.nan)
        for candidate in range(candidate_count):
            source = int(memory.train_index[query, candidate])
            source_start = int(memory.source_start[query, candidate])
            target_start = int(memory.target_start[query, candidate])
            duration = int(memory.duration[query, candidate])
            if (
                source >= 0
                and source_start + duration <= 144
                and target_start + duration <= 144
            ):
                source_patch = train_revision[
                    source, source_start : source_start + duration
                ]
                query_patch = val_revision[
                    query, target_start : target_start + duration
                ]
                if np.all(np.isfinite(source_patch)) and np.all(np.isfinite(query_patch)):
                    revision_distance[candidate] = float(
                        np.mean((source_patch - query_patch) ** 2)
                    )
            if source >= 0:
                source_recent = train_recent[source]
                query_recent = val_recent[query]
                if np.all(np.isfinite(source_recent)) and np.all(np.isfinite(query_recent)):
                    recent_distance[candidate] = float(
                        np.mean((source_recent - query_recent) ** 2)
                    )

        revision_rank = rank01(revision_distance)
        recent_rank = rank01(recent_distance)
        forecast_revision = np.stack([forecast_rank, revision_rank], axis=0)
        all_power = np.stack(
            [forecast_rank, revision_rank, recent_rank], axis=0
        )
        revision_score = np.nanmean(forecast_revision, axis=0)
        all_power_score = np.nanmean(all_power, axis=0)
        revision_weights[query] = softmax_distance(revision_score)
        all_power_weights[query] = softmax_distance(all_power_score)

    return {
        "forecast_similarity": forecast_weights,
        "forecast_revision": revision_weights,
        "all_power": all_power_weights,
    }


def train_event_thresholds(
    train_forecast: np.ndarray,
    train_actual: np.ndarray,
    train_valid: np.ndarray,
    quantile: float,
) -> dict[int, float]:
    error = train_forecast - train_actual
    thresholds: dict[int, float] = {}
    for duration in EVENT_DURATIONS:
        score = rolling_mean(error, duration)
        valid_windows = np.stack(
            [
                np.convolve(row.astype(np.int64), np.ones(duration, int), mode="valid")
                == duration
                for row in train_valid
            ],
            axis=0,
        )
        selected = score[valid_windows & np.isfinite(score)]
        if not len(selected):
            raise ValueError(f"no valid train event scores for duration={duration}")
        thresholds[duration] = float(np.quantile(selected, quantile))
    return thresholds


def build_lead_time_prior_weights(
    memory,
    train_forecast: np.ndarray,
    train_actual: np.ndarray,
    train_valid: np.ndarray,
    thresholds: dict[int, float],
) -> np.ndarray:
    """Train-only baseline containing lead-hour prevalence but no query features."""

    frequency: dict[int, np.ndarray] = {}
    error = train_forecast - train_actual
    for duration in EVENT_DURATIONS:
        score = rolling_mean(error, duration)
        valid_windows = np.stack(
            [
                np.convolve(row.astype(np.int64), np.ones(duration, int), mode="valid")
                == duration
                for row in train_valid
            ],
            axis=0,
        )
        event = valid_windows & np.isfinite(score) & (score >= thresholds[duration])
        raw = event.mean(axis=0)
        frequency[duration] = np.convolve(raw, np.ones(7) / 7.0, mode="same")

    weights = np.zeros_like(memory.prior_weight, dtype=np.float64)
    for query in range(len(weights)):
        for candidate in range(weights.shape[1]):
            duration = int(memory.duration[query, candidate])
            start = int(memory.target_start[query, candidate])
            weights[query, candidate] = frequency[duration][start] + 1e-6
        weights[query] /= weights[query].sum()
    return weights


def extract_validation_events(
    forecast: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
    issues: pd.DataFrame,
    thresholds: dict[int, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    error = forecast - actual
    for duration in EVENT_DURATIONS:
        scores = rolling_mean(error, duration)
        for issue in range(len(forecast)):
            complete = (
                np.convolve(valid[issue].astype(np.int64), np.ones(duration, int), mode="valid")
                == duration
            )
            candidates = np.flatnonzero(
                complete
                & np.isfinite(scores[issue])
                & (scores[issue] >= thresholds[duration])
            )
            order = candidates[np.argsort(scores[issue, candidates])[::-1]]
            selected: list[int] = []
            separation = max(duration, 3)
            for start in order:
                if all(abs(int(start) - kept) >= separation for kept in selected):
                    selected.append(int(start))
            for start in sorted(selected):
                stop = start + duration
                rows.append(
                    {
                        "issue_index": issue,
                        "issue_date": str(issues.iloc[issue]["issue_date"]),
                        "duration_hours": duration,
                        "lead_start": start,
                        "lead_end": stop - 1,
                        "lead_day": start // 24 + 1,
                        "severity_pu": float(scores[issue, start]),
                        "forecast_window_mean_pu": float(
                            forecast[issue, start:stop].mean()
                        ),
                        "actual_window_mean_pu": float(actual[issue, start:stop].mean()),
                    }
                )
    return pd.DataFrame(rows)


def select_peaks(hazard: np.ndarray, count: int, separation: int) -> list[int]:
    order = np.argsort(np.asarray(hazard))[::-1]
    selected: list[int] = []
    for start in order:
        if hazard[start] <= 0 and selected:
            break
        if all(abs(int(start) - kept) >= separation for kept in selected):
            selected.append(int(start))
        if len(selected) == count:
            break
    return selected


def candidate_event_records(
    events: pd.DataFrame,
    memory,
    method_weights: dict[str, np.ndarray],
    val_forecast: np.ndarray,
    val_actual: np.ndarray,
    wind_indices: np.ndarray,
    wind_weights: np.ndarray,
) -> pd.DataFrame:
    candidate_residual = np.einsum(
        "qksh,s->qkh",
        np.asarray(memory.residual)[:, :, wind_indices, :],
        wind_weights,
    )
    rows: list[dict[str, object]] = []
    downside_types = {
        EVENT_TYPES.index("sustained_drop"),
        EVENT_TYPES.index("down_ramp"),
    }
    for event_id, event in events.reset_index(drop=True).iterrows():
        issue = int(event.issue_index)
        duration = int(event.duration_hours)
        true_start = int(event.lead_start)
        true_stop = true_start + duration
        actual_mean = float(val_actual[issue, true_start:true_stop].mean())
        same_event_family = np.asarray(
            [
                int(value) in downside_types
                for value in memory.event_type[issue]
            ],
            dtype=bool,
        ) & (memory.duration[issue] == duration)
        indices = np.flatnonzero(same_event_family)
        starts = memory.target_start[issue, indices]
        scenario_mean = np.asarray(
            [
                (
                    val_forecast[issue, true_start:true_stop]
                    + candidate_residual[issue, candidate, true_start:true_stop]
                ).mean()
                for candidate in indices
            ],
            dtype=np.float64,
        )
        depth_hit = scenario_mean <= actual_mean

        for method, all_weights in method_weights.items():
            weights = np.asarray(all_weights[issue, indices], dtype=np.float64)
            weights = weights / weights.sum() if weights.sum() > 0 else np.full(
                len(indices), 1.0 / max(len(indices), 1)
            )
            ordered = np.argsort(weights)[::-1]
            hazard = np.zeros(HOURS, dtype=np.float64)
            for local_index, start in enumerate(starts):
                hazard[int(start)] += weights[local_index]
            row: dict[str, object] = {
                "event_id": int(event_id),
                "method": method,
                **event.to_dict(),
                "candidate_count": int(len(indices)),
                "oracle_depth_candidate_count": int(depth_hit.sum()),
            }
            for tolerance in TOLERANCES:
                time_hit = np.abs(starts - true_start) <= tolerance
                joint_hit = time_hit & depth_hit
                row[f"oracle_time_hit_{tolerance}h"] = bool(np.any(time_hit))
                row[f"oracle_joint_hit_{tolerance}h"] = bool(np.any(joint_hit))
                row[f"hazard_mass_{tolerance}h"] = float(
                    weights[time_hit].sum()
                )
                for peak_count in TOP_PEAK_COUNTS:
                    peaks = select_peaks(
                        hazard, peak_count, separation=max(3, duration // 2)
                    )
                    row[f"top{peak_count}_peak_hit_{tolerance}h"] = bool(
                        any(abs(peak - true_start) <= tolerance for peak in peaks)
                    )
            for candidate_count in TOP_CANDIDATE_COUNTS:
                chosen = ordered[: min(candidate_count, len(ordered))]
                for tolerance in TOLERANCES:
                    row[
                        f"top{candidate_count}_candidate_joint_hit_{tolerance}h"
                    ] = bool(
                        np.any(
                            depth_hit[chosen]
                            & (np.abs(starts[chosen] - true_start) <= tolerance)
                        )
                    )
            row["top5_predicted_starts"] = ";".join(
                str(value)
                for value in select_peaks(
                    hazard, 5, separation=max(3, duration // 2)
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, duration), group in records.groupby(
        ["method", "duration_hours"], sort=False
    ):
        row: dict[str, object] = {
            "method": method,
            "duration_hours": int(duration),
            "event_count": int(len(group)),
        }
        for tolerance in TOLERANCES:
            for name in (
                f"oracle_time_hit_{tolerance}h",
                f"oracle_joint_hit_{tolerance}h",
            ):
                row[f"{name}_rate"] = float(group[name].mean())
            row[f"mean_hazard_mass_{tolerance}h"] = float(
                group[f"hazard_mass_{tolerance}h"].mean()
            )
            for peak_count in TOP_PEAK_COUNTS:
                name = f"top{peak_count}_peak_hit_{tolerance}h"
                row[f"{name}_rate"] = float(group[name].mean())
            for candidate_count in TOP_CANDIDATE_COUNTS:
                name = f"top{candidate_count}_candidate_joint_hit_{tolerance}h"
                row[f"{name}_rate"] = float(group[name].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_lead_days(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, duration, lead_day), group in records.groupby(
        ["method", "duration_hours", "lead_day"], sort=False
    ):
        rows.append(
            {
                "method": method,
                "duration_hours": int(duration),
                "lead_day": int(lead_day),
                "event_count": int(len(group)),
                "oracle_time_hit_6h_rate": float(
                    group["oracle_time_hit_6h"].mean()
                ),
                "oracle_joint_hit_6h_rate": float(
                    group["oracle_joint_hit_6h"].mean()
                ),
                "top1_peak_hit_6h_rate": float(
                    group["top1_peak_hit_6h"].mean()
                ),
                "top5_peak_hit_6h_rate": float(
                    group["top5_peak_hit_6h"].mean()
                ),
                "mean_hazard_mass_6h": float(group["hazard_mass_6h"].mean()),
            }
        )
    return pd.DataFrame(rows)


def top5_details(records: pd.DataFrame) -> pd.DataFrame:
    primary = records[records.duration_hours == PRIMARY_DURATION]
    event_rows = (
        primary.drop_duplicates("event_id")
        .sort_values("severity_pu", ascending=False)
        .drop_duplicates("issue_index", keep="first")
        .head(5)
        [["event_id", "severity_pu"]]
    )
    return records.merge(event_rows, on=["event_id", "severity_pu"], how="inner")


def plot_top5_hazard(
    details: pd.DataFrame,
    memory,
    method_weights: dict[str, np.ndarray],
    output: Path,
) -> None:
    events = details.drop_duplicates("event_id").sort_values(
        "severity_pu", ascending=False
    )
    methods = list(method_weights)
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(14, 2.8 * len(methods)), sharex=True
    )
    axes = np.atleast_1d(axes)
    downside_types = {
        EVENT_TYPES.index("sustained_drop"),
        EVENT_TYPES.index("down_ramp"),
    }
    for axis, method in zip(axes, methods):
        image_rows = []
        true_starts = []
        for _, event in events.iterrows():
            issue = int(event.issue_index)
            duration = int(event.duration_hours)
            selected = np.asarray(
                [
                    int(value) in downside_types
                    for value in memory.event_type[issue]
                ],
                dtype=bool,
            ) & (memory.duration[issue] == duration)
            indices = np.flatnonzero(selected)
            weights = method_weights[method][issue, indices].astype(np.float64)
            weights /= weights.sum()
            hazard = np.zeros(HOURS, dtype=np.float64)
            for candidate, weight in zip(indices, weights):
                hazard[int(memory.target_start[issue, candidate])] += weight
            hazard /= max(float(hazard.max()), 1e-12)
            image_rows.append(hazard)
            true_starts.append(int(event.lead_start))
        axis.imshow(np.stack(image_rows), aspect="auto", vmin=0, vmax=1, cmap="magma")
        axis.scatter(true_starts, np.arange(len(true_starts)), marker="|", s=220, c="cyan")
        axis.set_ylabel(method)
        axis.set_yticks(np.arange(len(true_starts)))
        axis.set_yticklabels([f"event {index+1}" for index in range(len(true_starts))])
    axes[-1].set_xlabel("Lead hour; cyan marker = actual 6 h deep-drop start")
    fig.suptitle("Condition-only historical event-time hazard for Top-5 deep drops")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    details: pd.DataFrame,
    memory_audit: dict[str, object],
    thresholds: dict[int, float],
) -> None:
    six = summary[summary.duration_hours == PRIMARY_DURATION].set_index("method")
    forecast = six.loc["forecast_similarity"]
    all_power = six.loc["all_power"]
    oracle = float(forecast["oracle_joint_hit_6h_rate"])
    forecast_top5 = float(forecast["top5_peak_hit_6h_rate"])
    all_power_top5 = float(all_power["top5_peak_hit_6h_rate"])
    time_oracle = float(forecast["oracle_time_hit_6h_rate"])
    if time_oracle >= 0.75 and oracle < 0.50:
        diagnosis = (
            "训练历史候选通常覆盖了真实事件时刻附近，但可直接搬运的历史残差深度明显不足。"
            "当前瓶颈是“功率条件只能中等程度排序时刻”与“历史原型不能外推到足够深的尾部”同时存在，"
            "而不是简单增加选择器容量即可解决。"
        )
    elif oracle >= 0.80 and max(forecast_top5, all_power_top5) < 0.60:
        diagnosis = (
            "训练历史候选通常包含正确时刻和足够深度，但现有功率条件不能稳定把它排到前列。"
            "瓶颈主要在条件—事件匹配与不确定性表达，而不是历史库完全没有极端。"
        )
    elif oracle < 0.50:
        diagnosis = (
            "即使使用验证真实值做事后 Oracle，训练历史候选也经常缺少同时满足时刻和深度的原型。"
            "瓶颈首先是历史事件库覆盖不足，继续加强选择器不会解决。"
        )
    else:
        diagnosis = (
            "历史候选覆盖和条件排序都只达到中等水平；下一步必须同时改善事件库分层和多峰时刻分布，"
            "不能只扩大注意力网络。"
        )

    def pct(value: float) -> str:
        return f"{100.0 * float(value):.1f}%"

    lines = [
        "# 24场站功率条件—极端事件时刻可辨识上界诊断",
        "",
        "## 结论",
        "",
        diagnosis,
        "",
        "本诊断不训练扩散模型。候选检索只使用训练历史和发布时可获得的功率信息；"
        "验证真实功率仅在候选固定后用于评分；测试集未读取。",
        "",
        "## 6 h持续深跌核心结果",
        "",
        "| 条件排序 | 事件数 | 候选时刻Oracle±6h | 候选联合Oracle：时刻±6h且深度足够 | Top-5时刻峰命中±6h | ±6h平均概率质量 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "lead_time_prior": "仅训练提前时距先验",
        "forecast_similarity": "当前发布预测",
        "forecast_revision": "发布预测＋相邻版本修订",
        "all_power": "全部可用功率条件",
    }
    for method, label in labels.items():
        row = six.loc[method]
        lines.append(
            f"| {label} | {int(row.event_count)} | "
            f"{pct(row.oracle_time_hit_6h_rate)} | "
            f"{pct(row.oracle_joint_hit_6h_rate)} | "
            f"{pct(row.top5_peak_hit_6h_rate)} | "
            f"{pct(row.mean_hazard_mass_6h)} |"
        )
    lines.extend(
        [
            "",
            "候选Oracle只回答训练历史中是否存在答案，不代表生成时能够知道验证真实值。"
            "Top-5时刻峰才反映只用功率条件时，正确时段能否被排到前五。",
            "",
            "## 数据边界",
            "",
            f"- 训练事件库规模：{memory_audit['event_bank_count']}个去重事件原型；",
            f"- 每个验证窗口候选：{memory_audit['top_k_candidate_pool']}个；",
            f"- 训练标签阈值：各持续尺度forecast-actual误差的q90：{json.dumps(thresholds)}；",
            "- 允许条件：当前168 h发布功率预测、可对齐的上一版发布预测修订、上一发布日已经实现的24 h功率误差；",
            "- 禁止条件：验证未来真实功率、测试集、天气预报及任何未来外生变量。",
            "",
            "## 输出说明",
            "",
            "- `identifiability_summary.csv`：1/3/6/12 h总体上界与条件排序；",
            "- `lead_day_identifiability.csv`：按第1～7提前日拆分的时刻可辨识结果；",
            "- `validation_event_catalog.csv`：使用训练阈值识别的验证极端事件；",
            "- `event_candidate_records.csv`：每个事件的候选和时刻命中结果；",
            "- `top5_6h_details.csv`：最困难5个持续6 h深跌；",
            "- `figures/top5_event_time_hazard.png`：只用功率条件得到的Top-5事件时刻风险分布。",
        ]
    )
    (output / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()

    stations = (
        pd.read_csv(data_path / "station_order.csv")
        .sort_values("channel_index")
        .reset_index(drop=True)
    )
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    wind_capacity = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
    wind_weights = wind_capacity / wind_capacity.sum()

    train_forecast_full = np.load(data_path / "train_forecast.npy", mmap_mode="r")
    train_actual_full = np.load(data_path / "train_actual.npy", mmap_mode="r")
    train_residual_full = np.load(data_path / "train_residual.npy", mmap_mode="r")
    train_fill_full = np.load(data_path / "train_fill_mask.npy", mmap_mode="r")
    train_issues = pd.read_csv(data_path / "train_issue_dates.csv")
    val_forecast_full = np.load(data_path / "val_forecast.npy", mmap_mode="r")
    val_residual_full = np.load(data_path / "val_residual.npy", mmap_mode="r")
    val_fill_full = np.load(data_path / "val_fill_mask.npy", mmap_mode="r")
    val_issues = pd.read_csv(data_path / "val_issue_dates.csv")

    train_forecast = aggregate_wind(train_forecast_full, wind_indices, wind_weights)
    train_actual = aggregate_wind(train_actual_full, wind_indices, wind_weights)
    train_residual = aggregate_wind(train_residual_full, wind_indices, wind_weights)
    train_valid = np.all(train_fill_full[:, :, wind_indices] == 0, axis=2)
    val_forecast = aggregate_wind(val_forecast_full, wind_indices, wind_weights)
    val_residual = aggregate_wind(val_residual_full, wind_indices, wind_weights)
    val_valid = np.all(val_fill_full[:, :, wind_indices] == 0, axis=2)

    train_context = build_causal_power_context(
        train_forecast, train_residual, train_issues
    )
    val_context = build_causal_power_context(val_forecast, val_residual, val_issues)

    # Build every causal object before loading validation actual power.
    memory = build_discrete_event_arrays(
        data_dir=data_path,
        split="val",
        top_k=args.top_k,
        exclusion_days=args.retrieval_exclusion_days,
        event_quantile=args.event_memory_quantile,
        target_stride_hours=args.target_stride_hours,
        severe_downside_fraction=args.severe_downside_fraction,
        event_durations=EVENT_DURATIONS,
    )
    thresholds = train_event_thresholds(
        train_forecast,
        train_actual,
        train_valid,
        quantile=args.event_label_quantile,
    )
    method_weights = build_causal_candidate_weights(
        memory, train_context=train_context, val_context=val_context
    )
    method_weights = {
        "lead_time_prior": build_lead_time_prior_weights(
            memory, train_forecast, train_actual, train_valid, thresholds
        ),
        **method_weights,
    }

    val_actual_full = np.load(data_path / "val_actual.npy", mmap_mode="r")
    val_actual = aggregate_wind(val_actual_full, wind_indices, wind_weights)
    residual_identity_error = float(
        np.max(np.abs(val_residual - (val_actual - val_forecast)))
    )
    if residual_identity_error > 1e-6:
        raise ValueError(
            f"validation residual identity failed: {residual_identity_error}"
        )
    events = extract_validation_events(
        val_forecast,
        val_actual,
        val_valid,
        val_issues,
        thresholds,
    )
    if events.empty:
        raise ValueError("no validation events exceeded train-only thresholds")
    records = candidate_event_records(
        events,
        memory,
        method_weights,
        val_forecast,
        val_actual,
        wind_indices,
        wind_weights,
    )
    summary = summarize_records(records)
    lead_day_summary = summarize_lead_days(records)
    details = top5_details(records)

    events.to_csv(output / "validation_event_catalog.csv", index=False)
    records.to_csv(output / "event_candidate_records.csv", index=False)
    summary.to_csv(output / "identifiability_summary.csv", index=False)
    lead_day_summary.to_csv(output / "lead_day_identifiability.csv", index=False)
    details.to_csv(output / "top5_6h_details.csv", index=False)
    plot_top5_hazard(
        details,
        memory,
        method_weights,
        output / "figures" / "top5_event_time_hazard.png",
    )
    metadata = {
        "method": "station24_power_only_event_time_identifiability_v1",
        "train_only_event_memory": True,
        "validation_actual_used_for_retrieval": False,
        "validation_actual_used_for_posthoc_evaluation": True,
        "test_loaded": False,
        "available_condition_sources": [
            "current_issued_power_forecast",
            "aligned_previous_issue_power_forecast_revision",
            "elapsed_previous_issue_24h_power_error",
        ],
        "external_weather_used": False,
        "event_label_quantile": args.event_label_quantile,
        "event_thresholds": thresholds,
        "validation_event_count": int(len(events)),
        "validation_issue_count": int(len(val_forecast)),
        "residual_identity_max_error": residual_identity_error,
        "event_memory_audit": memory.audit,
        "causal_availability": {
            "train_previous_issue_count": int(
                np.sum(train_context["previous_index"] >= 0)
            ),
            "validation_previous_issue_count": int(
                np.sum(val_context["previous_index"] >= 0)
            ),
        },
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output, summary, details, memory.audit, thresholds)
    print(f"EVENT_TIME_IDENTIFIABILITY_COMPLETE output={output}")


if __name__ == "__main__":
    main()
