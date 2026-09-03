#!/usr/bin/env python3
"""Zero-training audit of forecast-only historical residual event priors.

The diagnostic fixes leakage-safe Top-K train neighbours using the current
issued forecast before validation actuals are loaded.  It then asks whether
events extracted from those train-only residual trajectories predict the
occurrence, onset, duration, depth, and ramp severity of validation events
better than equally sized random train histories.

No model, checkpoint, generated scenario, test target, or validation target is
used during retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from station_retrieval_memory import _forecast_features


HOURS = 168
RAMP_LAGS = (1, 3, 6)
EPS = 1e-12


@dataclass(frozen=True)
class Retrieval:
    indices: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    bank_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--random-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--episode-base-quantile", type=float, default=0.75)
    parser.add_argument("--onset-bandwidth-hours", type=float, default=3.0)
    parser.add_argument("--attribute-radius-hours", type=int, default=12)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="finish report/metadata from completed CSV outputs after a reporting-only failure",
    )
    return parser.parse_args()


def aggregate_wind(values: np.ndarray, wind_indices: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    return np.einsum(
        "nts,s->nt",
        np.asarray(values, dtype=np.float64)[:, :, wind_indices],
        capacities,
    )


def valid_wind(mask: np.ndarray, wind_indices: np.ndarray) -> np.ndarray:
    return ~np.any(np.asarray(mask)[:, :, wind_indices] != 0, axis=2)


def build_strict_forecast_retrieval(data_path: Path, top_k: int) -> Retrieval:
    """Retrieve by issued-forecast features only, excluding calendar columns."""

    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacities = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
    train_forecast = np.load(data_path / "train_forecast.npy", mmap_mode="r")
    val_forecast = np.load(data_path / "val_forecast.npy", mmap_mode="r")
    train_fill = np.load(data_path / "train_fill_mask.npy", mmap_mode="r")
    train_issues = pd.read_csv(data_path / "train_issue_dates.csv")
    val_issues = pd.read_csv(data_path / "val_issue_dates.csv")
    bank_indices = np.flatnonzero(
        ~np.any(np.asarray(train_fill[:, :, wind_indices]) != 0, axis=(1, 2))
    )
    if len(bank_indices) < top_k:
        raise ValueError(f"only {len(bank_indices)} complete train histories for Top-{top_k}")

    # _forecast_features ends with four calendar columns.  Removing them makes
    # the primary audit exactly forecast-tensor-only while retaining the same
    # forecast shape/ramp/daily descriptors as the deployed retrieval code.
    train_features = _forecast_features(
        train_forecast, train_issues, wind_indices, capacities
    )[:, :-4]
    val_features = _forecast_features(
        val_forecast, val_issues, wind_indices, capacities
    )[:, :-4]
    mean = train_features[bank_indices].mean(axis=0)
    std = train_features[bank_indices].std(axis=0)
    std[std < 1e-6] = 1.0
    train_z = np.nan_to_num((train_features - mean) / std)
    val_z = np.nan_to_num((val_features - mean) / std)

    indices = np.empty((len(val_z), top_k), dtype=np.int64)
    distances = np.empty((len(val_z), top_k), dtype=np.float64)
    weights = np.empty((len(val_z), top_k), dtype=np.float64)
    for query, feature in enumerate(val_z):
        distance = np.mean((train_z[bank_indices] - feature[None]) ** 2, axis=1)
        local = np.argsort(distance, kind="stable")[:top_k]
        selected = distance[local]
        temperature = max(float(np.median(selected)), 1e-6)
        logits = -(selected - selected.min()) / temperature
        weight = np.exp(logits - logits.max())
        weight /= weight.sum()
        indices[query] = bank_indices[local]
        distances[query] = selected
        weights[query] = weight
    return Retrieval(indices, distances, weights, bank_indices)


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.cumsum(values, axis=1)
    cumulative = np.concatenate([np.zeros((len(values), 1)), cumulative], axis=1)
    return (cumulative[:, width:] - cumulative[:, :-width]) / float(width)


def contiguous_runs(mask: np.ndarray, merge_gap: int = 1) -> list[tuple[int, int]]:
    points = np.flatnonzero(mask)
    if not len(points):
        return []
    groups = np.split(points, np.flatnonzero(np.diff(points) > merge_gap + 1) + 1)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def clustered_indices(mask: np.ndarray, severity: np.ndarray) -> list[int]:
    points = np.flatnonzero(mask)
    if not len(points):
        return []
    groups = np.split(points, np.flatnonzero(np.diff(points) > 1) + 1)
    return [int(group[np.argmax(np.abs(severity[group]))]) for group in groups]


def train_thresholds(
    forecast: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
    extreme_quantile: float,
    base_quantile: float,
) -> dict[str, object]:
    error = forecast - actual
    finite = valid & np.isfinite(error)
    selected = error[finite]
    thresholds: dict[str, object] = {
        "overestimate_base_mw": float(np.quantile(selected, base_quantile)),
        "overestimate_extreme_mw": float(np.quantile(selected, extreme_quantile)),
        "underestimate_base_mw": float(np.quantile(-selected, base_quantile)),
        "underestimate_extreme_mw": float(np.quantile(-selected, extreme_quantile)),
        "sustained_6h_mean_overestimate_mw": float(
            np.quantile(
                rolling_mean(error, 6)[
                    np.stack(
                        [
                            np.convolve(row.astype(int), np.ones(6, int), "valid") == 6
                            for row in valid
                        ]
                    )
                ],
                extreme_quantile,
            )
        ),
        "ramp_abs_mw": {},
    }
    for lag in RAMP_LAGS:
        ramp = actual[:, lag:] - actual[:, :-lag]
        ramp_valid = valid[:, lag:] & valid[:, :-lag] & np.isfinite(ramp)
        thresholds["ramp_abs_mw"][str(lag)] = float(
            np.quantile(np.abs(ramp[ramp_valid]), extreme_quantile)
        )
    return thresholds


def matching_excursion(
    error: np.ndarray,
    onset: int,
    direction: str,
    thresholds: dict[str, object],
) -> tuple[int, float]:
    if direction == "down":
        mask = error >= float(thresholds["overestimate_base_mw"])
    else:
        mask = -error >= float(thresholds["underestimate_base_mw"])
    runs = contiguous_runs(mask, merge_gap=1)
    containing = [run for run in runs if run[0] <= onset < run[1]]
    if not containing and runs:
        containing = [min(runs, key=lambda run: min(abs(run[0] - onset), abs(run[1] - 1 - onset)))]
        if min(abs(containing[0][0] - onset), abs(containing[0][1] - 1 - onset)) > 3:
            containing = []
    if not containing:
        return 1, float(abs(error[onset])) if 0 <= onset < len(error) else 0.0
    start, stop = containing[0]
    return stop - start, float(np.max(np.abs(error[start:stop])))


def extract_events(
    forecast: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
    thresholds: dict[str, object],
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    error = forecast - actual
    for issue in range(len(forecast)):
        issue_error = error[issue]
        issue_valid = valid[issue] & np.isfinite(issue_error)

        # Forecast-overestimate episodes.  Severe episodes lasting at least six
        # hours are also labelled as sustained deep drops.
        episode_mask = issue_valid & (
            issue_error >= float(thresholds["overestimate_base_mw"])
        )
        for start, stop in contiguous_runs(episode_mask, merge_gap=1):
            values = issue_error[start:stop]
            if not len(values) or float(np.max(values)) < float(
                thresholds["overestimate_extreme_mw"]
            ):
                continue
            base = {
                "split": split,
                "issue_index": issue,
                "direction": "down",
                "onset_hour": start,
                "duration_hours": stop - start,
                "depth_mw": float(np.max(values)),
                "mean_depth_mw": float(np.mean(values)),
                "ramp_severity_mw": float(
                    np.max(np.abs(np.diff(actual[issue, start:stop])))
                ) if stop - start > 1 else 0.0,
            }
            rows.append({"event_type": "forecast_overestimate", **base})
            six_score = (
                float(np.max(rolling_mean(issue_error[None], 6)))
                if stop - start >= 6
                else -np.inf
            )
            if stop - start >= 6 and six_score >= float(
                thresholds["sustained_6h_mean_overestimate_mw"]
            ):
                rows.append({"event_type": "sustained_deep_drop", **base})

        for lag in RAMP_LAGS:
            actual_ramp = actual[issue, lag:] - actual[issue, :-lag]
            forecast_ramp = forecast[issue, lag:] - forecast[issue, :-lag]
            ramp_valid = valid[issue, lag:] & valid[issue, :-lag]
            threshold = float(thresholds["ramp_abs_mw"][str(lag)])
            extreme = ramp_valid & (np.abs(actual_ramp) >= threshold)
            wrong = extreme & (np.sign(actual_ramp) != np.sign(forecast_ramp))
            weak = extreme & (np.abs(forecast_ramp) < 0.5 * np.abs(actual_ramp))
            missed = wrong | weak
            for family, event_mask in (
                (f"missed_ramp_{lag}h", missed),
                (f"wrong_direction_ramp_{lag}h", wrong),
            ):
                for local in clustered_indices(event_mask, actual_ramp):
                    onset = int(local + lag)
                    direction = "up" if actual_ramp[local] > 0 else "down"
                    duration, depth = matching_excursion(
                        issue_error, onset, direction, thresholds
                    )
                    rows.append(
                        {
                            "split": split,
                            "issue_index": issue,
                            "event_type": family,
                            "direction": direction,
                            "onset_hour": onset,
                            "duration_hours": int(duration),
                            "depth_mw": float(depth),
                            "mean_depth_mw": float(depth),
                            "ramp_severity_mw": float(abs(actual_ramp[local])),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no {split} events found")
    frame.insert(0, "event_id", [f"{split}_{i:05d}" for i in range(len(frame))])
    return frame


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values = np.asarray(values, float)[order]
    weights = np.asarray(weights, float)[order]
    cumulative = np.cumsum(weights) / max(float(weights.sum()), EPS)
    return float(values[min(int(np.searchsorted(cumulative, 0.5)), len(values) - 1)])


def build_event_lookup(events: pd.DataFrame) -> dict[tuple[int, str, str], dict[str, np.ndarray]]:
    lookup: dict[tuple[int, str, str], dict[str, np.ndarray]] = {}
    for keys, group in events.groupby(
        ["issue_index", "event_type", "direction"], sort=False
    ):
        lookup[(int(keys[0]), str(keys[1]), str(keys[2]))] = {
            name: group[name].to_numpy()
            for name in (
                "onset_hour", "duration_hours", "depth_mw", "ramp_severity_mw"
            )
        }
    return lookup


def predict_prior(
    event_type: str,
    direction: str,
    selected: np.ndarray,
    selected_weights: np.ndarray,
    event_lookup: dict[tuple[int, str, str], dict[str, np.ndarray]],
    bandwidth: float,
    attribute_radius: int,
) -> dict[str, object]:
    onset_parts = []
    duration_parts = []
    depth_parts = []
    ramp_parts = []
    weight_parts = []
    supporting_histories = 0
    for index, history in enumerate(selected):
        events = event_lookup.get((int(history), str(event_type), str(direction)))
        if events is None:
            continue
        count = len(events["onset_hour"])
        if not count:
            continue
        supporting_histories += 1
        onset_parts.append(events["onset_hour"].astype(float))
        duration_parts.append(events["duration_hours"].astype(float))
        depth_parts.append(events["depth_mw"].astype(float))
        ramp_parts.append(events["ramp_severity_mw"].astype(float))
        weight_parts.append(np.full(count, float(selected_weights[index]) / count))
    if not onset_parts:
        return {
            "event_supported": False,
            "support_history_fraction": 0.0,
            "predicted_onset_hour": np.nan,
            "onset_error_hours": np.nan,
            "onset_hit_6h": False,
            "onset_hit_12h": False,
            "predicted_duration_hours": np.nan,
            "duration_abs_error_hours": np.nan,
            "predicted_depth_mw": np.nan,
            "depth_abs_error_mw": np.nan,
            "predicted_ramp_severity_mw": np.nan,
            "ramp_severity_abs_error_mw": np.nan,
        }
    onset_values = np.concatenate(onset_parts)
    duration_values = np.concatenate(duration_parts)
    depth_values = np.concatenate(depth_parts)
    ramp_values = np.concatenate(ramp_parts)
    event_weights = np.concatenate(weight_parts)
    hours = np.arange(HOURS, dtype=float)
    hazard = np.zeros(HOURS, dtype=float)
    for onset, weight in zip(onset_values, event_weights):
        hazard += float(weight) * np.exp(
            -0.5 * ((hours - float(onset)) / max(bandwidth, 1e-6)) ** 2
        )
    predicted_onset = int(np.argmax(hazard))
    near = np.abs(onset_values - predicted_onset) <= attribute_radius
    if not np.any(near):
        near = np.ones(len(onset_values), dtype=bool)
    duration = weighted_median(duration_values[near], event_weights[near])
    depth = weighted_median(depth_values[near], event_weights[near])
    ramp = weighted_median(ramp_values[near], event_weights[near])
    return {
        "event_supported": True,
        "support_history_fraction": float(supporting_histories / len(selected)),
        "predicted_onset_hour": predicted_onset,
        "predicted_duration_hours": duration,
        "predicted_depth_mw": depth,
        "predicted_ramp_severity_mw": ramp,
    }


def score_prior(target: pd.Series, prior: dict[str, object]) -> dict[str, object]:
    if not bool(prior["event_supported"]):
        return {
            **prior,
            "onset_error_hours": np.nan,
            "onset_hit_6h": False,
            "onset_hit_12h": False,
            "duration_abs_error_hours": np.nan,
            "depth_abs_error_mw": np.nan,
            "ramp_severity_abs_error_mw": np.nan,
        }
    onset_error = abs(float(prior["predicted_onset_hour"]) - float(target.onset_hour))
    return {
        **prior,
        "onset_error_hours": onset_error,
        "onset_hit_6h": bool(onset_error <= 6),
        "onset_hit_12h": bool(onset_error <= 12),
        "duration_abs_error_hours": float(
            abs(float(prior["predicted_duration_hours"]) - float(target.duration_hours))
        ),
        "depth_abs_error_mw": float(
            abs(float(prior["predicted_depth_mw"]) - float(target.depth_mw))
        ),
        "ramp_severity_abs_error_mw": float(
            abs(float(prior["predicted_ramp_severity_mw"]) - float(target.ramp_severity_mw))
        ),
    }


def predict_one(
    target: pd.Series,
    selected: np.ndarray,
    selected_weights: np.ndarray,
    train_events_by_issue: dict[int, pd.DataFrame],
    bandwidth: float,
    attribute_radius: int,
) -> dict[str, object]:
    frames = []
    for issue, frame in train_events_by_issue.items():
        current = frame.copy()
        if "issue_index" not in current:
            current["issue_index"] = int(issue)
        frames.append(current)
    lookup = build_event_lookup(pd.concat(frames, ignore_index=True))
    prior = predict_prior(
        str(target.event_type), str(target.direction), selected, selected_weights,
        lookup, bandwidth, attribute_radius,
    )
    return score_prior(target, prior)


def evaluate_selection(
    val_events: pd.DataFrame,
    indices: np.ndarray,
    weights: np.ndarray,
    event_lookup: dict[tuple[int, str, str], dict[str, np.ndarray]],
    method: str,
    bandwidth: float,
    attribute_radius: int,
) -> pd.DataFrame:
    rows = []
    prior_cache: dict[tuple[int, str, str], dict[str, object]] = {}
    for target in val_events.itertuples(index=False):
        issue = int(target.issue_index)
        key = (issue, str(target.event_type), str(target.direction))
        if key not in prior_cache:
            prior_cache[key] = predict_prior(
                key[1], key[2], indices[issue], weights[issue], event_lookup,
                bandwidth, attribute_radius,
            )
        prediction = score_prior(pd.Series(target._asdict()), prior_cache[key])
        rows.append({"method": method, **target._asdict(), **prediction})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", frame)] + list(frame.groupby("event_type", sort=True))
    for event_type, group in groups:
        supported = group[group.event_supported]
        row = {
            "event_type": event_type,
            "event_count": int(len(group)),
            "event_recall": float(group.event_supported.mean()),
            "mean_support_history_fraction": float(group.support_history_fraction.mean()),
            "onset_hit_6h": float(group.onset_hit_6h.mean()),
            "onset_hit_12h": float(group.onset_hit_12h.mean()),
            "onset_mae_hours": float(supported.onset_error_hours.mean()) if len(supported) else np.nan,
            "duration_mae_hours": float(supported.duration_abs_error_hours.mean()) if len(supported) else np.nan,
            "depth_mae_mw": float(supported.depth_abs_error_mw.mean()) if len(supported) else np.nan,
            "ramp_severity_mae_mw": float(supported.ramp_severity_abs_error_mw.mean()) if len(supported) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def compare_random(retrieval: pd.DataFrame, random: pd.DataFrame) -> pd.DataFrame:
    higher = {"event_recall", "mean_support_history_fraction", "onset_hit_6h", "onset_hit_12h"}
    metrics = [
        "event_recall",
        "mean_support_history_fraction",
        "onset_hit_6h",
        "onset_hit_12h",
        "onset_mae_hours",
        "duration_mae_hours",
        "depth_mae_mw",
        "ramp_severity_mae_mw",
    ]
    rows = []
    for event_type in retrieval.event_type:
        actual_row = retrieval[retrieval.event_type.eq(event_type)].iloc[0]
        baseline = random[random.event_type.eq(event_type)]
        for metric in metrics:
            value = float(actual_row[metric])
            samples = baseline[metric].to_numpy(float)
            samples = samples[np.isfinite(samples)]
            if not len(samples) or not np.isfinite(value):
                continue
            random_mean = float(np.mean(samples))
            direction = "higher" if metric in higher else "lower"
            if direction == "higher":
                gain = value - random_mean
                relative = gain / max(abs(random_mean), EPS)
                p_value = (1 + np.sum(samples >= value)) / (len(samples) + 1)
            else:
                gain = random_mean - value
                relative = gain / max(abs(random_mean), EPS)
                p_value = (1 + np.sum(samples <= value)) / (len(samples) + 1)
            rows.append(
                {
                    "event_type": event_type,
                    "metric": metric,
                    "retrieval": value,
                    "random_mean": random_mean,
                    "random_q05": float(np.quantile(samples, 0.05)),
                    "random_q95": float(np.quantile(samples, 0.95)),
                    "absolute_gain": gain,
                    "relative_gain": relative,
                    "empirical_one_sided_p": float(p_value),
                    "better_direction": direction,
                }
            )
    return pd.DataFrame(rows)


def decision(comparison: pd.DataFrame) -> tuple[bool, dict[str, bool]]:
    overall = comparison[comparison.event_type.eq("all")].set_index("metric")

    def significant(metric: str, minimum_relative: float = 0.0) -> bool:
        return bool(
            metric in overall.index
            and float(overall.loc[metric, "relative_gain"]) >= minimum_relative
            and float(overall.loc[metric, "empirical_one_sided_p"]) <= 0.05
        )

    occurrence = significant("event_recall")
    timing = (
        significant("onset_hit_6h", 0.10)
        or significant("onset_hit_12h", 0.10)
    ) and significant("onset_mae_hours")
    severity_votes = sum(
        significant(metric, 0.10)
        for metric in (
            "duration_mae_hours",
            "depth_mae_mw",
            "ramp_severity_mae_mw",
        )
    )
    components = {
        "occurrence_better_than_random": occurrence,
        "timing_better_than_random": timing,
        "at_least_two_severity_attributes_better_than_random": severity_votes >= 2,
    }
    return bool(all(components.values())), components


def plot_comparison(comparison: pd.DataFrame, output: Path) -> None:
    metrics = ["event_recall", "onset_hit_6h", "onset_hit_12h"]
    all_rows = comparison[
        comparison.event_type.eq("all") & comparison.metric.isin(metrics)
    ].set_index("metric").loc[metrics]
    x = np.arange(len(metrics))
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, all_rows.retrieval, 0.36, label="forecast Top-40", color="#dc165d")
    axis.bar(x + 0.18, all_rows.random_mean, 0.36, label="random 40", color="#6b7280")
    axis.errorbar(
        x + 0.18,
        all_rows.random_mean,
        yerr=np.stack(
            [all_rows.random_mean - all_rows.random_q05, all_rows.random_q95 - all_rows.random_mean]
        ),
        fmt="none",
        color="black",
        capsize=4,
    )
    axis.set_xticks(x, ["Event recall", "Onset Hit ±6h", "Onset Hit ±12h"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Rate")
    axis.legend()
    axis.set_title("Historical event prior: forecast retrieval vs random histories")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(value) for value in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for record in frame.to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    thresholds: dict[str, object],
    passed: bool,
    components: dict[str, bool],
) -> None:
    overall = comparison[comparison.event_type.eq("all")].set_index("metric")

    def row(metric: str) -> str:
        value = overall.loc[metric]
        return (
            f"| {metric} | {value.retrieval:.4g} | {value.random_mean:.4g} "
            f"[{value.random_q05:.4g}, {value.random_q95:.4g}] | "
            f"{100 * value.relative_gain:+.1f}% | {value.empirical_one_sided_p:.4f} |"
        )

    metrics = [
        "event_recall", "onset_hit_6h", "onset_hit_12h", "onset_mae_hours",
        "duration_mae_hours", "depth_mae_mw", "ramp_severity_mae_mw",
    ]
    verdict = (
        "通过：forecast Top-40 历史 residual 同时提供了事件发生、时间和严重度信息，可进入 historical event prior 接入实验。"
        if passed
        else "未通过：forecast Top-40 历史 residual 尚未同时证明事件时间与严重度优于随机；按预注册规则停止接入路线。"
    )
    lines = [
        "# Forecast-only Top-40 历史事件先验零训练诊断",
        "",
        "## 结论",
        "",
        verdict,
        "",
        f"- 事件发生门：`{components['occurrence_better_than_random']}`",
        f"- onset 时间门：`{components['timing_better_than_random']}`",
        f"- 严重度门：`{components['at_least_two_severity_attributes_better_than_random']}`",
        "",
        "## 总体结果",
        "",
        "| 指标 | forecast Top-40 | 随机40：均值 [5%,95%] | 相对增益 | 单侧经验p |",
        "|---|---:|---:|---:|---:|",
        *[row(metric) for metric in metrics if metric in overall.index],
        "",
        "![检索与随机对照](figures/retrieval_vs_random.png)",
        "",
        "## 事件定义与防泄漏",
        "",
        "- 检索键严格只由当前发布 forecast 派生；现有检索键末尾的4维日历特征已在本诊断主结果中移除。",
        "- 候选和值均只来自 train；val actual 在 Top-40 索引固定后才载入评分；test 未读取。",
        "- 持续深跌/高估使用 train-only forecast-actual 阈值；1/3/6h漏报和错向 ramp 使用 train-only actual-ramp q90。",
        "- 历史事件不进行轨迹平均，而是形成 onset、duration、depth 和 ramp severity 的加权事件集合。",
        "- onset MAE、duration/depth/ramp MAE 是在存在同类历史支持的事件上计算；event recall 同时揭示无支持失败。",
        "",
        "## 预注册继续条件",
        "",
        "只有事件发生、onset时间和严重度三道门全部通过，才继续接入 mismatch tail。",
        "",
        "## Train-only阈值",
        "",
        "```json",
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分事件结果",
        "",
        markdown_table(summary),
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output_dir)
    if output.exists():
        if not args.finalize_existing:
            raise FileExistsError(f"refusing to overwrite {output}")
        required = [
            "retrieval_summary.csv", "retrieval_vs_random.csv",
            "train_event_catalog.csv", "validation_event_catalog.csv",
            "random_repeat_summary.csv",
        ]
        missing = [name for name in required if not (output / name).exists()]
        if missing:
            raise FileNotFoundError(f"cannot finalize; missing outputs: {missing}")
        summary = pd.read_csv(output / "retrieval_summary.csv")
        comparison = pd.read_csv(output / "retrieval_vs_random.csv")
        train_events = pd.read_csv(output / "train_event_catalog.csv")
        val_events = pd.read_csv(output / "validation_event_catalog.csv")
        passed, components = decision(comparison)

        stations = pd.read_csv(data_path / "station_order.csv").sort_values(
            "channel_index"
        ).reset_index(drop=True)
        wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
        capacities = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
        train_forecast = aggregate_wind(
            np.load(data_path / "train_forecast.npy"), wind_indices, capacities
        )
        train_actual = aggregate_wind(
            np.load(data_path / "train_actual.npy"), wind_indices, capacities
        )
        train_valid = valid_wind(
            np.load(data_path / "train_fill_mask.npy"), wind_indices
        )
        thresholds = train_thresholds(
            train_forecast, train_actual, train_valid,
            args.extreme_quantile, args.episode_base_quantile,
        )
        write_report(
            output / "historical_event_prior_diagnostic.md",
            summary, comparison, thresholds, passed, components,
        )
        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "purpose": "zero-training forecast-only Top-K historical residual event-prior audit",
            "data_path": str(data_path),
            "top_k": args.top_k,
            "random_repeats": args.random_repeats,
            "strict_forecast_only_key": True,
            "calendar_features_used": False,
            "train_event_count": int(len(train_events)),
            "validation_event_count": int(len(val_events)),
            "test_files_loaded": False,
            "validation_actual_used_for_retrieval": False,
            "passed_historical_event_prior_gate": passed,
            "decision_components": components,
            "thresholds": thresholds,
            "finalized_from_completed_csv_outputs": True,
        }
        (output / "diagnostic_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"HISTORICAL_EVENT_PRIOR_DIAGNOSTIC_FINALIZED output={output}")
        print(f"PASSED={passed} components={components}")
        return
    (output / "figures").mkdir(parents=True)

    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacities = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)

    # Critical ordering: neighbour indices are fixed before val actual is loaded.
    retrieval = build_strict_forecast_retrieval(data_path, args.top_k)
    train_forecast_raw = np.load(data_path / "train_forecast.npy")
    train_actual_raw = np.load(data_path / "train_actual.npy")
    train_fill = np.load(data_path / "train_fill_mask.npy")
    val_forecast_raw = np.load(data_path / "val_forecast.npy")
    val_fill = np.load(data_path / "val_fill_mask.npy")
    train_forecast = aggregate_wind(train_forecast_raw, wind_indices, capacities)
    train_actual = aggregate_wind(train_actual_raw, wind_indices, capacities)
    train_valid = valid_wind(train_fill, wind_indices)
    val_forecast = aggregate_wind(val_forecast_raw, wind_indices, capacities)

    thresholds = train_thresholds(
        train_forecast,
        train_actual,
        train_valid,
        args.extreme_quantile,
        args.episode_base_quantile,
    )
    train_events = extract_events(
        train_forecast, train_actual, train_valid, thresholds, "train"
    )
    train_event_lookup = build_event_lookup(train_events)

    # Validation targets enter only after retrieval and train event catalogue.
    val_actual_raw = np.load(data_path / "val_actual.npy")
    val_actual = aggregate_wind(val_actual_raw, wind_indices, capacities)
    val_valid = valid_wind(val_fill, wind_indices)
    val_events = extract_events(
        val_forecast, val_actual, val_valid, thresholds, "val"
    )

    uniform = np.full_like(retrieval.weights, 1.0 / args.top_k)
    primary_details = evaluate_selection(
        val_events,
        retrieval.indices,
        uniform,
        train_event_lookup,
        "forecast_top40_uniform",
        args.onset_bandwidth_hours,
        args.attribute_radius_hours,
    )
    weighted_details = evaluate_selection(
        val_events,
        retrieval.indices,
        retrieval.weights,
        train_event_lookup,
        "forecast_top40_distance_weighted",
        args.onset_bandwidth_hours,
        args.attribute_radius_hours,
    )
    primary_summary = summarize(primary_details)
    weighted_summary = summarize(weighted_details)
    primary_summary.insert(0, "method", "forecast_top40_uniform")
    weighted_summary.insert(0, "method", "forecast_top40_distance_weighted")

    rng = np.random.default_rng(args.seed)
    random_summaries = []
    for repeat in range(args.random_repeats):
        indices = np.stack(
            [
                rng.choice(retrieval.bank_indices, args.top_k, replace=False)
                for _ in range(len(val_forecast))
            ]
        )
        details = evaluate_selection(
            val_events,
            indices,
            uniform,
            train_event_lookup,
            f"random_{repeat:04d}",
            args.onset_bandwidth_hours,
            args.attribute_radius_hours,
        )
        current = summarize(details)
        current.insert(0, "repeat", repeat)
        random_summaries.append(current)
    random_summary = pd.concat(random_summaries, ignore_index=True)
    comparison = compare_random(primary_summary, random_summary)
    passed, components = decision(comparison)

    train_events.to_csv(output / "train_event_catalog.csv", index=False)
    val_events.to_csv(output / "validation_event_catalog.csv", index=False)
    pd.concat([primary_details, weighted_details], ignore_index=True).to_csv(
        output / "per_event_predictions.csv", index=False
    )
    pd.concat([primary_summary, weighted_summary], ignore_index=True).to_csv(
        output / "retrieval_summary.csv", index=False
    )
    random_summary.to_csv(output / "random_repeat_summary.csv", index=False)
    comparison.to_csv(output / "retrieval_vs_random.csv", index=False)
    neighbor_rows = []
    train_issues = pd.read_csv(data_path / "train_issue_dates.csv")
    val_issues = pd.read_csv(data_path / "val_issue_dates.csv")
    for query in range(len(retrieval.indices)):
        for rank, index in enumerate(retrieval.indices[query], start=1):
            neighbor_rows.append(
                {
                    "validation_issue_index": query,
                    "validation_issue_date": val_issues.iloc[query].issue_date,
                    "rank": rank,
                    "train_issue_index": int(index),
                    "train_issue_date": train_issues.iloc[int(index)].issue_date,
                    "distance": float(retrieval.distances[query, rank - 1]),
                    "weight": float(retrieval.weights[query, rank - 1]),
                }
            )
    pd.DataFrame(neighbor_rows).to_csv(output / "retrieved_neighbors.csv", index=False)
    plot_comparison(comparison, output / "figures" / "retrieval_vs_random.png")
    write_report(
        output / "historical_event_prior_diagnostic.md",
        pd.concat([primary_summary, weighted_summary], ignore_index=True),
        comparison,
        thresholds,
        passed,
        components,
    )
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "zero-training forecast-only Top-K historical residual event-prior audit",
        "data_path": str(data_path),
        "top_k": args.top_k,
        "random_repeats": args.random_repeats,
        "strict_forecast_only_key": True,
        "calendar_features_used": False,
        "train_bank_count": int(len(retrieval.bank_indices)),
        "validation_issue_count": int(len(val_forecast)),
        "train_event_count": int(len(train_events)),
        "validation_event_count": int(len(val_events)),
        "test_files_loaded": False,
        "validation_actual_used_for_retrieval": False,
        "passed_historical_event_prior_gate": passed,
        "decision_components": components,
        "thresholds": thresholds,
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"HISTORICAL_EVENT_PRIOR_DIAGNOSTIC_COMPLETE output={output}")
    print(f"PASSED={passed} components={components}")


if __name__ == "__main__":
    main()
