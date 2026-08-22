#!/usr/bin/env python3
"""Train-only feasibility audit for event prediction and residual tail graphs.

This diagnostic deliberately does not train or modify the diffusion model.  It asks:

1. Can future forecast-failure severity be ranked from information available at
   issuance time?
2. Is a lower-tail residual co-exceedance graph stable and more informative
   about future synchronous misses than geographic or ordinary-correlation
   priors?

Only ``train_*`` arrays are loaded.  Expanding temporal folds use a six-issue
embargo so that the 168-hour target windows of fit and validation issues do not
overlap.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr


WIND_COUNT = 13
HOURS = 168
EMBARGO_ISSUES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--first-validation-index", type=int, default=122)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--validation-check",
        action="store_true",
        help="fit on all train issues and perform a fixed secondary check on val",
    )
    return parser.parse_args()


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    cumsum = np.cumsum(values, axis=0, dtype=np.float64)
    cumsum = np.concatenate([np.zeros((1,) + values.shape[1:]), cumsum], axis=0)
    return (cumsum[width:] - cumsum[:-width]) / float(width)


def safe_max(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def safe_quantile(values: np.ndarray, q: float) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def build_available_features(
    forecast: np.ndarray,
    residual: np.ndarray,
    issues: pd.DataFrame,
    capacities: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    weights = capacities[:WIND_COUNT] / capacities[:WIND_COUNT].sum()
    wind_forecast = np.einsum("nth,h->nt", forecast[:, :, :WIND_COUNT], weights)
    wind_residual = np.einsum("nth,h->nt", residual[:, :, :WIND_COUNT], weights)
    dates = pd.to_datetime(issues["issue_date"])

    rows: list[list[float]] = []
    names: list[str] | None = None
    for index in range(len(forecast)):
        series = wind_forecast[index]
        row: list[float] = []
        current_names: list[str] = []

        for day in range(7):
            block = series[day * 24 : (day + 1) * 24]
            row.extend([float(block.mean()), float(block.std())])
            current_names.extend([f"forecast_day{day+1}_mean", f"forecast_day{day+1}_std"])
        row.extend(
            [
                float(series.mean()),
                float(series.std()),
                float(series.min()),
                float(series.max()),
                safe_max(-(series[1:] - series[:-1])),
                safe_max(series[1:] - series[:-1]),
                safe_max(-(series[3:] - series[:-3])),
                safe_max(series[3:] - series[:-3]),
                safe_max(-(series[6:] - series[:-6])),
                safe_max(series[6:] - series[:-6]),
                float(rolling_mean(series[:, None], 6).min()),
                float(rolling_mean(series[:, None], 6).max()),
            ]
        )
        current_names.extend(
            [
                "forecast_mean",
                "forecast_std",
                "forecast_min",
                "forecast_max",
                "forecast_max_down_1h",
                "forecast_max_up_1h",
                "forecast_max_down_3h",
                "forecast_max_up_3h",
                "forecast_max_down_6h",
                "forecast_max_up_6h",
                "forecast_min_6h_mean",
                "forecast_max_6h_mean",
            ]
        )

        previous_available = (
            index > 0 and (dates.iloc[index] - dates.iloc[index - 1]).days == 1
        )
        if previous_available:
            revision = series[:144] - wind_forecast[index - 1, 24:]
            recent_error = wind_residual[index - 1, :24]
        else:
            revision = np.zeros(144, dtype=np.float64)
            recent_error = np.zeros(24, dtype=np.float64)
        for day in range(6):
            block = revision[day * 24 : (day + 1) * 24]
            row.extend([float(block.mean()), float(np.abs(block).mean())])
            current_names.extend(
                [f"revision_day{day+1}_mean", f"revision_day{day+1}_abs_mean"]
            )
        row.extend(
            [
                float(revision.mean()),
                float(revision.std()),
                float(np.abs(revision).max()),
                float(previous_available),
            ]
        )
        current_names.extend(
            [
                "revision_mean",
                "revision_std",
                "revision_abs_max",
                "revision_available",
            ]
        )

        row.extend(
            [
                float(recent_error.mean()),
                float(recent_error.std()),
                float(recent_error.min()),
                float(recent_error.max()),
                safe_quantile(recent_error, 0.10),
                float(np.abs(recent_error).max()),
                float(previous_available),
            ]
        )
        current_names.extend(
            [
                "recent_error_mean",
                "recent_error_std",
                "recent_error_min",
                "recent_error_max",
                "recent_error_q10",
                "recent_error_abs_max",
                "recent_error_available",
            ]
        )

        month_phase = 2.0 * math.pi * (dates.iloc[index].month - 1) / 12.0
        weekday_phase = 2.0 * math.pi * dates.iloc[index].weekday() / 7.0
        row.extend(
            [
                math.sin(month_phase),
                math.cos(month_phase),
                math.sin(weekday_phase),
                math.cos(weekday_phase),
            ]
        )
        current_names.extend(
            ["month_sin", "month_cos", "weekday_sin", "weekday_cos"]
        )
        if names is None:
            names = current_names
        elif names != current_names:
            raise RuntimeError("feature name mismatch")
        rows.append(row)

    assert names is not None
    name_to_index = {name: i for i, name in enumerate(names)}
    forecast_names = [name for name in names if name.startswith("forecast_")]
    calendar_names = ["month_sin", "month_cos", "weekday_sin", "weekday_cos"]
    revision_names = [name for name in names if name.startswith("revision_")]
    recent_names = [name for name in names if name.startswith("recent_error_")]
    groups = {
        "forecast_only": [name_to_index[n] for n in forecast_names + calendar_names],
        "forecast_revision": [
            name_to_index[n] for n in forecast_names + revision_names + calendar_names
        ],
        "all_available": [
            name_to_index[n]
            for n in forecast_names + revision_names + recent_names + calendar_names
        ],
    }
    return np.asarray(rows, dtype=np.float64), names, groups


def build_issue_severity(
    forecast: np.ndarray,
    actual: np.ndarray,
    fill_mask: np.ndarray,
    capacities: np.ndarray,
    issues: pd.DataFrame,
    station_lower_tail_threshold: np.ndarray,
) -> pd.DataFrame:
    weights = capacities[:WIND_COUNT] / capacities[:WIND_COUNT].sum()
    f = np.einsum("nth,h->nt", forecast[:, :, :WIND_COUNT], weights)
    y = np.einsum("nth,h->nt", actual[:, :, :WIND_COUNT], weights)
    complete = np.all(fill_mask[:, :, :WIND_COUNT] == 0, axis=2)
    rows = []
    for issue in range(len(forecast)):
        valid6 = np.convolve(complete[issue].astype(np.int32), np.ones(6, int), "valid") == 6
        over = rolling_mean((f[issue] - y[issue])[:, None], 6)[:, 0]
        valid_over = np.where(valid6, over, np.nan)
        deep6 = safe_max(valid_over)
        # A damaged issue can contain no complete six-hour window.  Keep the
        # diagnostic auditable instead of letting nanargmax abort the run.
        deep_start = int(np.nanargmax(valid_over)) if np.any(np.isfinite(valid_over)) else 0
        deep_stop = deep_start + 6
        event_timestamp = pd.Timestamp(issues.iloc[issue]["target_start"]) + pd.Timedelta(
            hours=deep_start
        )
        station_window = actual[issue, deep_start:deep_stop, :WIND_COUNT] - forecast[
            issue, deep_start:deep_stop, :WIND_COUNT
        ]
        station_window_mean = np.nanmean(station_window, axis=0)
        synchronous_fraction = float(
            np.mean(station_window_mean <= station_lower_tail_threshold)
        )
        hourly_over = np.maximum(f[issue] - y[issue], 0.0)
        duration_threshold = max(0.5 * deep6, 1e-8)
        active = hourly_over >= duration_threshold
        anchor = min(deep_start + 2, HOURS - 1)
        left = anchor
        right = anchor
        while left > 0 and active[left - 1]:
            left -= 1
        while right + 1 < HOURS and active[right + 1]:
            right += 1
        duration = int(right - left + 1) if active[anchor] else 6
        recovery_stop = min(deep_stop + 6, HOURS)
        recovery = float(
            np.mean(y[issue, deep_stop:recovery_stop])
            - np.mean(y[issue, deep_start:deep_stop])
        ) if recovery_stop > deep_stop else float("nan")
        valid3 = complete[issue, 3:] & complete[issue, :-3]
        mismatch3 = np.abs(
            (y[issue, 3:] - y[issue, :-3])
            - (f[issue, 3:] - f[issue, :-3])
        )
        ramp3 = safe_max(np.where(valid3, mismatch3, np.nan))
        rows.append(
            {
                "sample_index": issue,
                "deep_drop_6h_severity": deep6,
                "deep_drop_lead_start": deep_start,
                "deep_drop_lead_end": deep_stop - 1,
                "deep_drop_lead_day": deep_start // 24 + 1,
                "deep_drop_event_timestamp": event_timestamp.isoformat(),
                "deep_drop_relative_duration_hours": duration,
                "deep_drop_synchronous_station_fraction": synchronous_fraction,
                "deep_drop_recovery_6h_pu": recovery,
                "ramp_mismatch_3h_severity": ramp3,
            }
        )
    return pd.DataFrame(rows)


def build_independent_event_catalog(
    severity: pd.DataFrame,
    issues: pd.DataFrame,
    source: str,
    thresholds: dict[float, float],
    merge_gap_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge overlapping issue windows that point to the same physical event."""

    catalog_rows = []
    mapping_rows = []
    for quantile, threshold in thresholds.items():
        selected = severity[
            severity["deep_drop_6h_severity"] >= float(threshold)
        ].copy()
        selected["event_time"] = pd.to_datetime(
            selected["deep_drop_event_timestamp"]
        )
        selected = selected.sort_values("event_time")
        groups: list[list[int]] = []
        current: list[int] = []
        previous: pd.Timestamp | None = None
        for row_index, row in selected.iterrows():
            timestamp = pd.Timestamp(row["event_time"])
            if previous is None or (
                timestamp - previous
            ) > pd.Timedelta(hours=merge_gap_hours):
                if current:
                    groups.append(current)
                current = [row_index]
            else:
                current.append(row_index)
            previous = timestamp
        if current:
            groups.append(current)

        for group_number, indices in enumerate(groups, start=1):
            group = selected.loc[indices]
            representative_index = group["deep_drop_6h_severity"].idxmax()
            representative = severity.loc[representative_index]
            event_id = f"{source}_q{int(quantile * 100)}_event_{group_number:03d}"
            sample_index = int(representative["sample_index"])
            catalog_rows.append(
                {
                    "event_id": event_id,
                    "source": source,
                    "label_quantile": quantile,
                    "threshold": float(threshold),
                    "member_issue_count": int(len(group)),
                    "first_event_time": group["event_time"].min().isoformat(),
                    "last_event_time": group["event_time"].max().isoformat(),
                    "representative_sample_index": sample_index,
                    "representative_issue_date": issues.iloc[sample_index]["issue_date"],
                    "severity": float(representative["deep_drop_6h_severity"]),
                    "lead_start": int(representative["deep_drop_lead_start"]),
                    "lead_end": int(representative["deep_drop_lead_end"]),
                    "lead_day": int(representative["deep_drop_lead_day"]),
                    "relative_duration_hours": int(
                        representative["deep_drop_relative_duration_hours"]
                    ),
                    "synchronous_station_fraction": float(
                        representative["deep_drop_synchronous_station_fraction"]
                    ),
                    "recovery_6h_pu": float(
                        representative["deep_drop_recovery_6h_pu"]
                    ),
                    "merge_gap_hours": int(merge_gap_hours),
                }
            )
            for _, member in group.iterrows():
                member_sample = int(member["sample_index"])
                mapping_rows.append(
                    {
                        "event_id": event_id,
                        "source": source,
                        "label_quantile": quantile,
                        "sample_index": member_sample,
                        "issue_date": issues.iloc[member_sample]["issue_date"],
                        "event_timestamp": member["deep_drop_event_timestamp"],
                        "severity": float(member["deep_drop_6h_severity"]),
                        "is_representative": member_sample == sample_index,
                    }
                )
    return pd.DataFrame(catalog_rows), pd.DataFrame(mapping_rows)


def expanding_folds(n: int, first_validation: int, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    validation_indices = np.arange(first_validation, n)
    blocks = [block for block in np.array_split(validation_indices, folds) if len(block)]
    output = []
    for block in blocks:
        train_stop = int(block[0]) - EMBARGO_ISSUES
        if train_stop < 60:
            raise ValueError("insufficient temporal training data before validation block")
        output.append((np.arange(train_stop), block))
    return output


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    seed: int,
    l2: float = 0.15,
) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = np.nan_to_num((x_train - mean) / std)
    x_eval = np.nan_to_num((x_eval - mean) / std)
    torch.manual_seed(seed)
    xt = torch.as_tensor(x_train, dtype=torch.float64)
    yt = torch.as_tensor(y_train, dtype=torch.float64)
    xe = torch.as_tensor(x_eval, dtype=torch.float64)
    weight = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    bias_value = np.clip(float(y_train.mean()), 1e-4, 1 - 1e-4)
    bias = torch.tensor(
        math.log(bias_value / (1 - bias_value)), dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=250, line_search_fn="strong_wolfe", tolerance_grad=1e-9
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = xt @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + 0.5 * l2 * torch.mean(weight.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.sigmoid(xe @ weight + bias).cpu().numpy()


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positive = y == 1
    negative = y == 0
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float(
        (ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
        / (positive.sum() * negative.sum())
    )


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    if positives == 0:
        return float("nan")
    # Add all equal-score observations at the same threshold.  This makes a
    # constant climatology score return AP=prevalence instead of depending on
    # the arbitrary original row order.
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order]
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_score) != 0) + 1, len(y)]
    starts = np.r_[0, boundaries[:-1]]
    true_positive = 0
    predicted_positive = 0
    ap = 0.0
    for start, stop in zip(starts, boundaries):
        group_positive = int(sorted_y[start:stop].sum())
        true_positive += group_positive
        predicted_positive += int(stop - start)
        ap += (group_positive / positives) * (true_positive / predicted_positive)
    return float(ap)


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    score = np.clip(score, 1e-6, 1 - 1e-6)
    prevalence = float(y.mean())
    top_count = max(1, int(math.ceil(0.20 * len(y))))
    sorted_score = np.sort(score)[::-1]
    threshold = sorted_score[top_count - 1]
    selected = score >= threshold
    top_rate = float(y[selected].mean())
    return {
        "sample_count": int(len(y)),
        "positive_count": int(y.sum()),
        "prevalence": prevalence,
        "roc_auc": roc_auc(y, score),
        "pr_auc": average_precision(y, score),
        "pr_auc_lift_over_prevalence": average_precision(y, score) / prevalence
        if prevalence > 0
        else float("nan"),
        "brier": float(np.mean((score - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(score) + (1 - y) * np.log(1 - score))),
        "top20_event_rate": top_rate,
        "top20_lift": top_rate / prevalence if prevalence > 0 else float("nan"),
    }


def run_predictability(
    features: np.ndarray,
    feature_groups: dict[str, list[int]],
    severity: pd.DataFrame,
    issues: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = []
    thresholds = []
    event_columns = {
        "deep_drop_6h": "deep_drop_6h_severity",
        "ramp_mismatch_3h": "ramp_mismatch_3h_severity",
    }
    for event_name, column in event_columns.items():
        values = severity[column].to_numpy(dtype=np.float64)
        for quantile in (0.80, 0.90):
            for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
                threshold = float(np.quantile(values[train_idx], quantile))
                y_train = (values[train_idx] >= threshold).astype(np.float64)
                y_val = (values[val_idx] >= threshold).astype(np.int64)
                thresholds.append(
                    {
                        "event": event_name,
                        "label_quantile": quantile,
                        "fold": fold_id,
                        "train_count": len(train_idx),
                        "validation_count": len(val_idx),
                        "threshold": threshold,
                        "train_positive_rate": float(y_train.mean()),
                        "validation_positive_rate": float(y_val.mean()),
                        "train_last_issue": issues.iloc[train_idx[-1]]["issue_date"],
                        "validation_first_issue": issues.iloc[val_idx[0]]["issue_date"],
                        "validation_last_issue": issues.iloc[val_idx[-1]]["issue_date"],
                        "embargo_issue_days": EMBARGO_ISSUES,
                    }
                )
                baseline_score = np.full(len(val_idx), y_train.mean(), dtype=np.float64)
                for local, issue_idx in enumerate(val_idx):
                    records.append(
                        {
                            "event": event_name,
                            "label_quantile": quantile,
                            "fold": fold_id,
                            "sample_index": int(issue_idx),
                            "issue_date": issues.iloc[issue_idx]["issue_date"],
                            "severity": values[issue_idx],
                            "label": int(y_val[local]),
                            "model": "rolling_climatology",
                            "score": baseline_score[local],
                        }
                    )
                for group_name, columns in feature_groups.items():
                    score = fit_logistic(
                        features[train_idx][:, columns],
                        y_train,
                        features[val_idx][:, columns],
                        seed=seed + fold_id,
                    )
                    for local, issue_idx in enumerate(val_idx):
                        records.append(
                            {
                                "event": event_name,
                                "label_quantile": quantile,
                                "fold": fold_id,
                                "sample_index": int(issue_idx),
                                "issue_date": issues.iloc[issue_idx]["issue_date"],
                                "severity": values[issue_idx],
                                "label": int(y_val[local]),
                                "model": group_name,
                                "score": float(score[local]),
                            }
                        )
    prediction_frame = pd.DataFrame(records)
    metric_rows = []
    for keys, group in prediction_frame.groupby(["event", "label_quantile", "model"]):
        metric = binary_metrics(group["label"].to_numpy(), group["score"].to_numpy())
        rho = spearmanr(group["score"], group["severity"]).statistic
        metric_rows.append(
            {
                "event": keys[0],
                "label_quantile": keys[1],
                "model": keys[2],
                **metric,
                "score_severity_spearman": float(rho),
            }
        )
    return prediction_frame, pd.DataFrame(metric_rows), pd.DataFrame(thresholds)


def run_external_validation(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    feature_groups: dict[str, list[int]],
    train_severity: pd.DataFrame,
    validation_severity: pd.DataFrame,
    validation_issues: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    event_columns = {
        "deep_drop_6h": "deep_drop_6h_severity",
        "ramp_mismatch_3h": "ramp_mismatch_3h_severity",
    }
    for event_name, column in event_columns.items():
        train_values = train_severity[column].to_numpy(dtype=np.float64)
        validation_values = validation_severity[column].to_numpy(dtype=np.float64)
        for quantile in (0.80, 0.90):
            threshold = float(np.quantile(train_values, quantile))
            y_train = (train_values >= threshold).astype(np.float64)
            y_validation = (validation_values >= threshold).astype(np.int64)
            scores = {"train_climatology": np.full(len(y_validation), y_train.mean())}
            for group_name, columns in feature_groups.items():
                scores[group_name] = fit_logistic(
                    train_features[:, columns],
                    y_train,
                    validation_features[:, columns],
                    seed=seed + int(quantile * 100),
                )
            for model_name, score in scores.items():
                for index in range(len(y_validation)):
                    records.append(
                        {
                            "event": event_name,
                            "label_quantile": quantile,
                            "sample_index": index,
                            "issue_date": validation_issues.iloc[index]["issue_date"],
                            "threshold_fitted_on_train": threshold,
                            "severity": validation_values[index],
                            "label": int(y_validation[index]),
                            "model": model_name,
                            "score": float(score[index]),
                        }
                    )
    prediction_frame = pd.DataFrame(records)
    metric_rows = []
    for keys, group in prediction_frame.groupby(["event", "label_quantile", "model"]):
        metric = binary_metrics(group["label"].to_numpy(), group["score"].to_numpy())
        rho = spearmanr(group["score"], group["severity"]).statistic
        metric_rows.append(
            {
                "event": keys[0],
                "label_quantile": keys[1],
                "model": keys[2],
                **metric,
                "score_severity_spearman": float(rho),
            }
        )
    return prediction_frame, pd.DataFrame(metric_rows)


def episode_summary(predictions: pd.DataFrame, source: str) -> pd.DataFrame:
    rows = []
    first_model = sorted(predictions["model"].unique())[0]
    labels = predictions[predictions.model == first_model]
    for (event, quantile), group in labels.groupby(["event", "label_quantile"]):
        positive_dates = sorted(pd.to_datetime(group.loc[group.label == 1, "issue_date"]))
        episode_count = 0
        previous = None
        for date in positive_dates:
            if previous is None or (date - previous).days > EMBARGO_ISSUES:
                episode_count += 1
            previous = date
        rows.append(
            {
                "source": source,
                "event": event,
                "label_quantile": quantile,
                "positive_issue_count": len(positive_dates),
                "conservative_nonoverlap_episode_count": episode_count,
                "episode_merge_gap_days": EMBARGO_ISSUES,
            }
        )
    return pd.DataFrame(rows)


def corr_graph(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)


def tail_graph(values: np.ndarray, thresholds: np.ndarray | None = None, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.quantile(values, q, axis=0)
    events = values <= thresholds[None, :]
    probability = events.mean(axis=0)
    joint = events.astype(np.float64).T @ events.astype(np.float64) / len(events)
    denominator = np.sqrt(np.outer(probability, probability))
    graph = np.divide(joint, denominator, out=np.zeros_like(joint), where=denominator > 0)
    np.fill_diagonal(graph, 1.0)
    return graph, thresholds


def upper(graph: np.ndarray) -> np.ndarray:
    return graph[np.triu_indices_from(graph, k=1)]


def graph_similarity(name: str, left: np.ndarray, right: np.ndarray, top_k: int = 20) -> dict[str, float | str]:
    a, b = upper(left), upper(right)
    k = min(top_k, len(a))
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return {
        "comparison": name,
        "edge_spearman": float(spearmanr(a, b).statistic),
        "edge_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "top20_edge_overlap": float(len(top_a & top_b) / k),
    }


def unique_actual_rows(
    actual: np.ndarray,
    fill_mask: np.ndarray,
    issues: pd.DataFrame,
    issue_indices: Iterable[int],
) -> np.ndarray:
    observed: dict[pd.Timestamp, np.ndarray] = {}
    for issue_index in issue_indices:
        start = pd.Timestamp(issues.iloc[issue_index]["target_start"])
        for lead in range(HOURS):
            if np.any(fill_mask[issue_index, lead, :WIND_COUNT] != 0):
                continue
            timestamp = start + pd.Timedelta(hours=lead)
            observed.setdefault(timestamp, actual[issue_index, lead, :WIND_COUNT])
    return np.stack(list(observed.values()), axis=0)


def valid_residual_rows(
    residual: np.ndarray,
    fill_mask: np.ndarray,
    issue_indices: Iterable[int],
    lead_days: set[int] | None = None,
) -> np.ndarray:
    issue_indices = np.asarray(list(issue_indices), dtype=np.int64)
    values = residual[issue_indices, :, :WIND_COUNT]
    masks = fill_mask[issue_indices, :, :WIND_COUNT]
    valid = np.all(masks == 0, axis=2)
    if lead_days is not None:
        day_mask = np.array([(hour // 24 + 1) in lead_days for hour in range(HOURS)])
        valid &= day_mask[None, :]
    return values[valid]


def run_graph_audit(
    residual: np.ndarray,
    actual: np.ndarray,
    fill_mask: np.ndarray,
    issues: pd.DataFrame,
    adjacency: np.ndarray,
    distance: np.ndarray,
    stations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    n = len(residual)
    full_idx = np.arange(n)
    split = int(round(0.70 * n))
    fit_idx = np.arange(split - EMBARGO_ISSUES)
    held_idx = np.arange(split, n)
    first_idx = np.arange(n // 2 - EMBARGO_ISSUES)
    second_idx = np.arange(n // 2, n)

    full_residual = valid_residual_rows(residual, fill_mask, full_idx)
    full_actual = unique_actual_rows(actual, fill_mask, issues, full_idx)
    actual_corr = corr_graph(full_actual)
    residual_corr = corr_graph(full_residual)
    tail10, _ = tail_graph(full_residual, q=0.10)
    tail05, _ = tail_graph(full_residual, q=0.05)
    sigma = float(np.median(distance[:WIND_COUNT, :WIND_COUNT][distance[:WIND_COUNT, :WIND_COUNT] > 0]))
    geo_similarity = np.exp(-np.square(distance[:WIND_COUNT, :WIND_COUNT] / sigma))
    np.fill_diagonal(geo_similarity, 1.0)

    first_rows = valid_residual_rows(residual, fill_mask, first_idx)
    second_rows = valid_residual_rows(residual, fill_mask, second_idx)
    first_tail, _ = tail_graph(first_rows, q=0.10)
    second_tail, _ = tail_graph(second_rows, q=0.10)
    early_lead = valid_residual_rows(residual, fill_mask, full_idx, {1, 2, 3})
    late_lead = valid_residual_rows(residual, fill_mask, full_idx, {5, 6, 7})
    early_tail, _ = tail_graph(early_lead, q=0.10)
    late_tail, _ = tail_graph(late_lead, q=0.10)

    fit_residual = valid_residual_rows(residual, fill_mask, fit_idx)
    held_residual = valid_residual_rows(residual, fill_mask, held_idx)
    fit_tail, fit_threshold = tail_graph(fit_residual, q=0.10)
    held_tail, _ = tail_graph(held_residual, thresholds=fit_threshold, q=0.10)
    fit_residual_corr = corr_graph(fit_residual)
    fit_actual_corr = corr_graph(unique_actual_rows(actual, fill_mask, issues, fit_idx))

    similarities = [
        graph_similarity("tail_q10_vs_geographic_similarity", tail10, geo_similarity),
        graph_similarity("tail_q10_vs_geographic_adjacency", tail10, adjacency[:WIND_COUNT, :WIND_COUNT]),
        graph_similarity("tail_q10_vs_actual_power_correlation", tail10, actual_corr),
        graph_similarity("tail_q10_vs_residual_correlation", tail10, residual_corr),
        graph_similarity("tail_q10_first_half_vs_second_half", first_tail, second_tail),
        graph_similarity("tail_q10_lead_days_1_3_vs_5_7", early_tail, late_tail),
        graph_similarity("tail_q10_vs_tail_q05", tail10, tail05),
    ]
    predictive = []
    priors = {
        "geographic_similarity": geo_similarity,
        "geographic_adjacency": adjacency[:WIND_COUNT, :WIND_COUNT],
        "actual_power_correlation": fit_actual_corr,
        "ordinary_residual_correlation": fit_residual_corr,
        "residual_lower_tail_graph": fit_tail,
    }
    for name, prior in priors.items():
        row = graph_similarity(f"{name}_vs_future_tail_graph", prior, held_tail)
        row["prior"] = name
        row["fit_issue_count"] = len(fit_idx)
        row["embargo_issue_count"] = EMBARGO_ISSUES
        row["heldout_issue_count"] = len(held_idx)
        predictive.append(row)

    names = stations.iloc[:WIND_COUNT]["FARM_NAME"].tolist()
    station_ids = stations.iloc[:WIND_COUNT]["station_id"].astype(int).tolist()
    edge_rows = []
    for i in range(WIND_COUNT):
        for j in range(i + 1, WIND_COUNT):
            edge_rows.append(
                {
                    "station_i": station_ids[i],
                    "station_i_name": names[i],
                    "station_j": station_ids[j],
                    "station_j_name": names[j],
                    "distance_km": float(distance[i, j]),
                    "geographic_adjacency": float(adjacency[i, j]),
                    "actual_power_correlation": float(actual_corr[i, j]),
                    "ordinary_residual_correlation": float(residual_corr[i, j]),
                    "residual_lower_tail_q10": float(tail10[i, j]),
                    "residual_lower_tail_q05": float(tail05[i, j]),
                    "future_tail_score": float(held_tail[i, j]),
                }
            )
    edge_frame = pd.DataFrame(edge_rows).sort_values("residual_lower_tail_q10", ascending=False)
    graphs = {
        "geographic_similarity": geo_similarity,
        "geographic_adjacency": adjacency[:WIND_COUNT, :WIND_COUNT],
        "actual_power_correlation": actual_corr,
        "ordinary_residual_correlation": residual_corr,
        "residual_lower_tail_q10": tail10,
        "residual_lower_tail_q05": tail05,
        "first_half_tail_q10": first_tail,
        "second_half_tail_q10": second_tail,
        "early_lead_tail_q10": early_tail,
        "late_lead_tail_q10": late_tail,
        "heldout_future_tail_q10": held_tail,
    }
    return pd.DataFrame(similarities), pd.DataFrame(predictive), graphs, edge_frame


def plot_predictability(metrics: pd.DataFrame, output: Path) -> None:
    events = ["deep_drop_6h", "ramp_mismatch_3h"]
    models = ["rolling_climatology", "forecast_only", "forecast_revision", "all_available"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for row, event in enumerate(events):
        for col, quantile in enumerate((0.80, 0.90)):
            ax = axes[row, col]
            subset = metrics[(metrics.event == event) & (metrics.label_quantile == quantile)].set_index("model")
            values = [subset.loc[m, "pr_auc_lift_over_prevalence"] for m in models]
            ax.bar(np.arange(len(models)), values, color=["#9ca3af", "#4c78a8", "#f2cf5b", "#e45756"])
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_xticks(np.arange(len(models)), ["climatology", "forecast", "+revision", "+recent error"], rotation=20)
            ax.set_ylabel("PR-AUC / prevalence")
            ax.set_title(f"{event}, training q{int(quantile*100)} label")
    fig.suptitle("Strict rolling predictability of future forecast-failure events")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_graphs(graphs: dict[str, np.ndarray], output: Path) -> None:
    selected = [
        "geographic_adjacency",
        "actual_power_correlation",
        "ordinary_residual_correlation",
        "residual_lower_tail_q10",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    for ax, name in zip(axes, selected):
        image = ax.imshow(graphs[name], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("wind station index")
        ax.set_ylabel("wind station index")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.suptitle("Train-only wind-station spatial priors")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    predictability: pd.DataFrame,
    similarity: pd.DataFrame,
    predictive: pd.DataFrame,
    edges: pd.DataFrame,
    metadata: dict[str, object],
    episodes: pd.DataFrame,
    validation_metrics: pd.DataFrame | None = None,
) -> None:
    best_rows = []
    for (event, q), group in predictability.groupby(["event", "label_quantile"]):
        candidates = group[group.model != "rolling_climatology"].sort_values("pr_auc_lift_over_prevalence", ascending=False)
        best = candidates.iloc[0]
        best_rows.append(
            {
                "event": event,
                "q": int(q * 100),
                "model": best.model,
                "prevalence": best.prevalence,
                "roc_auc": best.roc_auc,
                "pr_auc": best.pr_auc,
                "pr_lift": best.pr_auc_lift_over_prevalence,
                "top20_lift": best.top20_lift,
                "brier": best.brier,
            }
        )
    best_frame = pd.DataFrame(best_rows)
    future = predictive.set_index("prior")
    tail_future = future.loc["residual_lower_tail_graph", "edge_spearman"]
    best_non_tail = future.drop(index="residual_lower_tail_graph")["edge_spearman"].max()
    half_stability = similarity.set_index("comparison").loc[
        "tail_q10_first_half_vs_second_half", "edge_spearman"
    ]
    lead_stability = similarity.set_index("comparison").loc[
        "tail_q10_lead_days_1_3_vs_5_7", "edge_spearman"
    ]

    lines = [
        "# 24场站事件可预测性与残差下尾空间图诊断",
        "",
        "## 1. 审计口径",
        "",
        (
            f"- 模型与阈值仅使用训练集 {metadata['train_issue_count']} 个发布日期拟合；"
            + ("另做固定验证集外推检查；" if metadata["validation_loaded"] else "验证集未读取；")
            + "测试集未读取。"
        ),
        "- 事件标签由未来真实值定义，但输入特征仅来自当前168 h发布预测、上一版预测修订、发布前近期误差和日历信息。",
        f"- 采用 {metadata['fold_count']} 个扩展时间折，每折训练与验证之间隔离 {EMBARGO_ISSUES} 个发布日期，防止168 h目标窗口重叠。",
        "- 诊断模型为带L2约束的线性Logistic模型；它用于判断是否存在基础可预测信号，不代表最终风险编码器上限。",
        "",
        "## 2. 事件可预测性",
        "",
        "| 事件 | 标签 | 最佳可用特征 | 事件率 | ROC-AUC | PR-AUC | PR提升倍数 | Top20%提升 | Brier |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_frame.iterrows():
        lines.append(
            f"| {row.event} | train q{row.q} | {row.model} | {row.prevalence:.3f} | "
            f"{row.roc_auc:.3f} | {row.pr_auc:.3f} | {row.pr_lift:.2f}× | "
            f"{row.top20_lift:.2f}× | {row.brier:.3f} |"
        )
    lines.extend(
        [
            "",
            "PR提升倍数等于 PR-AUC/事件率；1 表示不优于按基础发生率随机排序。Top20%提升表示风险分数最高20%的发布日期中，事件率相对总体事件率的倍数。",
            "",
            "完整结果见 `predictability_metrics.csv`，逐发布日期滚动预测见 `issue_predictions.csv`。",
            "",
            "由于相邻发布日期的168 h目标窗口重叠，正例发布日期不能当作独立自然事件。保守合并后的事件过程数见下表。",
            "",
            "| 来源 | 事件 | 标签 | 正例发布日期 | 保守非重叠事件过程 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in episodes.iterrows():
        lines.append(
            f"| {row.source} | {row.event} | q{int(row.label_quantile*100)} | "
            f"{int(row.positive_issue_count)} | {int(row.conservative_nonoverlap_episode_count)} |"
        )
    lines.extend(
        [
            "",
            "按最严重6 h物理发生时刻进一步合并的事件清单见 `independent_event_catalog.csv`，",
            "发布日期到独立事件的映射见 `event_issue_mapping.csv`，训练集q90代表事件的完整24×168三元组见 `train_q90_event_prototypes.npz`。",
        ]
    )
    if validation_metrics is not None:
        lines.extend(
            [
                "",
                "### 固定验证集外推检查",
                "",
                "该表的阈值和Logistic参数均只在290个训练发布日期拟合，然后直接应用于23个11月验证发布日期。验证集不参与拟合；测试集仍未读取。",
                "",
                "| 事件 | 标签 | 模型 | 正例数 | ROC-AUC | PR-AUC | PR提升倍数 | Brier |",
                "|---|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in validation_metrics.iterrows():
            lines.append(
                f"| {row.event} | q{int(row.label_quantile*100)} | {row.model} | "
                f"{int(row.positive_count)} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | "
                f"{row.pr_auc_lift_over_prevalence:.2f}× | {row.brier:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 3. 下尾空间图",
            "",
            f"- 下尾图前后半段边权 Spearman：**{half_stability:.3f}**。",
            f"- 第1～3提前日与第5～7提前日边权 Spearman：**{lead_stability:.3f}**。",
            f"- 训练前段下尾图对后段下尾共现图的边权 Spearman：**{tail_future:.3f}**。",
            f"- 地理图、实际功率相关图和普通残差相关图中最好的对应值：**{best_non_tail:.3f}**。",
            "",
            "| 先验图 | 与未来下尾共现的Spearman | RMSE | Top20边重合率 |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in predictive.sort_values("edge_spearman", ascending=False).iterrows():
        lines.append(
            f"| {row.prior} | {row.edge_spearman:.3f} | {row.edge_rmse:.3f} | {row.top20_edge_overlap:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 判定规则",
            "",
            "- 若稀有事件的滚动 PR-AUC 明显高于事件率、Top20%风险组有稳定提升，则发布时可用信息支持学习风险先验；否则不能宣称可精确定位未来极端事件。",
            "- 若下尾图跨时间稳定，并且比地理图和普通相关图更能解释后段下尾共现，则值得作为尾部路径先验；否则不应为了创新而增加尾部图。",
            "- 这些结果只证明输入信息和空间先验的可行性，不证明复杂扩散模型必然提升。",
            "",
            "## 5. 高权重尾部边（训练全集）",
            "",
            "| 场站1 | 场站2 | 距离km | 实际相关 | 残差相关 | 下尾q10 | 后段下尾 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in edges.head(15).iterrows():
        lines.append(
            f"| {row.station_i_name} | {row.station_j_name} | {row.distance_km:.1f} | "
            f"{row.actual_power_correlation:.3f} | {row.ordinary_residual_correlation:.3f} | "
            f"{row.residual_lower_tail_q10:.3f} | {row.future_tail_score:.3f} |"
        )
    (output / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = Path(args.data_path)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    (output / "figures").mkdir()

    # Deliberately load train only.
    forecast = np.load(data / "train_forecast.npy").astype(np.float64)
    actual = np.load(data / "train_actual.npy").astype(np.float64)
    residual = np.load(data / "train_residual.npy").astype(np.float64)
    fill_mask = np.load(data / "train_fill_mask.npy")
    issues = pd.read_csv(data / "train_issue_dates.csv")
    stations = pd.read_csv(data / "station_order.csv")
    capacities = stations["capacity_mw"].to_numpy(dtype=np.float64)
    adjacency = np.load(data / "station_adjacency.npy").astype(np.float64)
    distance = np.load(data / "station_distance.npy").astype(np.float64)
    if forecast.shape != (len(issues), HOURS, 24):
        raise ValueError(f"unexpected train shape {forecast.shape}")
    if np.max(np.abs(residual - (actual - forecast))) > 1e-6:
        raise ValueError("residual is not actual - forecast")

    wind_residual_for_threshold = np.where(
        fill_mask[:, :, :WIND_COUNT] == 0,
        residual[:, :, :WIND_COUNT],
        np.nan,
    )
    station_lower_tail_threshold = np.nanquantile(
        wind_residual_for_threshold, 0.10, axis=(0, 1)
    )
    features, feature_names, feature_groups = build_available_features(
        forecast, residual, issues, capacities
    )
    severity = build_issue_severity(
        forecast,
        actual,
        fill_mask,
        capacities,
        issues,
        station_lower_tail_threshold,
    )
    catalog_thresholds = {
        quantile: float(np.quantile(severity["deep_drop_6h_severity"], quantile))
        for quantile in (0.80, 0.90)
    }
    train_catalog, train_mapping = build_independent_event_catalog(
        severity,
        issues,
        "train",
        catalog_thresholds,
    )
    folds = expanding_folds(
        len(issues), args.first_validation_index, args.folds
    )
    predictions, metrics, thresholds = run_predictability(
        features, feature_groups, severity, issues, folds, args.seed
    )
    similarity, graph_predictive, graphs, edges = run_graph_audit(
        residual, actual, fill_mask, issues, adjacency, distance, stations
    )
    validation_predictions = None
    validation_metrics = None
    episode_frames = [episode_summary(predictions, "train_rolling_oof")]
    if args.validation_check:
        validation_forecast = np.load(data / "val_forecast.npy").astype(np.float64)
        validation_actual = np.load(data / "val_actual.npy").astype(np.float64)
        validation_residual = np.load(data / "val_residual.npy").astype(np.float64)
        validation_fill_mask = np.load(data / "val_fill_mask.npy")
        validation_issues = pd.read_csv(data / "val_issue_dates.csv")
        validation_features, validation_names, validation_groups = build_available_features(
            validation_forecast, validation_residual, validation_issues, capacities
        )
        if validation_names != feature_names or validation_groups != feature_groups:
            raise ValueError("train and validation feature schema mismatch")
        validation_severity = build_issue_severity(
            validation_forecast,
            validation_actual,
            validation_fill_mask,
            capacities,
            validation_issues,
            station_lower_tail_threshold,
        )
        validation_catalog, validation_mapping = build_independent_event_catalog(
            validation_severity,
            validation_issues,
            "validation_read_only",
            catalog_thresholds,
        )
        validation_predictions, validation_metrics = run_external_validation(
            features,
            validation_features,
            feature_groups,
            severity,
            validation_severity,
            validation_issues,
            args.seed,
        )
        validation_predictions.to_csv(
            output / "validation_issue_predictions.csv", index=False
        )
        validation_metrics.to_csv(
            output / "validation_predictability_metrics.csv", index=False
        )
        validation_severity.assign(issue_date=validation_issues["issue_date"]).to_csv(
            output / "validation_event_severity.csv", index=False
        )
        episode_frames.append(
            episode_summary(validation_predictions, "fixed_validation")
        )
        train_catalog = pd.concat(
            [train_catalog, validation_catalog], ignore_index=True
        )
        train_mapping = pd.concat(
            [train_mapping, validation_mapping], ignore_index=True
        )
    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes.to_csv(output / "event_episode_summary.csv", index=False)
    train_catalog.to_csv(output / "independent_event_catalog.csv", index=False)
    train_mapping.to_csv(output / "event_issue_mapping.csv", index=False)
    prototype_rows = train_catalog[
        (train_catalog["source"] == "train")
        & np.isclose(train_catalog["label_quantile"], 0.90)
    ]
    prototype_indices = prototype_rows[
        "representative_sample_index"
    ].to_numpy(dtype=np.int64)
    np.savez_compressed(
        output / "train_q90_event_prototypes.npz",
        sample_index=prototype_indices,
        event_id=prototype_rows["event_id"].to_numpy(dtype=str),
        forecast=forecast[prototype_indices].astype(np.float32),
        actual=actual[prototype_indices].astype(np.float32),
        residual=residual[prototype_indices].astype(np.float32),
        fill_mask=fill_mask[prototype_indices],
    )
    pd.DataFrame(
        {
            "channel_index": np.arange(WIND_COUNT),
            "station_id": stations.iloc[:WIND_COUNT]["station_id"].to_numpy(),
            "residual_lower_q10_threshold": station_lower_tail_threshold,
        }
    ).to_csv(output / "train_wind_lower_tail_thresholds.csv", index=False)

    pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names)),
            "feature_name": feature_names,
            "forecast_only": [i in feature_groups["forecast_only"] for i in range(len(feature_names))],
            "forecast_revision": [i in feature_groups["forecast_revision"] for i in range(len(feature_names))],
            "all_available": [i in feature_groups["all_available"] for i in range(len(feature_names))],
            "generation_time_known": True,
        }
    ).to_csv(output / "feature_audit.csv", index=False)
    severity.assign(issue_date=issues["issue_date"]).to_csv(
        output / "event_severity.csv", index=False
    )
    predictions.to_csv(output / "issue_predictions.csv", index=False)
    metrics.to_csv(output / "predictability_metrics.csv", index=False)
    thresholds.to_csv(output / "fold_thresholds.csv", index=False)
    similarity.to_csv(output / "tail_graph_stability.csv", index=False)
    graph_predictive.to_csv(output / "tail_graph_future_comparison.csv", index=False)
    edges.to_csv(output / "tail_graph_edges.csv", index=False)
    np.savez_compressed(output / "spatial_graphs.npz", **graphs)
    plot_predictability(metrics, output / "figures" / "event_predictability.png")
    plot_graphs(graphs, output / "figures" / "spatial_prior_comparison.png")

    metadata = {
        "method": "station24_train_only_event_predictability_tail_graph_v1",
        "train_issue_count": int(len(issues)),
        "train_issue_start": str(issues.iloc[0]["issue_date"]),
        "train_issue_end": str(issues.iloc[-1]["issue_date"]),
        "validation_loaded": bool(args.validation_check),
        "validation_used_for_model_fit": False,
        "test_loaded": False,
        "future_actual_used_as_model_input": False,
        "future_actual_used_for_training_label_only": True,
        "fold_count": len(folds),
        "fold_embargo_issue_days": EMBARGO_ISSUES,
        "feature_groups": {key: len(value) for key, value in feature_groups.items()},
        "tail_quantiles": [0.10, 0.05],
        "event_catalog_quantiles": [0.80, 0.90],
        "event_catalog_thresholds": {
            str(quantile): value for quantile, value in catalog_thresholds.items()
        },
        "event_catalog_merge_gap_hours": 24,
        "train_q90_independent_event_count": int(
            np.sum(
                (train_catalog["source"] == "train")
                & np.isclose(train_catalog["label_quantile"], 0.90)
            )
        ),
        "wind_station_count": WIND_COUNT,
        "random_seed": args.seed,
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output,
        metrics,
        similarity,
        graph_predictive,
        edges,
        metadata,
        episodes,
        validation_metrics,
    )
    print(f"DIAGNOSTIC_COMPLETE output={output}")


if __name__ == "__main__":
    main()
