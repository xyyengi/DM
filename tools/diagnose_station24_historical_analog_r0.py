#!/usr/bin/env python3
"""No-training historical analog retrieval diagnostic for Station24.

The script builds a train-only bank of forecast/residual issue windows and asks
whether issuance-time information can retrieve validation residual trajectories
that are more useful than equally-sized random histories.  It deliberately:

* loads only train and validation arrays (never test arrays);
* uses validation actual/residual only after retrieval, for evaluation;
* constructs forecast revision and recent error with the same causal alignment
  used by :class:`station_dataset.StationForecastDataset`;
* does not train or modify the diffusion model.

The first implementation is wind-focused because the unresolved failure mode is
the aggregated-wind peak/drop/recovery mismatch.  Full 13-station wind residual
trajectories are retained in the history bank, so cross-station structure is not
collapsed during retrieval evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOURS = 168
BLOCK_HOURS = 6
RAMP_LAGS = (1, 3, 6)
TARGET_ISSUES = (12, 13, 14, 21)
EPS = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--dtw-band", type=int, default=6)
    return parser.parse_args()


def load_split(data_path: Path, split: str) -> dict[str, object]:
    if split not in {"train", "val"}:
        raise ValueError("R0 may load train or val only")
    return {
        "forecast": np.load(data_path / f"{split}_forecast.npy").astype(
            np.float64
        ),
        "actual": np.load(data_path / f"{split}_actual.npy").astype(np.float64),
        "residual": np.load(data_path / f"{split}_residual.npy").astype(
            np.float64
        ),
        "fill_mask": np.load(data_path / f"{split}_fill_mask.npy"),
        "issues": pd.read_csv(data_path / f"{split}_issue_dates.csv"),
    }


def validate_split(data: dict[str, object], split: str) -> None:
    forecast = np.asarray(data["forecast"])
    expected = (len(forecast), HOURS, 24)
    for name in ("forecast", "actual", "residual", "fill_mask"):
        value = np.asarray(data[name])
        if value.shape != expected:
            raise ValueError(f"{split}_{name} expected {expected}, got {value.shape}")
    error = np.max(
        np.abs(
            np.asarray(data["residual"])
            - (np.asarray(data["actual"]) - np.asarray(data["forecast"]))
        )
    )
    if error > 1e-6:
        raise ValueError(f"{split} residual identity failed: {error}")
    issues = pd.DataFrame(data["issues"])
    if len(issues) != len(forecast):
        raise ValueError(f"{split} issue count does not match arrays")


def block_mean(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[1] % width:
        raise ValueError("time length must be divisible by block width")
    shape = (values.shape[0], values.shape[1] // width, width) + values.shape[2:]
    return values.reshape(shape).mean(axis=2)


def previous_indices(issues: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(issues["issue_date"]).dt.normalize()
    lookup = {date: index for index, date in enumerate(dates)}
    return np.asarray(
        [lookup.get(date - pd.Timedelta(days=1), -1) for date in dates],
        dtype=np.int64,
    )


def causal_arrays(
    forecast: np.ndarray,
    residual: np.ndarray,
    issues: pd.DataFrame,
    station_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build only information available at the issue time.

    Revision aligns current leads 1..144 with the preceding issue's leads
    25..168.  Recent error uses the preceding issue's first 24 target hours,
    which have elapsed by the current midnight issuance.
    """

    n = len(forecast)
    current = forecast[:, :, station_indices]
    revision = np.zeros((n, 144, len(station_indices)), dtype=np.float64)
    recent = np.zeros((n, 24, len(station_indices)), dtype=np.float64)
    revision_available = np.zeros(n, dtype=bool)
    recent_available = np.zeros(n, dtype=bool)
    previous = previous_indices(issues)
    for index, previous_index in enumerate(previous):
        if previous_index < 0:
            continue
        revision[index] = (
            current[index, :144] - current[previous_index, 24:]
        )
        recent[index] = residual[previous_index, :24][:, station_indices]
        revision_available[index] = True
        recent_available[index] = True
    return {
        "forecast": current,
        "revision": revision,
        "recent_error": recent,
        "revision_available": revision_available,
        "recent_error_available": recent_available,
        "previous_index": previous,
    }


def feature_blocks(
    causal: dict[str, np.ndarray],
    issues: pd.DataFrame,
    capacities: np.ndarray,
) -> dict[str, np.ndarray]:
    forecast = causal["forecast"]
    revision = causal["revision"]
    recent = causal["recent_error"]
    weights = capacities / capacities.sum()

    forecast_6h = block_mean(forecast, BLOCK_HOURS)
    aggregate = np.einsum("nts,s->nt", forecast, weights)
    ramp_parts = []
    for lag in RAMP_LAGS:
        ramp = np.zeros_like(aggregate)
        ramp[:, lag:] = aggregate[:, lag:] - aggregate[:, :-lag]
        ramp_parts.append(block_mean(ramp[:, :, None], BLOCK_HOURS)[:, :, 0])
    daily = forecast.reshape(len(forecast), 7, 24, len(capacities))
    daily_mean = daily.mean(axis=2)
    daily_std = daily.std(axis=2)
    dates = pd.to_datetime(issues["issue_date"])
    calendar = np.stack(
        [
            np.sin(2 * np.pi * (dates.dt.month.to_numpy() - 1) / 12.0),
            np.cos(2 * np.pi * (dates.dt.month.to_numpy() - 1) / 12.0),
            np.sin(2 * np.pi * dates.dt.dayofweek.to_numpy() / 7.0),
            np.cos(2 * np.pi * dates.dt.dayofweek.to_numpy() / 7.0),
        ],
        axis=1,
    )
    forecast_features = np.concatenate(
        [
            forecast_6h.reshape(len(forecast), -1),
            *ramp_parts,
            daily_mean.reshape(len(forecast), -1),
            daily_std.reshape(len(forecast), -1),
            calendar,
        ],
        axis=1,
    )

    revision_6h = block_mean(revision, BLOCK_HOURS)
    revision_features = np.concatenate(
        [
            revision_6h.reshape(len(revision), -1),
            revision.mean(axis=1),
            revision.std(axis=1),
        ],
        axis=1,
    )
    recent_3h = block_mean(recent, 3)
    recent_features = np.concatenate(
        [
            recent_3h.reshape(len(recent), -1),
            recent.mean(axis=1),
            recent.std(axis=1),
        ],
        axis=1,
    )
    return {
        "forecast": forecast_features,
        "revision": revision_features,
        "recent_error": recent_features,
    }


def fit_standardization(
    train_blocks: dict[str, np.ndarray], bank_indices: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fitted = {}
    for name, values in train_blocks.items():
        subset = values[bank_indices]
        mean = subset.mean(axis=0)
        std = subset.std(axis=0)
        std[std < 1e-6] = 1.0
        fitted[name] = (mean, std)
    return fitted


def standardize_blocks(
    blocks: dict[str, np.ndarray],
    fitted: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        name: np.nan_to_num((values - fitted[name][0]) / fitted[name][1])
        for name, values in blocks.items()
    }


def retrieve(
    bank_blocks: dict[str, np.ndarray],
    query_blocks: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    query_causal: dict[str, np.ndarray],
    variant: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested = {
        "forecast_only": ("forecast",),
        "forecast_revision": ("forecast", "revision"),
        "forecast_revision_recent": (
            "forecast",
            "revision",
            "recent_error",
        ),
    }[variant]
    all_indices = []
    all_weights = []
    all_distances = []
    for query_index in range(len(query_blocks["forecast"])):
        active = list(requested)
        if not bool(query_causal["revision_available"][query_index]):
            active = [name for name in active if name != "revision"]
        if not bool(query_causal["recent_error_available"][query_index]):
            active = [name for name in active if name != "recent_error"]
        component_distances = []
        for name in active:
            difference = (
                bank_blocks[name][bank_indices]
                - query_blocks[name][query_index][None]
            )
            component_distances.append(np.mean(difference * difference, axis=1))
        distance = np.mean(np.stack(component_distances, axis=0), axis=0)
        local = np.argsort(distance, kind="stable")[:top_k]
        chosen_distance = distance[local]
        temperature = max(float(np.median(chosen_distance)), 1e-6)
        logits = -(chosen_distance - chosen_distance.min()) / temperature
        weight = np.exp(logits)
        weight /= weight.sum()
        all_indices.append(bank_indices[local])
        all_weights.append(weight)
        all_distances.append(chosen_distance)
    return (
        np.asarray(all_indices, dtype=np.int64),
        np.asarray(all_weights, dtype=np.float64),
        np.asarray(all_distances, dtype=np.float64),
    )


def constrained_dtw(left: np.ndarray, right: np.ndarray, band: int) -> float:
    n, m = len(left), len(right)
    width = max(int(band), abs(n - m))
    previous = np.full(m + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full(m + 1, np.inf, dtype=np.float64)
        lower = max(1, i - width)
        upper = min(m, i + width)
        for j in range(lower, upper + 1):
            cost = (left[i - 1] - right[j - 1]) ** 2
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(math.sqrt(previous[m] / max(n + m, 1)))


def any_window_support(
    candidate: np.ndarray,
    target: np.ndarray,
    event_mask: np.ndarray,
    tolerance: int,
    mode: str,
) -> tuple[int, int]:
    hit = 0
    total = int(event_mask.sum())
    for hour in np.flatnonzero(event_mask):
        left = max(0, int(hour) - tolerance)
        right = min(candidate.shape[-1], int(hour) + tolerance + 1)
        values = candidate[:, left:right]
        if mode == "negative":
            supported = np.any(values <= target[hour])
        elif mode == "positive":
            supported = np.any(values >= target[hour])
        else:
            raise ValueError(mode)
        hit += int(supported)
    return hit, total


def selection_metrics(
    selected_indices: np.ndarray,
    selected_weights: np.ndarray,
    bank_residual_mw: np.ndarray,
    validation_residual_mw: np.ndarray,
    validation_forecast_mw: np.ndarray,
    validation_actual_mw: np.ndarray,
    train_residual_lower: float,
    train_actual_ramp_thresholds: dict[int, float],
    total_capacity: float,
    dtw_band: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    candidates = bank_residual_mw[selected_indices]
    prototype = np.einsum("qk,qkt->qt", selected_weights, candidates)
    target = validation_residual_mw
    member_rmse = np.sqrt(np.mean((candidates - target[:, None]) ** 2, axis=2))
    prototype_rmse = np.sqrt(np.mean((prototype - target) ** 2, axis=1))
    best_member_rmse = member_rmse.min(axis=1)
    prototype_correlation = []
    prototype_dtw = []
    negative_hit = []
    negative_count = []
    ramp_accuracy: dict[int, list[float]] = {lag: [] for lag in RAMP_LAGS}
    missed_hit: dict[int, list[float]] = {lag: [] for lag in RAMP_LAGS}
    missed_count: dict[int, list[int]] = {lag: [] for lag in RAMP_LAGS}
    per_issue_rows = []

    candidate_scenarios = np.clip(
        validation_forecast_mw[:, None, :] + candidates,
        0.0,
        total_capacity,
    )
    for issue in range(len(target)):
        correlation = np.corrcoef(target[issue], prototype[issue])[0, 1]
        prototype_correlation.append(float(np.nan_to_num(correlation)))
        prototype_dtw.append(
            constrained_dtw(
                target[issue] / total_capacity,
                prototype[issue] / total_capacity,
                dtw_band,
            )
        )
        negative_events = target[issue] <= train_residual_lower
        hits, count = any_window_support(
            candidates[issue], target[issue], negative_events, 3, "negative"
        )
        negative_hit.append(float(hits / count) if count else float("nan"))
        negative_count.append(count)

        row = {
            "issue_index": issue,
            "prototype_rmse_mw": float(prototype_rmse[issue]),
            "best_member_rmse_mw": float(best_member_rmse[issue]),
            "mean_member_rmse_mw": float(member_rmse[issue].mean()),
            "prototype_dtw_pu": float(prototype_dtw[-1]),
            "prototype_correlation": float(prototype_correlation[-1]),
            "negative_tail_event_hours": count,
            "negative_tail_any_hit_rate_pm3h": negative_hit[-1],
        }
        for lag in RAMP_LAGS:
            target_residual_ramp = target[issue, lag:] - target[issue, :-lag]
            candidate_residual_ramp = (
                candidates[issue, :, lag:] - candidates[issue, :, :-lag]
            )
            event = np.abs(target_residual_ramp) >= np.quantile(
                np.abs(target_residual_ramp), 0.90
            )
            if np.any(event):
                agreement = (
                    np.sign(candidate_residual_ramp[:, event])
                    == np.sign(target_residual_ramp[event])[None]
                ).mean()
            else:
                agreement = float("nan")
            ramp_accuracy[lag].append(float(agreement))

            actual_ramp = (
                validation_actual_mw[issue, lag:]
                - validation_actual_mw[issue, :-lag]
            )
            forecast_ramp = (
                validation_forecast_mw[issue, lag:]
                - validation_forecast_mw[issue, :-lag]
            )
            missed = (
                np.abs(actual_ramp) >= train_actual_ramp_thresholds[lag]
            ) & (
                (np.sign(actual_ramp) != np.sign(forecast_ramp))
                | (np.abs(forecast_ramp) < 0.5 * np.abs(actual_ramp))
            )
            hits = 0
            for local_hour in np.flatnonzero(missed):
                event_hour = int(local_hour) + lag
                left = max(lag, event_hour - 3)
                right = min(HOURS, event_hour + 4)
                scenario_ramps = (
                    candidate_scenarios[issue, :, left:right]
                    - candidate_scenarios[issue, :, left - lag : right - lag]
                )
                required = 0.5 * abs(actual_ramp[local_hour])
                support = (
                    np.sign(scenario_ramps) == np.sign(actual_ramp[local_hour])
                ) & (np.abs(scenario_ramps) >= required)
                hits += int(np.any(support))
            rate = float(hits / missed.sum()) if np.any(missed) else float("nan")
            missed_hit[lag].append(rate)
            missed_count[lag].append(int(missed.sum()))
            row[f"residual_ramp_direction_accuracy_{lag}h"] = float(agreement)
            row[f"forecast_missed_ramp_count_{lag}h"] = int(missed.sum())
            row[f"forecast_missed_ramp_any_hit_rate_{lag}h_pm3h"] = rate
        per_issue_rows.append(row)

    aggregate = {
        "prototype_rmse_mw": float(np.mean(prototype_rmse)),
        "best_member_rmse_mw": float(np.mean(best_member_rmse)),
        "mean_member_rmse_mw": float(np.mean(member_rmse)),
        "prototype_dtw_pu": float(np.mean(prototype_dtw)),
        "prototype_correlation": float(np.mean(prototype_correlation)),
        "negative_tail_any_hit_rate_pm3h": float(np.nanmean(negative_hit)),
        "negative_tail_event_hour_count": int(np.sum(negative_count)),
    }
    for lag in RAMP_LAGS:
        aggregate[f"residual_ramp_direction_accuracy_{lag}h"] = float(
            np.nanmean(ramp_accuracy[lag])
        )
        aggregate[f"forecast_missed_ramp_any_hit_rate_{lag}h_pm3h"] = float(
            np.nanmean(missed_hit[lag])
        )
        aggregate[f"forecast_missed_ramp_event_count_{lag}h"] = int(
            np.sum(missed_count[lag])
        )
    return aggregate, pd.DataFrame(per_issue_rows)


def random_selections(
    rng: np.random.Generator, repeats: int, queries: int, bank: np.ndarray, top_k: int
) -> Iterable[np.ndarray]:
    for _ in range(repeats):
        yield np.stack(
            [rng.choice(bank, size=top_k, replace=False) for _ in range(queries)]
        )


def metric_direction(name: str) -> str:
    if "rmse" in name or "dtw" in name:
        return "lower"
    return "higher"


def comparison_rows(
    variant: str,
    analog: dict[str, float],
    random_metrics: list[dict[str, float]],
) -> list[dict[str, object]]:
    rows = []
    keys = [
        key
        for key, value in analog.items()
        if isinstance(value, float) and "count" not in key
    ]
    for key in keys:
        random_values = np.asarray([row[key] for row in random_metrics], dtype=float)
        direction = metric_direction(key)
        analog_value = float(analog[key])
        random_mean = float(np.nanmean(random_values))
        if direction == "lower":
            gain = (random_mean - analog_value) / max(abs(random_mean), EPS)
            p_value = float(
                (1 + np.sum(random_values <= analog_value))
                / (len(random_values) + 1)
            )
        else:
            gain = (analog_value - random_mean) / max(abs(random_mean), EPS)
            p_value = float(
                (1 + np.sum(random_values >= analog_value))
                / (len(random_values) + 1)
            )
        rows.append(
            {
                "variant": variant,
                "metric": key,
                "direction": direction,
                "analog": analog_value,
                "random_mean": random_mean,
                "random_std": float(np.nanstd(random_values, ddof=1)),
                "relative_gain": gain,
                "empirical_one_sided_p": p_value,
                "random_q05": float(np.nanquantile(random_values, 0.05)),
                "random_q95": float(np.nanquantile(random_values, 0.95)),
            }
        )
    return rows


def plot_metric_comparison(frame: pd.DataFrame, output: Path) -> None:
    metrics = [
        "prototype_rmse_mw",
        "best_member_rmse_mw",
        "prototype_dtw_pu",
        "negative_tail_any_hit_rate_pm3h",
        "forecast_missed_ramp_any_hit_rate_3h_pm3h",
        "forecast_missed_ramp_any_hit_rate_6h_pm3h",
    ]
    labels = {
        "forecast_only": "Forecast",
        "forecast_revision": "Forecast + revision",
        "forecast_revision_recent": "Forecast + revision + recent error",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, metric in zip(axes.flat, metrics):
        subset = frame[frame.metric == metric].set_index("variant")
        variants = list(labels)
        x = np.arange(len(variants))
        analog = [subset.loc[v, "analog"] for v in variants]
        random = [subset.loc[v, "random_mean"] for v in variants]
        random_std = [subset.loc[v, "random_std"] for v in variants]
        axis.bar(x - 0.18, analog, width=0.36, label="retrieved", color="#d81b60")
        axis.bar(
            x + 0.18,
            random,
            width=0.36,
            yerr=random_std,
            label="random mean +/- sd",
            color="#8c96a0",
            alpha=0.8,
        )
        axis.set_xticks(x, [labels[v] for v in variants], rotation=18, ha="right")
        axis.set_title(metric.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("R0 historical retrieval vs equally sized random histories", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_issues(
    output: Path,
    issue_indices: list[int],
    issues: pd.DataFrame,
    validation_forecast_mw: np.ndarray,
    validation_actual_mw: np.ndarray,
    validation_residual_mw: np.ndarray,
    bank_residual_mw: np.ndarray,
    analog_indices: np.ndarray,
    analog_weights: np.ndarray,
    random_indices: np.ndarray,
) -> None:
    fig, axes = plt.subplots(len(issue_indices), 2, figsize=(16, 4 * len(issue_indices)))
    if len(issue_indices) == 1:
        axes = np.asarray([axes])
    lead = np.arange(HOURS)
    for row, issue in enumerate(issue_indices):
        candidates = bank_residual_mw[analog_indices[issue]]
        prototype = np.sum(candidates * analog_weights[issue, :, None], axis=0)
        random_mean = bank_residual_mw[random_indices[issue]].mean(axis=0)
        scenario_candidates = np.clip(
            validation_forecast_mw[issue][None] + candidates,
            0.0,
            np.inf,
        )
        axis = axes[row, 0]
        axis.plot(lead, validation_actual_mw[issue], color="#111827", lw=1.8, label="actual")
        axis.plot(
            lead,
            validation_forecast_mw[issue],
            color="#009688",
            lw=1.5,
            ls="--",
            label="issued forecast",
        )
        for member in scenario_candidates[: min(10, len(scenario_candidates))]:
            axis.plot(lead, member, color="#ef6c8f", alpha=0.15, lw=0.8)
        axis.plot(
            lead,
            validation_forecast_mw[issue] + prototype,
            color="#d81b60",
            lw=1.8,
            label="retrieved residual prototype",
        )
        axis.set_title(
            f"Validation issue {issue} ({issues.iloc[issue].issue_date}): power"
        )
        axis.set_ylabel("Aggregated wind MW")
        axis.grid(alpha=0.25)
        if row == 0:
            axis.legend(frameon=False, ncol=3)

        axis = axes[row, 1]
        for member in candidates:
            axis.plot(lead, member, color="#ef6c8f", alpha=0.13, lw=0.8)
        axis.plot(
            lead,
            validation_residual_mw[issue],
            color="#111827",
            lw=1.8,
            label="true residual",
        )
        axis.plot(lead, prototype, color="#d81b60", lw=1.8, label="retrieved prototype")
        axis.plot(lead, random_mean, color="#6b7280", lw=1.3, ls="--", label="one random mean")
        axis.axhline(0.0, color="#111827", lw=0.6, alpha=0.5)
        axis.set_title("Retrieved historical residual trajectories")
        axis.set_ylabel("Aggregated residual MW")
        axis.grid(alpha=0.25)
        if row == 0:
            axis.legend(frameon=False, ncol=3)
    for axis in axes[-1]:
        axis.set_xlabel("Lead hour")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def target_issue_audit(
    issue_indices: list[int],
    validation_issues: pd.DataFrame,
    train_issues: pd.DataFrame,
    validation_forecast_mw: np.ndarray,
    validation_actual_mw: np.ndarray,
    validation_residual_mw: np.ndarray,
    bank_residual_mw: np.ndarray,
    analog_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for issue in issue_indices:
        hour = int(np.argmin(validation_residual_mw[issue]))
        selected = analog_indices[issue]
        candidate_residual = bank_residual_mw[selected]
        scenario_exact = np.clip(
            validation_forecast_mw[issue, hour] + candidate_residual[:, hour],
            0.0,
            np.inf,
        )
        exact_hit = scenario_exact <= validation_actual_mw[issue, hour]
        left, right = max(0, hour - 3), min(HOURS, hour + 4)
        pm3_hit = (
            candidate_residual[:, left:right].min(axis=1)
            <= validation_residual_mw[issue, hour]
        )
        rows.append(
            {
                "validation_issue_index": issue,
                "validation_issue_date": str(
                    validation_issues.iloc[issue].issue_date
                ),
                "deepest_negative_residual_lead_hour": hour,
                "actual_mw": float(validation_actual_mw[issue, hour]),
                "forecast_mw": float(validation_forecast_mw[issue, hour]),
                "true_residual_mw": float(validation_residual_mw[issue, hour]),
                "retrieved_member_count": int(len(selected)),
                "minimum_retrieved_scenario_exact_hour_mw": float(
                    scenario_exact.min()
                ),
                "members_at_or_below_actual_exact_hour": int(exact_hit.sum()),
                "members_with_residual_as_deep_within_pm3h": int(pm3_hit.sum()),
                "exact_hit_train_issue_indices": ";".join(
                    str(int(index)) for index in selected[exact_hit]
                ),
                "exact_hit_train_issue_dates": ";".join(
                    str(train_issues.iloc[int(index)].issue_date)
                    for index in selected[exact_hit]
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    result: dict[str, object],
    comparison: pd.DataFrame,
    best_variant: str,
    decision: str,
) -> None:
    key_metrics = [
        "prototype_rmse_mw",
        "best_member_rmse_mw",
        "prototype_dtw_pu",
        "negative_tail_any_hit_rate_pm3h",
        "forecast_missed_ramp_any_hit_rate_1h_pm3h",
        "forecast_missed_ramp_any_hit_rate_3h_pm3h",
        "forecast_missed_ramp_any_hit_rate_6h_pm3h",
    ]
    best = comparison[comparison.variant == best_variant].set_index("metric")
    rows = []
    for metric in key_metrics:
        row = best.loc[metric]
        rows.append(
            f"| {metric} | {row.analog:.6g} | {row.random_mean:.6g} | "
            f"{100 * row.relative_gain:+.2f}% | {row.empirical_one_sided_p:.4f} |"
        )
    target_rows = []
    for item in result["target_issue_audit"]:
        target_rows.append(
            f"| {item['validation_issue_index']} | {item['validation_issue_date']} | "
            f"{item['deepest_negative_residual_lead_hour']} | "
            f"{item['actual_mw']:.1f} | {item['forecast_mw']:.1f} | "
            f"{item['minimum_retrieved_scenario_exact_hour_mw']:.1f} | "
            f"{item['members_at_or_below_actual_exact_hour']}/{item['retrieved_member_count']} |"
        )
    report = f"""# R0：24场站历史类比检索无训练诊断

> 生成时间：{result['created_at']}  
> 数据边界：仅训练集建立历史库，验证集23个发布窗口只作查询与评价；测试集未读取。  
> 检索规模：Top-{result['top_k']}；随机对照重复{result['random_repeats']}次。

## 1. 结论先行

**判定：{decision}**

本诊断不是扩散训练实验。它检验的是：只凭生成时已经知道的168 h发布预测、上一版预测修订和最近24 h已观测误差，能否从290个训练发布窗口中找到比随机历史更接近验证真实残差的完整历史轨迹。

综合关键指标后，本次最优检索键为 `{best_variant}`。正增益表示优于同样Top-K规模的随机历史；经验单侧p值来自重复随机抽样，而不是神经网络种子。

| 指标 | 检索 | 随机均值 | 相对增益 | 经验p值 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 2. 数据与防泄漏口径

- 历史库目标：训练集完整的13场站风电、168 h真实残差轨迹；含填补值的训练发布窗口被排除。
- 当前预测形态：13站6 h块均值、逐日均值/标准差和聚合1/3/6 h爬坡。
- 上一版修订：当前lead 1–144 h减去前一日发布的lead 25–168 h。
- 最近误差：前一日发布样本已经走完的前24 h残差。
- 验证集未来actual/residual只在邻居选定后用于评分，未进入距离计算。
- `test_*` 文件未读取。

## 3. 三组消融

1. `forecast_only`：只按当前发布预测形态检索；
2. `forecast_revision`：加入相邻发布版修订；
3. `forecast_revision_recent`：再加入最近24 h已观测误差。

每类特征均只用训练历史拟合均值和标准差；不同特征块分别计算标准化均方距离后等权组合，避免高维块仅凭维数支配距离。

![检索与随机对比](figures/retrieval_vs_random.png)

## 4. 问题窗口核查

下图固定检查验证Issue 12、13、14、21。左列是在当前发布预测上叠加检索历史残差后的候选轨迹；右列直接展示真实残差、Top-K历史残差和其距离加权原型。

![问题窗口历史检索](figures/target_issue_analogs.png)

| Issue | 日期 | 最深负残差小时 | 实际MW | 预测MW | 检索成员最小MW | 同小时达到真实深度 |
|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(target_rows)}

这里要特别区分“原型均值”和“尾部支持”：检索历史的距离加权均值仍会平滑少数深跌，但只要Top-K中存在同小时达到真实深度的少数轨迹，就证明历史实例可作为尾部候选。它不应被平均后强行加到全部场景中心。

完整邻居日期、距离和指标见：

- `retrieved_neighbors.csv`
- `per_issue_metrics.csv`
- `metric_comparison.csv`
- `random_repeat_metrics.csv`

## 5. 下一步规则

- 若判定为“通过R0”：先做R1非参数Analog Ensemble，不接神经网络；验证检索到的完整残差重采样能否真正提高错向/深跌成员命中，同时检查CRPS、覆盖宽度和空间相关性。
- 若判定为“未通过R0”：停止把当前检索键输入尾部网络。下一步应先扩大可识别信息或重做事件分层，而不是增加Transformer、动态图或历史编码器。
- 无论判定如何，测试集仍保持封存。
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_k < 2:
        raise ValueError("top-k must be at least two")
    data_path = Path(args.data_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"outputs_shandong/station24/historical_analog_r0_{timestamp}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True)

    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy()
    capacities = stations.loc[wind_indices, "capacity_mw"].to_numpy(dtype=np.float64)
    if len(wind_indices) != 13:
        raise ValueError(f"expected 13 wind stations, got {len(wind_indices)}")
    total_capacity = float(capacities.sum())

    train = load_split(data_path, "train")
    validation = load_split(data_path, "val")
    validate_split(train, "train")
    validate_split(validation, "val")
    train_causal = causal_arrays(
        np.asarray(train["forecast"]),
        np.asarray(train["residual"]),
        pd.DataFrame(train["issues"]),
        wind_indices,
    )
    validation_causal = causal_arrays(
        np.asarray(validation["forecast"]),
        np.asarray(validation["residual"]),
        pd.DataFrame(validation["issues"]),
        wind_indices,
    )
    train_blocks = feature_blocks(
        train_causal,
        pd.DataFrame(train["issues"]),
        capacities,
    )
    validation_blocks = feature_blocks(
        validation_causal,
        pd.DataFrame(validation["issues"]),
        capacities,
    )

    train_fill = np.asarray(train["fill_mask"])[:, :, wind_indices]
    bank_indices = np.flatnonzero(~np.any(train_fill != 0, axis=(1, 2)))
    if len(bank_indices) < args.top_k:
        raise ValueError("too few complete training histories for requested top-k")
    fitted = fit_standardization(train_blocks, bank_indices)
    train_standard = standardize_blocks(train_blocks, fitted)
    validation_standard = standardize_blocks(validation_blocks, fitted)

    train_residual = np.asarray(train["residual"])[:, :, wind_indices]
    validation_residual = np.asarray(validation["residual"])[:, :, wind_indices]
    train_forecast = np.asarray(train["forecast"])[:, :, wind_indices]
    train_actual = np.asarray(train["actual"])[:, :, wind_indices]
    validation_forecast = np.asarray(validation["forecast"])[:, :, wind_indices]
    validation_actual = np.asarray(validation["actual"])[:, :, wind_indices]
    train_residual_mw = np.einsum("nts,s->nt", train_residual, capacities)
    validation_residual_mw = np.einsum(
        "nts,s->nt", validation_residual, capacities
    )
    train_forecast_mw = np.einsum("nts,s->nt", train_forecast, capacities)
    train_actual_mw = np.einsum("nts,s->nt", train_actual, capacities)
    validation_forecast_mw = np.einsum(
        "nts,s->nt", validation_forecast, capacities
    )
    validation_actual_mw = np.einsum(
        "nts,s->nt", validation_actual, capacities
    )
    train_residual_lower = float(
        np.quantile(train_residual_mw[bank_indices], 0.05)
    )
    train_actual_ramp_thresholds = {
        lag: float(
            np.quantile(
                np.abs(
                    train_actual_mw[bank_indices, lag:]
                    - train_actual_mw[bank_indices, :-lag]
                ),
                0.90,
            )
        )
        for lag in RAMP_LAGS
    }

    variants = (
        "forecast_only",
        "forecast_revision",
        "forecast_revision_recent",
    )
    retrieved: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    analog_metrics: dict[str, dict[str, float]] = {}
    per_issue_frames = []
    neighbor_rows = []
    rng = np.random.default_rng(args.seed)
    random_index_sets = list(
        random_selections(
            rng,
            args.random_repeats,
            len(validation_forecast),
            bank_indices,
            args.top_k,
        )
    )
    uniform_weights = np.full(
        (len(validation_forecast), args.top_k), 1.0 / args.top_k
    )
    random_metric_rows = []
    # Random history performance is independent of retrieval feature variant.
    random_metric_objects = []
    random_per_issue_accumulator = []
    for repeat, indices in enumerate(random_index_sets):
        metrics, issue_frame = selection_metrics(
            indices,
            uniform_weights,
            train_residual_mw,
            validation_residual_mw,
            validation_forecast_mw,
            validation_actual_mw,
            train_residual_lower,
            train_actual_ramp_thresholds,
            total_capacity,
            args.dtw_band,
        )
        random_metric_objects.append(metrics)
        random_per_issue_accumulator.append(issue_frame)
        random_metric_rows.append({"repeat": repeat, **metrics})

    comparison = []
    for variant in variants:
        indices, weights, distances = retrieve(
            train_standard,
            validation_standard,
            bank_indices,
            validation_causal,
            variant,
            args.top_k,
        )
        retrieved[variant] = (indices, weights, distances)
        metrics, issue_frame = selection_metrics(
            indices,
            weights,
            train_residual_mw,
            validation_residual_mw,
            validation_forecast_mw,
            validation_actual_mw,
            train_residual_lower,
            train_actual_ramp_thresholds,
            total_capacity,
            args.dtw_band,
        )
        analog_metrics[variant] = metrics
        issue_frame.insert(0, "variant", variant)
        issue_frame["issue_date"] = pd.DataFrame(validation["issues"])[
            "issue_date"
        ].astype(str)
        per_issue_frames.append(issue_frame)
        comparison.extend(comparison_rows(variant, metrics, random_metric_objects))
        for issue in range(len(indices)):
            for rank, train_index in enumerate(indices[issue], start=1):
                neighbor_rows.append(
                    {
                        "variant": variant,
                        "validation_issue_index": issue,
                        "validation_issue_date": pd.DataFrame(validation["issues"])
                        .iloc[issue]
                        .issue_date,
                        "rank": rank,
                        "train_issue_index": int(train_index),
                        "train_issue_date": pd.DataFrame(train["issues"])
                        .iloc[int(train_index)]
                        .issue_date,
                        "distance": float(distances[issue, rank - 1]),
                        "weight": float(weights[issue, rank - 1]),
                    }
                )

    comparison_frame = pd.DataFrame(comparison)
    random_frame = pd.DataFrame(random_metric_rows)
    per_issue_frame = pd.concat(per_issue_frames, ignore_index=True)
    neighbor_frame = pd.DataFrame(neighbor_rows)
    comparison_frame.to_csv(output_dir / "metric_comparison.csv", index=False)
    random_frame.to_csv(output_dir / "random_repeat_metrics.csv", index=False)
    per_issue_frame.to_csv(output_dir / "per_issue_metrics.csv", index=False)
    neighbor_frame.to_csv(output_dir / "retrieved_neighbors.csv", index=False)

    score_metrics = [
        "prototype_rmse_mw",
        "best_member_rmse_mw",
        "prototype_dtw_pu",
        "negative_tail_any_hit_rate_pm3h",
        "forecast_missed_ramp_any_hit_rate_3h_pm3h",
        "forecast_missed_ramp_any_hit_rate_6h_pm3h",
    ]
    score = (
        comparison_frame[comparison_frame.metric.isin(score_metrics)]
        .groupby("variant")["relative_gain"]
        .mean()
    )
    best_variant = str(score.idxmax())
    best_compare = comparison_frame[comparison_frame.variant == best_variant].set_index(
        "metric"
    )
    trajectory_pass = any(
        best_compare.loc[name, "relative_gain"] >= 0.02
        and best_compare.loc[name, "empirical_one_sided_p"] <= 0.10
        for name in (
            "prototype_rmse_mw",
            "best_member_rmse_mw",
            "prototype_dtw_pu",
        )
    )
    event_pass = any(
        best_compare.loc[name, "relative_gain"] > 0.0
        and best_compare.loc[name, "empirical_one_sided_p"] <= 0.10
        for name in (
            "negative_tail_any_hit_rate_pm3h",
            "forecast_missed_ramp_any_hit_rate_1h_pm3h",
            "forecast_missed_ramp_any_hit_rate_3h_pm3h",
            "forecast_missed_ramp_any_hit_rate_6h_pm3h",
        )
    )
    decision = (
        "通过R0：历史类比同时提供轨迹与事件增益，可进入R1非参数基线。"
        if trajectory_pass and event_pass
        else "未通过R0：当前检索键未同时证明轨迹与事件增益，暂不接入神经网络。"
    )

    plot_metric_comparison(
        comparison_frame, figure_dir / "retrieval_vs_random.png"
    )
    target_issues = [i for i in TARGET_ISSUES if i < len(validation_forecast)]
    best_indices, best_weights, _ = retrieved[best_variant]
    plot_target_issues(
        figure_dir / "target_issue_analogs.png",
        target_issues,
        pd.DataFrame(validation["issues"]),
        validation_forecast_mw,
        validation_actual_mw,
        validation_residual_mw,
        train_residual_mw,
        best_indices,
        best_weights,
        random_index_sets[0],
    )
    target_audit = target_issue_audit(
        target_issues,
        pd.DataFrame(validation["issues"]),
        pd.DataFrame(train["issues"]),
        validation_forecast_mw,
        validation_actual_mw,
        validation_residual_mw,
        train_residual_mw,
        best_indices,
    )
    target_audit.to_csv(output_dir / "target_issue_event_audit.csv", index=False)

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "station24_historical_analog_r0_no_training",
        "scope": "aggregated_wind_retrieval_with_13_station_joint_residual_bank",
        "data_path": str(data_path),
        "train_issue_count": int(len(train_forecast)),
        "complete_train_bank_count": int(len(bank_indices)),
        "excluded_filled_train_issue_count": int(
            len(train_forecast) - len(bank_indices)
        ),
        "validation_issue_count": int(len(validation_forecast)),
        "top_k": int(args.top_k),
        "random_repeats": int(args.random_repeats),
        "seed": int(args.seed),
        "dtw_band_hours": int(args.dtw_band),
        "feature_dimensions": {
            name: int(values.shape[1]) for name, values in train_blocks.items()
        },
        "validation_previous_issue_available_count": int(
            validation_causal["recent_error_available"].sum()
        ),
        "train_residual_lower_q05_mw": train_residual_lower,
        "train_actual_ramp_q90_mw": train_actual_ramp_thresholds,
        "best_variant": best_variant,
        "decision": decision,
        "decision_components": {
            "trajectory_pass": bool(trajectory_pass),
            "event_pass": bool(event_pass),
        },
        "test_files_loaded": False,
        "future_validation_actual_used_for_retrieval": False,
        "validation_actual_use": "evaluation_after_neighbor_selection_only",
        "analog_metrics": analog_metrics,
        "target_issue_audit": target_audit.to_dict(orient="records"),
    }
    (output_dir / "r0_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "historical_analog_r0_report.md",
        result,
        comparison_frame,
        best_variant,
        decision,
    )
    print(f"R0_COMPLETE output={output_dir}")
    print(f"BEST_VARIANT={best_variant}")
    print(f"DECISION={decision}")


if __name__ == "__main__":
    main()
