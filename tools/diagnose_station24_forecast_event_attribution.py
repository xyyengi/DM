"""Attribute wind-event timing failures to forecast anchoring or generation.

The diagnostic is validation-only and post-hoc: it reuses an existing paired
event table and generated scenarios, and never trains a model.  For every
actual extreme-ramp event it compares the event time in the issued forecast
with the event-time distribution of generated members.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CATEGORY_LABELS = {
    "A_condition_anchor": "A forecast late / scenarios follow",
    "B_model_delay": "B forecast on-time / scenarios late",
    "C_forecast_omission": "C forecast misses / scenarios do not recover",
    "D_low_probability_mass": "D forecast has event / too few members",
    "E_other_or_recovered": "E other or recovered",
}
CATEGORY_ORDER = list(CATEGORY_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only forecast/event timing attribution."
    )
    parser.add_argument("result_dirs", nargs=2)
    parser.add_argument("--event-records", required=True)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forecast-strength-ratio", type=float, default=0.50)
    parser.add_argument("--low-mass-rate", type=float, default=0.05)
    parser.add_argument("--on-time-hours", type=int, default=1)
    parser.add_argument("--alignment-hours", type=int, default=2)
    parser.add_argument("--search-radius-hours", type=int, default=6)
    return parser.parse_args()


def _load_results(paths: Iterable[str | Path]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    signature = None
    reference_actual = None
    reference_forecast = None
    for raw_path in paths:
        path = Path(raw_path)
        metadata = json.loads(
            (path / "generation_metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("split") != "val" or bool(metadata.get("test_used")):
            raise ValueError("diagnostic is restricted to sealed validation data")
        variant = str(metadata["condition_variant"])
        if variant in results:
            raise ValueError(f"duplicate condition variant: {variant}")
        current_signature = (
            int(metadata["n_samples"]),
            int(metadata["generation_seed"]),
            str(metadata["physical_projection"]),
        )
        if signature is None:
            signature = current_signature
        elif current_signature != signature:
            raise ValueError("paired generation protocols do not match")
        actual = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
        forecast = np.load(path / "forecast_data_normalized.npy", mmap_mode="r")
        scenarios = np.load(path / "actual_scenarios_normalized.npy", mmap_mode="r")
        if scenarios.shape[:1] + scenarios.shape[2:] != actual.shape:
            raise ValueError(f"scenario/actual shape mismatch in {path}")
        if forecast.shape != actual.shape:
            raise ValueError(f"forecast/actual shape mismatch in {path}")
        if reference_actual is None:
            reference_actual = np.asarray(actual)
            reference_forecast = np.asarray(forecast)
        elif not np.array_equal(reference_actual, actual):
            raise ValueError("paired runs do not share identical observations")
        elif not np.array_equal(reference_forecast, forecast):
            raise ValueError("paired runs do not share identical forecasts")
        results[variant] = {
            "path": path,
            "metadata": metadata,
            "actual": actual,
            "forecast": forecast,
            "scenarios": scenarios,
        }
    return results


def _scope_arrays(
    result: dict, stations: pd.DataFrame, scope: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wind = stations.loc[stations.data_type.eq("wind")].sort_values("channel_index")
    indices = wind.channel_index.to_numpy(int)
    actual = np.asarray(result["actual"][:, :, indices], dtype=np.float64)
    forecast = np.asarray(result["forecast"][:, :, indices], dtype=np.float64)
    scenarios = np.asarray(result["scenarios"][:, :, :, indices], dtype=np.float64)
    if scope == "station_pu":
        return actual, forecast, scenarios
    if scope == "aggregate_mw":
        capacity = wind.capacity_mw.to_numpy(float)
        return (
            np.sum(actual * capacity[None, None, :], axis=-1, keepdims=True),
            np.sum(forecast * capacity[None, None, :], axis=-1, keepdims=True),
            np.sum(scenarios * capacity[None, None, None, :], axis=-1, keepdims=True),
        )
    raise ValueError(f"unknown scope: {scope}")


def _nearest_directional_event(
    ramp: np.ndarray,
    event_index: int,
    minimum_magnitude: float,
    direction: str,
    search_radius: int,
) -> tuple[float, float]:
    """Return signed offset and ramp for the nearest qualifying event."""

    left = max(0, int(event_index) - int(search_radius))
    right = min(ramp.shape[0] - 1, int(event_index) + int(search_radius))
    values = np.asarray(ramp[left : right + 1], dtype=float)
    offsets = np.arange(left, right + 1, dtype=int) - int(event_index)
    magnitude = values if direction == "up" else -values
    positions = np.flatnonzero(magnitude >= float(minimum_magnitude))
    if not len(positions):
        return np.nan, np.nan
    candidate_offsets = offsets[positions]
    candidate_values = values[positions]
    order = np.lexsort((-np.abs(candidate_values), np.abs(candidate_offsets)))
    chosen = int(positions[order[0]])
    return float(offsets[chosen]), float(values[chosen])


def _strongest_directional_event(
    ramp: np.ndarray,
    event_index: int,
    direction: str,
    search_radius: int,
) -> tuple[float, float]:
    """Return the strongest same-direction forecast change in the search window."""
    left = max(0, int(event_index) - int(search_radius))
    right = min(ramp.shape[0] - 1, int(event_index) + int(search_radius))
    values = np.asarray(ramp[left : right + 1], dtype=float)
    magnitude = values if direction == "up" else -values
    chosen = int(np.argmax(magnitude))
    if magnitude[chosen] <= 0:
        return np.nan, np.nan
    return float(left + chosen - int(event_index)), float(values[chosen])


def _classify_event(
    forecast_present: bool,
    forecast_offset: float,
    model_offset: float,
    member_hit_rate_3h: float,
    member_hit_rate_6h: float,
    low_mass_rate: float = 0.05,
    on_time_hours: int = 1,
    alignment_hours: int = 2,
) -> str:
    """Assign one primary category; E retains mixed and successfully recovered cases."""

    if not forecast_present and member_hit_rate_6h < low_mass_rate:
        return "C_forecast_omission"
    if forecast_present and np.isfinite(forecast_offset) and np.isfinite(model_offset):
        if (
            forecast_offset > on_time_hours
            and model_offset > on_time_hours
            and abs(model_offset - forecast_offset) <= alignment_hours
        ):
            return "A_condition_anchor"
        if abs(forecast_offset) <= on_time_hours and model_offset > on_time_hours:
            return "B_model_delay"
    if forecast_present and member_hit_rate_3h < low_mass_rate:
        return "D_low_probability_mass"
    return "E_other_or_recovered"


def _augment_events(
    events: pd.DataFrame,
    results: dict[str, dict],
    stations: pd.DataFrame,
    forecast_strength_ratio: float,
    low_mass_rate: float,
    on_time_hours: int,
    alignment_hours: int,
    search_radius: int,
) -> pd.DataFrame:
    arrays = {
        model: {
            scope: _scope_arrays(result, stations, scope)
            for scope in ("station_pu", "aggregate_mw")
        }
        for model, result in results.items()
    }
    model_order = list(results)
    station_channels = (
        stations.loc[stations.data_type.eq("wind")]
        .sort_values("channel_index")
        .channel_index.to_list()
    )
    channel_to_local = {channel: index for index, channel in enumerate(station_channels)}
    rows = []
    for record in events.to_dict("records"):
        model = str(record["model"])
        if model not in arrays:
            raise ValueError(f"event table contains unknown model: {model}")
        scope = str(record["scope"])
        issue = int(record["issue_index"])
        lag = int(record["lag_hours"])
        lead_index = int(record["event_lead_hour"]) - 1
        ramp_index = lead_index - lag
        node_index = (
            channel_to_local[int(record["station_channel_index"])]
            if scope == "station_pu"
            else 0
        )
        _, forecast, _ = arrays[model][scope]
        forecast_ramp = (
            forecast[issue, lag:, node_index] - forecast[issue, :-lag, node_index]
        )
        actual_magnitude = abs(float(record["actual_ramp"]))
        relaxed_offset, relaxed_ramp = _nearest_directional_event(
            forecast_ramp,
            ramp_index,
            forecast_strength_ratio * actual_magnitude,
            str(record["direction"]),
            search_radius,
        )
        strict_offset, strict_ramp = _nearest_directional_event(
            forecast_ramp,
            ramp_index,
            float(record["absolute_ramp_threshold"]),
            str(record["direction"]),
            search_radius,
        )
        peak_offset, peak_ramp = _strongest_directional_event(
            forecast_ramp,
            ramp_index,
            str(record["direction"]),
            search_radius,
        )
        model_offset = float(record["median_timing_offset_hours"])
        relaxed_present = bool(np.isfinite(relaxed_offset))
        category = _classify_event(
            relaxed_present,
            relaxed_offset,
            model_offset,
            float(record["member_hit_rate_3h"]),
            float(record["member_hit_rate_6h"]),
            low_mass_rate,
            on_time_hours,
            alignment_hours,
        )
        record.update(
            {
                "lead_day": int(lead_index // 24 + 1),
                "forecast_strength_ratio_threshold": float(forecast_strength_ratio),
                "forecast_event_present_relaxed": relaxed_present,
                "forecast_event_present_strict": bool(np.isfinite(strict_offset)),
                "forecast_event_offset_hours": relaxed_offset,
                "forecast_strict_offset_hours": strict_offset,
                "forecast_matched_ramp": relaxed_ramp,
                "forecast_strict_matched_ramp": strict_ramp,
                "forecast_peak_offset_hours": peak_offset,
                "forecast_peak_ramp": peak_ramp,
                "forecast_peak_strength_ratio": (
                    abs(peak_ramp) / actual_magnitude
                    if np.isfinite(peak_ramp) and actual_magnitude > 0
                    else 0.0
                ),
                "forecast_matched_strength_ratio": (
                    abs(relaxed_ramp) / actual_magnitude
                    if relaxed_present and actual_magnitude > 0
                    else 0.0
                ),
                "model_minus_forecast_offset_hours": (
                    model_offset - relaxed_offset
                    if relaxed_present and np.isfinite(model_offset)
                    else np.nan
                ),
                "low_probability_mass_threshold": float(low_mass_rate),
                "attribution_category": category,
                "attribution_label": CATEGORY_LABELS[category],
            }
        )
        rows.append(record)
    augmented = pd.DataFrame(rows)
    if set(augmented.model.unique()) != set(model_order):
        raise ValueError("paired event table does not contain both result variants")
    return augmented


def _category_summary(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "model_label", "scope", "lag_hours", "direction"]
    rows = []
    for group_keys, group in events.groupby(keys, dropna=False):
        base = dict(zip(keys, group_keys))
        base["event_count"] = int(len(group))
        base["forecast_omission_rate"] = float(
            (~group.forecast_event_present_relaxed).mean()
        )
        base["forecast_strict_event_rate"] = float(
            group.forecast_event_present_strict.mean()
        )
        base["mean_member_hit_rate_3h"] = float(group.member_hit_rate_3h.mean())
        base["mean_member_hit_rate_6h"] = float(group.member_hit_rate_6h.mean())
        for category in CATEGORY_ORDER:
            base[f"share_{category}"] = float(
                group.attribution_category.eq(category).mean()
            )
        rows.append(base)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _lead_day_summary(events: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "model_label", "scope", "lag_hours", "direction", "lead_day"]
    rows = []
    for group_keys, group in events.groupby(keys, dropna=False):
        row = dict(zip(keys, group_keys))
        row.update(
            {
                "event_count": int(len(group)),
                "forecast_omission_rate": float(
                    (~group.forecast_event_present_relaxed).mean()
                ),
                "mean_member_hit_rate_3h": float(group.member_hit_rate_3h.mean()),
                "late_forecast_rate": float(
                    (group.forecast_event_offset_hours > 1).mean()
                ),
                "late_model_rate": float(
                    (group.median_timing_offset_hours > 1).mean()
                ),
            }
        )
        for category in CATEGORY_ORDER[:4]:
            row[f"share_{category}"] = float(
                group.attribution_category.eq(category).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _offset_alignment_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "model_label", "scope", "lag_hours", "direction"]
    present = events[
        events.forecast_event_present_relaxed
        & events.forecast_event_offset_hours.notna()
        & events.median_timing_offset_hours.notna()
    ]
    for group_keys, group in present.groupby(keys, dropna=False):
        forecast_offset = group.forecast_event_offset_hours.to_numpy(float)
        model_offset = group.median_timing_offset_hours.to_numpy(float)
        correlation = (
            float(np.corrcoef(forecast_offset, model_offset)[0, 1])
            if len(group) >= 3
            and np.std(forecast_offset) > 0
            and np.std(model_offset) > 0
            else np.nan
        )
        row = dict(zip(keys, group_keys))
        row.update(
            {
                "event_count": int(len(group)),
                "forecast_model_offset_correlation": correlation,
                "mean_absolute_model_minus_forecast_offset_h": float(
                    np.mean(np.abs(model_offset - forecast_offset))
                ),
                "model_within_1h_of_forecast_rate": float(
                    np.mean(np.abs(model_offset - forecast_offset) <= 1)
                ),
                "model_within_2h_of_forecast_rate": float(
                    np.mean(np.abs(model_offset - forecast_offset) <= 2)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float64) / width
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), -1, values)


def _sustained_deep_drop_summary(
    results: dict[str, dict],
    stations: pd.DataFrame,
    issues: pd.DataFrame,
    count: int = 5,
    window: int = 6,
) -> tuple[pd.DataFrame, list[dict]]:
    first = next(iter(results.values()))
    _, forecast, _ = _scope_arrays(first, stations, "aggregate_mw")
    actual, _, _ = _scope_arrays(first, stations, "aggregate_mw")
    actual = actual[..., 0]
    forecast = forecast[..., 0]
    gap = _rolling_mean(forecast - actual, window)
    best_start = np.argmax(gap, axis=1)
    best_score = gap[np.arange(len(gap)), best_start]
    selected_issues = np.argsort(best_score)[::-1][: min(count, len(best_score))]
    definitions = []
    rows = []
    for rank, issue_index in enumerate(selected_issues, start=1):
        start = int(best_start[issue_index])
        stop = start + window
        actual_mean = float(actual[issue_index, start:stop].mean())
        forecast_mean = float(forecast[issue_index, start:stop].mean())
        definition = {
            "event_rank": rank,
            "issue_index": int(issue_index),
            "issue_date": str(issues.iloc[issue_index].issue_date),
            "lead_start": start,
            "lead_end": stop - 1,
            "actual_window_mean_mw": actual_mean,
            "forecast_window_mean_mw": forecast_mean,
            "forecast_minus_actual_mw": forecast_mean - actual_mean,
        }
        definitions.append(definition)
        for model, result in results.items():
            _, _, scenarios = _scope_arrays(result, stations, "aggregate_mw")
            member_mean = scenarios[issue_index, :, start:stop, 0].mean(axis=1)
            rows.append(
                {
                    "model": model,
                    **definition,
                    "member_count": int(len(member_mean)),
                    "minimum_member_mean_mw": float(member_mean.min()),
                    "median_member_mean_mw": float(np.median(member_mean)),
                    "members_at_or_below_actual": int(np.sum(member_mean <= actual_mean)),
                    "hit_rate_at_or_below_actual": float(np.mean(member_mean <= actual_mean)),
                    "lower_90_mw": float(np.quantile(member_mean, 0.05)),
                    "lower_95_mw": float(np.quantile(member_mean, 0.025)),
                    "lower_99_mw": float(np.quantile(member_mean, 0.005)),
                    "covered_90": bool(actual_mean >= np.quantile(member_mean, 0.05)),
                    "covered_95": bool(actual_mean >= np.quantile(member_mean, 0.025)),
                    "covered_99": bool(actual_mean >= np.quantile(member_mean, 0.005)),
                }
            )
    return pd.DataFrame(rows), definitions


def _plot_sustained_deep_drops(
    deep_drops: pd.DataFrame,
    definitions: list[dict],
    results: dict[str, dict],
    stations: pd.DataFrame,
    output: Path,
) -> None:
    model_order = list(results)
    arrays = {
        model: _scope_arrays(result, stations, "aggregate_mw")
        for model, result in results.items()
    }
    selected = definitions[: min(3, len(definitions))]
    fig, axes = plt.subplots(len(selected), 2, figsize=(15, 3.8 * len(selected)), squeeze=False)
    for row, event in enumerate(selected):
        issue = int(event["issue_index"])
        event_start = int(event["lead_start"])
        event_end = int(event["lead_end"])
        left, right = max(0, event_start - 18), min(167, event_end + 18)
        lead = np.arange(left + 1, right + 2)
        for column, model in enumerate(model_order):
            actual, forecast, scenarios = arrays[model]
            sample = scenarios[issue, :, left : right + 1, 0]
            member_window_mean = scenarios[issue, :, event_start : event_end + 1, 0].mean(axis=1)
            extreme_members = np.argsort(member_window_mean)[:3]
            axis = axes[row, column]
            axis.fill_between(
                lead, np.quantile(sample, 0.005, axis=0), np.quantile(sample, 0.995, axis=0),
                color="#f9a8d4", alpha=0.20, label="99% envelope",
            )
            axis.fill_between(
                lead, np.quantile(sample, 0.05, axis=0), np.quantile(sample, 0.95, axis=0),
                color="#fb7185", alpha=0.25, label="90% envelope",
            )
            for rank, member in enumerate(extreme_members):
                axis.plot(
                    lead, sample[member], color="#7e22ce", linewidth=0.8, alpha=0.55,
                    label="3 deepest members" if rank == 0 else None,
                )
            axis.plot(lead, np.median(sample, axis=0), color="#e11d48", label="median")
            axis.plot(lead, forecast[issue, left : right + 1, 0], color="#0d9488", linestyle="--", label="forecast")
            axis.plot(lead, actual[issue, left : right + 1, 0], color="#111827", linewidth=1.4, label="actual")
            axis.axvspan(event_start + 1, event_end + 1, color="#f59e0b", alpha=0.10)
            record = deep_drops[
                deep_drops.model.eq(model) & deep_drops.event_rank.eq(event["event_rank"])
            ].iloc[0]
            axis.set_title(
                f"{'Baseline' if column == 0 else 'Candidate'} | event {event['event_rank']} | "
                f"members at/below actual={int(record.members_at_or_below_actual)}/{int(record.member_count)}",
                fontsize=10,
            )
            axis.set_ylabel("Aggregated wind (MW)")
            axis.set_xlabel("Lead hour")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Five hardest forecast-missed sustained wind drops (top three shown)", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _sensitivity_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strength_ratio in (0.30, 0.50, 0.70):
        present = events.forecast_peak_strength_ratio.ge(strength_ratio)
        offset = events.forecast_peak_offset_hours
        for low_mass_rate in (0.01, 0.05, 0.10):
            categories = [
                _classify_event(
                    bool(p),
                    float(o) if np.isfinite(o) else np.nan,
                    float(m),
                    float(h3),
                    float(h6),
                    low_mass_rate,
                )
                for p, o, m, h3, h6 in zip(
                    present,
                    offset,
                    events.median_timing_offset_hours,
                    events.member_hit_rate_3h,
                    events.member_hit_rate_6h,
                    strict=True,
                )
            ]
            frame = events.assign(_category=categories)
            for (model, scope), group in frame.groupby(["model", "scope"]):
                row = {
                    "model": model,
                    "scope": scope,
                    "forecast_strength_ratio": strength_ratio,
                    "low_mass_rate": low_mass_rate,
                    "event_count": int(len(group)),
                }
                for category in CATEGORY_ORDER:
                    row[f"share_{category}"] = float(group._category.eq(category).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def _plot_category_shares(summary: pd.DataFrame, output: Path) -> None:
    models = list(summary.model.drop_duplicates())
    colors = ["#7c3aed", "#2563eb", "#dc2626", "#d97706", "#94a3b8"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for row, scope in enumerate(("station_pu", "aggregate_mw")):
        for column, lag in enumerate((1, 3, 6)):
            axis = axes[row, column]
            subset = summary[
                summary.scope.eq(scope)
                & summary.lag_hours.eq(lag)
                & summary.direction.eq("all")
            ].set_index("model").reindex(models)
            bottom = np.zeros(len(models))
            for category, color in zip(CATEGORY_ORDER, colors, strict=True):
                values = subset[f"share_{category}"].to_numpy(float)
                axis.bar(
                    np.arange(len(models)), values, bottom=bottom, color=color,
                    label=CATEGORY_LABELS[category],
                )
                bottom += values
            axis.set_xticks(np.arange(len(models)), ["Baseline", "Candidate"])
            axis.set_ylim(0, 1)
            axis.set_title(f"{scope.replace('_', ' ')} | {lag}h ramps")
            axis.grid(axis="y", alpha=0.2)
            if column == 0:
                axis.set_ylabel("Share of actual events")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Forecast-to-scenario event attribution", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lead_day(events: pd.DataFrame, candidate: str, output: Path) -> None:
    frame = events[events.model.eq(candidate)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    colors = {
        "A_condition_anchor": "#7c3aed",
        "C_forecast_omission": "#dc2626",
        "D_low_probability_mass": "#d97706",
    }
    for row, scope in enumerate(("station_pu", "aggregate_mw")):
        for column, direction in enumerate(("up", "down")):
            axis = axes[row, column]
            group = frame[frame.scope.eq(scope) & frame.direction.eq(direction)]
            for category, color in colors.items():
                values = (
                    group.assign(hit=group.attribution_category.eq(category).astype(float))
                    .groupby("lead_day").hit.mean().reindex(range(1, 8))
                )
                axis.plot(values.index, values.values, marker="o", color=color, label=CATEGORY_LABELS[category])
            axis.set_title(f"{scope.replace('_', ' ')} | {direction}")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel("Event share")
            if row == 1:
                axis.set_xlabel("Forecast lead day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Candidate attribution by forecast lead day", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_offset_alignment(events: pd.DataFrame, candidate: str, output: Path) -> None:
    frame = events[
        events.model.eq(candidate)
        & events.forecast_event_present_relaxed
        & events.median_timing_offset_hours.notna()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for axis, scope in zip(axes, ("station_pu", "aggregate_mw"), strict=True):
        subset = frame[frame.scope.eq(scope)]
        for direction, color in (("up", "#dc2626"), ("down", "#2563eb")):
            group = subset[subset.direction.eq(direction)]
            axis.scatter(
                group.forecast_event_offset_hours,
                group.median_timing_offset_hours,
                s=14,
                alpha=0.28,
                color=color,
                label=direction,
            )
        axis.plot([-6, 6], [-6, 6], color="#111827", linestyle="--", linewidth=1, label="scenario follows forecast")
        axis.axhline(0, color="#94a3b8", linewidth=1)
        axis.axvline(0, color="#94a3b8", linewidth=1)
        axis.set_xlim(-6.5, 6.5)
        axis.set_ylim(-6.5, 6.5)
        axis.set_title(scope.replace("_", " "))
        axis.set_xlabel("Forecast event time - actual time (h)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Median generated event time - actual time (h)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Does generated timing follow forecast timing?", y=1.02)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_events(
    events: pd.DataFrame,
    results: dict[str, dict],
    stations: pd.DataFrame,
    candidate: str,
    output: Path,
) -> list[str]:
    candidate_events = events[
        events.model.eq(candidate)
        & events.scope.eq("aggregate_mw")
        & events.lag_hours.eq(3)
    ].copy()
    candidate_events["severity"] = candidate_events.actual_ramp.abs() / candidate_events.absolute_ramp_threshold
    selected = []
    for category in ("A_condition_anchor", "C_forecast_omission", "D_low_probability_mass"):
        group = candidate_events[candidate_events.attribution_category.eq(category)]
        if len(group):
            selected.append(group.sort_values(["member_hit_rate_3h", "severity"], ascending=[True, False]).iloc[0])
    if not selected:
        return []
    model_order = list(results)
    arrays = {model: _scope_arrays(result, stations, "aggregate_mw") for model, result in results.items()}
    fig, axes = plt.subplots(len(selected), 2, figsize=(15, 3.8 * len(selected)), squeeze=False)
    selected_ids = []
    for row, event in enumerate(selected):
        selected_ids.append(str(event.event_uid))
        issue = int(event.issue_index)
        center = int(event.event_lead_hour) - 1
        left, right = max(0, center - 18), min(167, center + 18)
        lead = np.arange(left + 1, right + 2)
        for column, model in enumerate(model_order):
            actual, forecast, scenarios = arrays[model]
            sample = scenarios[issue, :, left : right + 1, 0]
            axis = axes[row, column]
            axis.fill_between(lead, np.quantile(sample, 0.05, axis=0), np.quantile(sample, 0.95, axis=0), color="#fb7185", alpha=0.25)
            axis.plot(lead, np.quantile(sample, 0.5, axis=0), color="#e11d48", label="median")
            axis.plot(lead, forecast[issue, left : right + 1, 0], color="#0d9488", linestyle="--", label="forecast")
            axis.plot(lead, actual[issue, left : right + 1, 0], color="#111827", linewidth=1.4, label="actual")
            axis.axvline(center + 1, color="#7c3aed", linestyle=":", linewidth=1.4, label="actual event")
            paired = events[events.event_uid.eq(event.event_uid) & events.model.eq(model)].iloc[0]
            if np.isfinite(paired.forecast_event_offset_hours):
                axis.axvline(center + 1 + paired.forecast_event_offset_hours, color="#0d9488", linestyle=":", linewidth=1)
            axis.set_title(f"{paired.model_label} | {CATEGORY_LABELS[event.attribution_category]} | hit ±3h={paired.member_hit_rate_3h:.1%}")
            axis.set_ylabel("Aggregated wind (MW)")
            axis.set_xlabel("Lead hour")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Representative attribution events", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected_ids


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    events: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    alignment: pd.DataFrame,
    deep_drops: pd.DataFrame,
    model_order: list[str],
    selected: list[str],
) -> None:
    core = summary[summary.direction.eq("all")][
        [
            "model_label", "scope", "lag_hours", "event_count",
            "forecast_omission_rate", "mean_member_hit_rate_3h",
            "share_A_condition_anchor", "share_B_model_delay",
            "share_C_forecast_omission", "share_D_low_probability_mass",
        ]
    ]
    candidate = model_order[1]
    candidate_events = events[events.model.eq(candidate)]
    candidate_label = str(candidate_events.model_label.iloc[0])
    sensitivity_core = sensitivity[
        sensitivity.model.eq(candidate)
        & sensitivity.scope.eq("aggregate_mw")
        & sensitivity.forecast_strength_ratio.eq(0.50)
    ][
        ["low_mass_rate", "event_count", "share_A_condition_anchor", "share_B_model_delay", "share_C_forecast_omission", "share_D_low_probability_mass"]
    ]
    deep_core = deep_drops[
        deep_drops.model.eq(candidate)
    ][
        [
            "event_rank", "issue_date", "lead_start", "lead_end",
            "actual_window_mean_mw", "forecast_window_mean_mw",
            "minimum_member_mean_mw", "members_at_or_below_actual",
            "hit_rate_at_or_below_actual", "covered_99",
        ]
    ]
    alignment_core = alignment[
        alignment.model.eq(candidate)
        & alignment.scope.eq("aggregate_mw")
    ][
        [
            "lag_hours", "direction", "event_count",
            "forecast_model_offset_correlation",
            "mean_absolute_model_minus_forecast_offset_h",
            "model_within_2h_of_forecast_rate",
        ]
    ]
    report = "# 风电极端爬坡时刻归因诊断（验证集、无需训练）\n\n"
    report += (
        "本诊断复用现有 23 个验证发布窗口和每窗口 500 个场景，没有重新训练，也没有使用测试集。"
        "实际极端事件沿用原诊断口径：按单站/13站聚合与 1/3/6 h 时距分别计算实测绝对爬坡的 90% 分位阈值，"
        "连续超阈值点只保留幅度最大的一个。\n\n"
    )
    report += "## A/B/C/D 的可执行定义\n\n"
    report += (
        "- **A 条件锚定**：发布预测中的同方向事件晚于真实事件超过 1 h，且生成成员的中位事件偏移与预测偏移相差不超过 2 h。\n"
        "- **B 模型时间解码**：预测事件在真实事件 ±1 h 内，但生成成员的中位事件仍晚于真实事件超过 1 h。\n"
        "- **C 预测遗漏且场景未恢复**：预测在 ±6 h 内没有达到真实事件幅度 50% 的同方向变化，场景在 ±6 h 内的严格事件命中率又低于 5%。\n"
        "- **D 概率质量不足**：预测已经包含事件，但 500 个成员中在真实时刻 ±3 h 内生成严格极端事件的比例低于 5%。\n"
        "- **E 其他或已恢复**：包括提前、混合偏移，以及预测遗漏但扩散模型成功恢复出事件的情况。E 不是失败类型。\n\n"
        "严格场景事件仍须达到历史 q90 极端阈值；预测是否‘看见’事件采用 50% 相对幅度，是为了避免把幅度稍弱但时刻明确的预测误判为完全遗漏。\n\n"
    )
    report += "## 核心结果\n\n" + _markdown_table(core) + "\n\n"
    report += (
        f"对 **{candidate_label}**，真实极端爬坡在发布预测中达不到真实幅度 50% 的比例很高；"
        "但其中不少普通 q90 事件仍被扩散成员部分恢复，因此不能把全部预测遗漏都记成 C。"
        "真正需要优先优化的是 C 类、D 类及下表持续深跌事件，而不是让所有场景的中位数追随每一个罕见事件。\n\n"
    )
    report += "## 预测偏移与生成偏移的一致性（候选模型、13站聚合）\n\n"
    report += _markdown_table(alignment_core) + "\n\n"
    report += (
        "相关系数和‘生成偏移落在预测偏移 ±2 h 内的比例’用于判断条件锚定；"
        "它们只在预测确实包含相应事件、且场景中存在严格极端事件时计算。\n\n"
    )
    report += "## 五个最困难的持续 6 h 深跌\n\n"
    report += _markdown_table(deep_core) + "\n\n"
    report += (
        "这五个窗口按验证集实测与发布预测的 6 h 平均差选择，不使用任何模型输出。"
        "`members_at_or_below_actual` 直接回答 500 条场景中有多少条达到真实低谷深度；"
        "它补充了爬坡时刻诊断无法判断的‘是否维持足够低’。\n\n"
    )
    report += "## 低概率阈值敏感性（候选模型、13站聚合、预测强度阈值50%）\n\n"
    report += _markdown_table(sensitivity_core) + "\n\n"
    report += "## 输出图\n\n"
    report += (
        "- `figures/category_share.png`：A/B/C/D/E 在基线与候选模型中的占比。\n"
        "- `figures/lead_day_attribution.png`：第 1—7 提前日中 A/C/D 的变化。\n"
        "- `figures/forecast_vs_model_offset.png`：预测事件偏移与生成事件偏移是否沿对角线同步。\n"
        "- `figures/representative_events.png`：A/C/D 的聚合风电典型曲线（若对应类别存在）。\n\n"
        "- `figures/sustained_deep_drop_examples.png`：最困难持续深跌的 90%/99% 包络与三条最深成员。\n\n"
    )
    report += "代表事件：`" + "`, `".join(selected) + "`。\n\n"
    report += "## 边界\n\n"
    report += (
        "同一自然事件可能出现在多个滚动发布窗口中，因此事件记录不是互相独立的自然事件计数。"
        "本诊断用于选择下一项训练实验；最终结论仍需在封存测试集上一次性确认。\n"
    )
    (output / "forecast_event_attribution.md").write_text(report, encoding="utf-8")


def run_diagnostic(
    result_dirs: Iterable[str | Path],
    event_records: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    forecast_strength_ratio: float = 0.50,
    low_mass_rate: float = 0.05,
    on_time_hours: int = 1,
    alignment_hours: int = 2,
    search_radius: int = 6,
) -> Path:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)
    results = _load_results(result_dirs)
    model_order = list(results)
    stations = pd.read_csv(Path(data_path) / "station_order.csv")
    events = pd.read_csv(event_records)
    expected_models = set(model_order)
    if set(events.model.unique()) != expected_models:
        raise ValueError(
            f"event table models {set(events.model.unique())} do not match {expected_models}"
        )
    augmented = _augment_events(
        events, results, stations, forecast_strength_ratio, low_mass_rate,
        on_time_hours, alignment_hours, search_radius,
    )
    # Add all-direction summaries without duplicating event records.
    summary_inputs = pd.concat(
        [augmented, augmented.assign(direction="all")], ignore_index=True
    )
    summary = _category_summary(summary_inputs)
    lead_day = _lead_day_summary(augmented)
    alignment = _offset_alignment_summary(augmented)
    sensitivity = _sensitivity_summary(augmented)
    issues = pd.read_csv(Path(data_path) / "val_issue_dates.csv").sort_values("sample_index")
    deep_drops, deep_drop_definitions = _sustained_deep_drop_summary(
        results, stations, issues
    )
    label_map = augmented.drop_duplicates("model").set_index("model").model_label
    deep_drops["model_label"] = deep_drops.model.map(label_map)
    augmented.to_csv(output / "attribution_event_records.csv", index=False)
    summary.to_csv(output / "attribution_summary.csv", index=False)
    lead_day.to_csv(output / "lead_day_summary.csv", index=False)
    alignment.to_csv(output / "offset_alignment_summary.csv", index=False)
    sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
    deep_drops.to_csv(output / "sustained_deep_drop_summary.csv", index=False)
    _plot_category_shares(summary, figures / "category_share.png")
    _plot_lead_day(augmented, model_order[1], figures / "lead_day_attribution.png")
    _plot_offset_alignment(augmented, model_order[1], figures / "forecast_vs_model_offset.png")
    selected = _plot_representative_events(
        augmented, results, stations, model_order[1], figures / "representative_events.png"
    )
    _plot_sustained_deep_drops(
        deep_drops, deep_drop_definitions, results, stations,
        figures / "sustained_deep_drop_examples.png",
    )
    metadata = {
        "purpose": "validation-only post-hoc forecast-to-scenario wind event attribution",
        "models": model_order,
        "result_directories": {model: str(results[model]["path"]) for model in model_order},
        "source_event_records": str(event_records),
        "split": "val",
        "test_used": False,
        "issue_count": int(next(iter(results.values()))["actual"].shape[0]),
        "member_count": int(next(iter(results.values()))["scenarios"].shape[1]),
        "forecast_strength_ratio": forecast_strength_ratio,
        "low_mass_rate": low_mass_rate,
        "on_time_hours": on_time_hours,
        "alignment_hours": alignment_hours,
        "search_radius_hours": search_radius,
        "selected_representative_event_uids": selected,
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(
        output, augmented, summary, sensitivity, alignment, deep_drops,
        model_order, selected,
    )
    return output


def main() -> None:
    args = parse_args()
    output = run_diagnostic(
        args.result_dirs,
        args.event_records,
        args.data_path,
        args.output_dir,
        args.forecast_strength_ratio,
        args.low_mass_rate,
        args.on_time_hours,
        args.alignment_hours,
        args.search_radius_hours,
    )
    print(f"FORECAST_EVENT_ATTRIBUTION_COMPLETE output={output}")


if __name__ == "__main__":
    main()
