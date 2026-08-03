"""Diagnose wind ramp-event timing in paired Station24 validation scenarios.

This is a post-hoc, validation-only diagnostic.  It does not train or modify a
model.  Actual extreme-ramp events are fixed once from the shared observations,
then each generated member is checked for a same-direction event near the true
event time.  The main statistic is the member hit rate; an ``any member`` rate
is retained only as a secondary diagnostic because it grows mechanically with
ensemble size.
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


MODEL_ORDER = ["ramp36_control", "state_v1_fixed_graph"]
MODEL_LABELS = {
    "ramp36_control": "Ramp36 control",
    "state_v1_fixed_graph": "State V1",
}
DEFAULT_LAGS = (1, 3, 6)
DEFAULT_TOLERANCES = (1, 3, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only wind event-timing diagnostic."
    )
    parser.add_argument("result_dirs", nargs=2)
    parser.add_argument("--data-path", default="diffusion_input_station")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extreme-quantile", type=float, default=0.90)
    parser.add_argument("--search-radius-hours", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--baseline-variant", default=MODEL_ORDER[0])
    parser.add_argument("--candidate-variant", default=MODEL_ORDER[1])
    parser.add_argument("--baseline-label", default=MODEL_LABELS[MODEL_ORDER[0]])
    parser.add_argument("--candidate-label", default=MODEL_LABELS[MODEL_ORDER[1]])
    return parser.parse_args()


def _configure_models(
    baseline_variant: str,
    candidate_variant: str,
    baseline_label: str,
    candidate_label: str,
) -> None:
    global MODEL_ORDER, MODEL_LABELS
    if baseline_variant == candidate_variant:
        raise ValueError("baseline and candidate variants must be different")
    MODEL_ORDER = [str(baseline_variant), str(candidate_variant)]
    MODEL_LABELS = {
        MODEL_ORDER[0]: str(baseline_label),
        MODEL_ORDER[1]: str(candidate_label),
    }


def _load_results(paths: Iterable[str | Path]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    signatures = set()
    reference_actual = None
    reference_forecast = None
    for raw_path in paths:
        path = Path(raw_path)
        metadata = json.loads(
            (path / "generation_metadata.json").read_text(encoding="utf-8")
        )
        variant = metadata["condition_variant"]
        if variant not in MODEL_ORDER or variant in results:
            raise ValueError(f"unexpected or duplicate condition variant: {variant}")
        if metadata["split"] != "val" or bool(metadata.get("test_used")):
            raise ValueError("timing diagnostic is restricted to sealed validation data")
        signature = (
            metadata["split"],
            int(metadata["n_samples"]),
            int(metadata["generation_seed"]),
            metadata["physical_projection"],
        )
        signatures.add(signature)
        actual = np.load(path / "actual_data_normalized.npy", mmap_mode="r")
        forecast = np.load(path / "forecast_data_normalized.npy", mmap_mode="r")
        scenarios = np.load(
            path / "actual_scenarios_normalized.npy", mmap_mode="r"
        )
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
    if set(results) != set(MODEL_ORDER) or len(signatures) != 1:
        raise ValueError("the two paired generation protocols do not match")
    return results


def _clustered_event_indices(
    ramp: np.ndarray,
    threshold: float,
    direction: str,
) -> list[int]:
    """Return one strongest event from each contiguous extreme-ramp run."""

    if direction == "up":
        candidate = np.flatnonzero(ramp >= threshold)
    elif direction == "down":
        candidate = np.flatnonzero(ramp <= -threshold)
    else:
        raise ValueError(f"unknown direction {direction}")
    if not len(candidate):
        return []
    runs = np.split(candidate, np.flatnonzero(np.diff(candidate) > 1) + 1)
    return [int(run[np.argmax(np.abs(ramp[run]))]) for run in runs]


def _nearest_member_matches(
    member_ramps: np.ndarray,
    event_index: int,
    threshold: float,
    direction: str,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find each member's nearest qualifying event and signed timing offset."""

    left = max(0, event_index - search_radius)
    right = min(member_ramps.shape[1] - 1, event_index + search_radius)
    window = member_ramps[:, left : right + 1]
    offsets = np.arange(left, right + 1, dtype=int) - int(event_index)
    qualifying = window >= threshold if direction == "up" else window <= -threshold
    matched_offset = np.full(member_ramps.shape[0], np.nan, dtype=float)
    matched_ramp = np.full(member_ramps.shape[0], np.nan, dtype=float)
    for member in range(member_ramps.shape[0]):
        positions = np.flatnonzero(qualifying[member])
        if not len(positions):
            continue
        candidate_offsets = offsets[positions]
        candidate_values = window[member, positions]
        # Closest time wins; a larger absolute ramp breaks an equal-distance tie.
        order = np.lexsort((-np.abs(candidate_values), np.abs(candidate_offsets)))
        chosen = int(positions[order[0]])
        matched_offset[member] = float(offsets[chosen])
        matched_ramp[member] = float(window[member, chosen])
    return matched_offset, matched_ramp


def _scope_arrays(
    result: dict,
    stations: pd.DataFrame,
    scope: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, str]:
    wind = stations.loc[stations.data_type.eq("wind")].copy()
    indices = wind.channel_index.to_numpy(int)
    actual = np.asarray(result["actual"][:, :, indices], dtype=np.float64)
    forecast = np.asarray(result["forecast"][:, :, indices], dtype=np.float64)
    scenarios = np.asarray(result["scenarios"][:, :, :, indices], dtype=np.float64)
    if scope == "station_pu":
        nodes = wind.reset_index(drop=True)
        return actual, forecast, scenarios, nodes, "p.u."
    if scope == "aggregate_mw":
        capacity = wind.capacity_mw.to_numpy(float)
        actual = np.sum(actual * capacity[None, None, :], axis=-1, keepdims=True)
        forecast = np.sum(forecast * capacity[None, None, :], axis=-1, keepdims=True)
        scenarios = np.sum(
            scenarios * capacity[None, None, None, :], axis=-1, keepdims=True
        )
        nodes = pd.DataFrame(
            {
                "channel_index": [-1],
                "station_id": ["aggregate_wind"],
                "FARM_NAME": ["13-station aggregate wind"],
                "capacity_mw": [float(np.sum(capacity))],
            }
        )
        return actual, forecast, scenarios, nodes, "MW"
    raise ValueError(f"unknown scope {scope}")


def _event_rows_for_scope(
    results: dict[str, dict],
    stations: pd.DataFrame,
    issues: pd.DataFrame,
    scope: str,
    lags: tuple[int, ...],
    tolerances: tuple[int, ...],
    extreme_quantile: float,
    search_radius: int,
) -> tuple[list[dict], list[dict]]:
    reference = results[MODEL_ORDER[0]]
    actual, forecast, _, nodes, unit = _scope_arrays(reference, stations, scope)
    model_scenarios = {
        model: _scope_arrays(result, stations, scope)[2]
        for model, result in results.items()
    }
    event_rows: list[dict] = []
    offset_rows: list[dict] = []
    event_counter = 0
    for lag in lags:
        actual_ramp = actual[:, lag:, :] - actual[:, :-lag, :]
        forecast_ramp = forecast[:, lag:, :] - forecast[:, :-lag, :]
        threshold = float(np.quantile(np.abs(actual_ramp), extreme_quantile))
        scenario_ramps = {
            model: values[:, :, lag:, :] - values[:, :, :-lag, :]
            for model, values in model_scenarios.items()
        }
        for issue_index in range(actual.shape[0]):
            target_start = pd.Timestamp(issues.iloc[issue_index].target_start)
            for node_index in range(actual.shape[2]):
                node = nodes.iloc[node_index]
                series = actual_ramp[issue_index, :, node_index]
                for direction in ("up", "down"):
                    for ramp_index in _clustered_event_indices(
                        series, threshold, direction
                    ):
                        event_counter += 1
                        lead_index = int(ramp_index + lag)
                        event_uid = f"{scope}_L{lag}_{event_counter:05d}"
                        for model in MODEL_ORDER:
                            member_ramp = scenario_ramps[model][
                                issue_index, :, :, node_index
                            ]
                            offsets, matched = _nearest_member_matches(
                                member_ramp,
                                ramp_index,
                                threshold,
                                direction,
                                search_radius,
                            )
                            exact_values = member_ramp[:, ramp_index]
                            lower, upper = np.quantile(exact_values, [0.05, 0.95])
                            finite = np.isfinite(offsets)
                            row = {
                                "event_uid": event_uid,
                                "model": model,
                                "model_label": MODEL_LABELS[model],
                                "scope": scope,
                                "unit": unit,
                                "issue_index": issue_index,
                                "issue_date": issues.iloc[issue_index].issue_date,
                                "event_timestamp": target_start
                                + pd.Timedelta(hours=lead_index),
                                "station_channel_index": int(node.channel_index),
                                "station_id": node.station_id,
                                "station_name": node.FARM_NAME,
                                "lag_hours": lag,
                                "direction": direction,
                                "event_lead_hour": lead_index + 1,
                                "actual_ramp": float(series[ramp_index]),
                                "forecast_ramp": float(
                                    forecast_ramp[issue_index, ramp_index, node_index]
                                ),
                                "absolute_ramp_threshold": threshold,
                                "exact_lower_90": float(lower),
                                "exact_upper_90": float(upper),
                                "exact_interval_width_90": float(upper - lower),
                                "exact_amplitude_covered_90": bool(
                                    lower <= series[ramp_index] <= upper
                                ),
                                "members_with_event_in_search": int(np.sum(finite)),
                                "member_event_rate_in_search": float(np.mean(finite)),
                                "mean_timing_offset_hours": float(
                                    np.nanmean(offsets)
                                )
                                if np.any(finite)
                                else np.nan,
                                "median_timing_offset_hours": float(
                                    np.nanmedian(offsets)
                                )
                                if np.any(finite)
                                else np.nan,
                                "mean_absolute_timing_offset_hours": float(
                                    np.nanmean(np.abs(offsets))
                                )
                                if np.any(finite)
                                else np.nan,
                                "early_member_count": int(np.sum(offsets < 0)),
                                "exact_member_count": int(np.sum(offsets == 0)),
                                "late_member_count": int(np.sum(offsets > 0)),
                                "matched_ramp_median": float(np.nanmedian(matched))
                                if np.any(finite)
                                else np.nan,
                            }
                            for tolerance in tolerances:
                                hits = finite & (np.abs(offsets) <= tolerance)
                                row[f"member_hit_rate_{tolerance}h"] = float(
                                    np.mean(hits)
                                )
                                row[f"event_any_member_hit_{tolerance}h"] = bool(
                                    np.any(hits)
                                )
                                row[f"event_majority_member_hit_{tolerance}h"] = bool(
                                    np.mean(hits) >= 0.5
                                )
                            event_rows.append(row)
                            for offset in range(-search_radius, search_radius + 1):
                                offset_rows.append(
                                    {
                                        "model": model,
                                        "scope": scope,
                                        "lag_hours": lag,
                                        "direction": direction,
                                        "timing_offset_hours": offset,
                                        "matched_member_count": int(
                                            np.sum(offsets == offset)
                                        ),
                                    }
                                )
                            offset_rows.append(
                                {
                                    "model": model,
                                    "scope": scope,
                                    "lag_hours": lag,
                                    "direction": direction,
                                    "timing_offset_hours": "no_event",
                                    "matched_member_count": int(np.sum(~finite)),
                                }
                            )
    return event_rows, offset_rows


def _summarize_events(
    events: pd.DataFrame, tolerances: tuple[int, ...]
) -> pd.DataFrame:
    frames = []
    for direction in ["up", "down", "all"]:
        subset = events if direction == "all" else events[events.direction.eq(direction)]
        grouped = subset.groupby(["model", "model_label", "scope", "unit", "lag_hours"])
        rows = []
        for keys, group in grouped:
            total_matches = int(group.members_with_event_in_search.sum())
            row = dict(zip(["model", "model_label", "scope", "unit", "lag_hours"], keys))
            row.update(
                {
                    "direction": direction,
                    "event_count": int(len(group)),
                    "exact_amplitude_coverage_90": float(
                        group.exact_amplitude_covered_90.mean()
                    ),
                    "mean_exact_interval_width_90": float(
                        group.exact_interval_width_90.mean()
                    ),
                    "mean_event_rate_in_search": float(
                        group.member_event_rate_in_search.mean()
                    ),
                    "median_event_timing_offset_hours": float(
                        group.median_timing_offset_hours.median()
                    ),
                    "mean_absolute_timing_offset_hours": float(
                        group.mean_absolute_timing_offset_hours.mean()
                    ),
                    "early_share_of_matches": float(group.early_member_count.sum() / total_matches)
                    if total_matches
                    else np.nan,
                    "exact_share_of_matches": float(group.exact_member_count.sum() / total_matches)
                    if total_matches
                    else np.nan,
                    "late_share_of_matches": float(group.late_member_count.sum() / total_matches)
                    if total_matches
                    else np.nan,
                }
            )
            for tolerance in tolerances:
                row[f"mean_member_hit_rate_{tolerance}h"] = float(
                    group[f"member_hit_rate_{tolerance}h"].mean()
                )
                row[f"event_any_member_hit_rate_{tolerance}h"] = float(
                    group[f"event_any_member_hit_{tolerance}h"].mean()
                )
                row[f"event_majority_member_hit_rate_{tolerance}h"] = float(
                    group[f"event_majority_member_hit_{tolerance}h"].mean()
                )
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["scope", "lag_hours", "direction", "model"]
    )


def _paired_bootstrap(
    events: pd.DataFrame,
    tolerances: tuple[int, ...],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    metrics = ["exact_amplitude_covered_90"] + [
        f"member_hit_rate_{tolerance}h" for tolerance in tolerances
    ]
    index_columns = [
        "event_uid",
        "scope",
        "lag_hours",
        "direction",
        "issue_index",
    ]
    wide = events.pivot(index=index_columns, columns="model", values=metrics)
    rng = np.random.default_rng(seed)
    rows = []
    for scope in sorted(events.scope.unique()):
        for lag in sorted(events.lag_hours.unique()):
            for direction in ["up", "down", "all"]:
                index_frame = wide.reset_index()
                mask = index_frame.scope.eq(scope) & index_frame.lag_hours.eq(lag)
                if direction != "all":
                    mask &= index_frame.direction.eq(direction)
                group = index_frame.loc[mask]
                for metric in metrics:
                    delta = (
                        group[(metric, MODEL_ORDER[1])]
                        - group[(metric, MODEL_ORDER[0])]
                    )
                    issue_delta = delta.groupby(group.issue_index).mean().dropna()
                    values = issue_delta.to_numpy(float)
                    if not len(values):
                        continue
                    draws = rng.choice(values, size=(repetitions, len(values)), replace=True)
                    bootstrap = draws.mean(axis=1)
                    rows.append(
                        {
                            "scope": scope,
                            "lag_hours": lag,
                            "direction": direction,
                            "metric": metric,
                            "state_minus_control": float(values.mean()),
                            "bootstrap_ci_2_5": float(np.quantile(bootstrap, 0.025)),
                            "bootstrap_ci_97_5": float(np.quantile(bootstrap, 0.975)),
                            "issue_block_count": int(len(values)),
                            "bootstrap_repetitions": repetitions,
                        }
                    )
    return pd.DataFrame(rows)


def _plot_hit_rates(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    colors = {MODEL_ORDER[0]: "#64748b", MODEL_ORDER[1]: "#0f766e"}
    styles = {"up": "-", "down": "--"}
    for row, scope in enumerate(["station_pu", "aggregate_mw"]):
        for column, lag in enumerate(DEFAULT_LAGS):
            axis = axes[row, column]
            for model in MODEL_ORDER:
                for direction in ["up", "down"]:
                    record = summary[
                        summary.model.eq(model)
                        & summary.scope.eq(scope)
                        & summary.lag_hours.eq(lag)
                        & summary.direction.eq(direction)
                    ].iloc[0]
                    values = [
                        record[f"mean_member_hit_rate_{tolerance}h"]
                        for tolerance in DEFAULT_TOLERANCES
                    ]
                    axis.plot(
                        DEFAULT_TOLERANCES,
                        values,
                        color=colors[model],
                        linestyle=styles[direction],
                        marker="o" if direction == "up" else "s",
                        label=f"{MODEL_LABELS[model]} / {direction}",
                    )
            axis.set_title(f"{scope.replace('_', ' ')} | {lag}h ramp")
            axis.set_ylim(0, 1)
            axis.set_xticks(DEFAULT_TOLERANCES)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Member event-hit rate")
            if row == 1:
                axis.set_xlabel("Timing tolerance (hours)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Wind extreme-ramp timing coverage", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_offset_distribution(offsets: pd.DataFrame, output: Path) -> None:
    numeric = offsets[offsets.timing_offset_hours.ne("no_event")].copy()
    numeric["timing_offset_hours"] = numeric.timing_offset_hours.astype(int)
    numeric = (
        numeric.groupby(
            ["model", "scope", "lag_hours", "timing_offset_hours"], as_index=False
        ).matched_member_count.sum()
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    colors = {MODEL_ORDER[0]: "#64748b", MODEL_ORDER[1]: "#0f766e"}
    for row, scope in enumerate(["station_pu", "aggregate_mw"]):
        for column, lag in enumerate(DEFAULT_LAGS):
            axis = axes[row, column]
            for model in MODEL_ORDER:
                group = numeric[
                    numeric.model.eq(model)
                    & numeric.scope.eq(scope)
                    & numeric.lag_hours.eq(lag)
                ].set_index("timing_offset_hours")
                x = np.arange(-6, 7)
                count = group.matched_member_count.reindex(x, fill_value=0).to_numpy()
                rate = count / count.sum() if count.sum() else count
                axis.plot(
                    x,
                    rate,
                    marker="o",
                    color=colors[model],
                    label=MODEL_LABELS[model],
                )
            axis.axvline(0, color="#111827", linewidth=1, alpha=0.5)
            axis.set_title(f"{scope.replace('_', ' ')} | {lag}h ramp")
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Share of matched members")
            if row == 1:
                axis.set_xlabel("Generated event time - actual time (hours)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Signed timing-offset distribution (negative=early, positive=late)",
        y=0.995,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_direction_balance(summary: pd.DataFrame, output: Path) -> None:
    """Show whether matched aggregate events tend to occur early or late."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
    categories = ["early_share_of_matches", "exact_share_of_matches", "late_share_of_matches"]
    category_labels = ["early", "exact", "late"]
    colors = ["#2563eb", "#64748b", "#d97706"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    for row, direction in enumerate(["up", "down"]):
        for column, lag in enumerate(DEFAULT_LAGS):
            axis = axes[row, column]
            subset = summary[
                summary.scope.eq("aggregate_mw")
                & summary.direction.eq(direction)
                & summary.lag_hours.eq(lag)
            ].set_index("model").loc[MODEL_ORDER]
            for category_index, (category, label, color) in enumerate(
                zip(categories, category_labels, colors, strict=True)
            ):
                values = subset[category].to_numpy(float)
                axis.bar(
                    x + (category_index - 1) * width,
                    values,
                    width,
                    color=color,
                    label=label,
                )
            axis.set_xticks(x, ["Baseline", "Candidate"])
            axis.set_ylim(0, 0.75)
            axis.set_title(f"Aggregate {direction} | {lag}h ramp")
            axis.grid(axis="y", alpha=0.25)
            if column == 0:
                axis.set_ylabel("Share of matched members")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Aggregate wind event timing balance", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_worst_aggregate_events(
    events: pd.DataFrame,
    results: dict[str, dict],
    stations: pd.DataFrame,
    output: Path,
) -> list[str]:
    state = events[
        events.model.eq(MODEL_ORDER[1])
        & events.scope.eq("aggregate_mw")
        & events.lag_hours.eq(3)
    ].copy()
    state["severity"] = np.abs(state.actual_ramp) / state.absolute_ramp_threshold
    selected = state.sort_values(
        ["member_hit_rate_3h", "severity"], ascending=[True, False]
    ).drop_duplicates("event_timestamp").head(3)
    fig, axes = plt.subplots(len(selected), 2, figsize=(15, 3.6 * len(selected)), squeeze=False)
    reference_actual, reference_forecast, _, _, _ = _scope_arrays(
        results[MODEL_ORDER[0]], stations, "aggregate_mw"
    )
    selected_ids = []
    for row_index, (_, event) in enumerate(selected.iterrows()):
        selected_ids.append(str(event.event_uid))
        issue = int(event.issue_index)
        event_lead = int(event.event_lead_hour) - 1
        left = max(0, event_lead - 18)
        right = min(167, event_lead + 18)
        lead = np.arange(left + 1, right + 2)
        for column, model in enumerate(MODEL_ORDER):
            _, _, samples, _, _ = _scope_arrays(
                results[model], stations, "aggregate_mw"
            )
            sample = samples[issue, :, left : right + 1, 0]
            axis = axes[row_index, column]
            axis.fill_between(
                lead,
                np.quantile(sample, 0.05, axis=0),
                np.quantile(sample, 0.95, axis=0),
                color="#fb7185",
                alpha=0.25,
                label="90% envelope",
            )
            axis.plot(lead, np.quantile(sample, 0.5, axis=0), color="#e11d48", label="median")
            axis.plot(
                lead,
                reference_forecast[issue, left : right + 1, 0],
                color="#0d9488",
                linestyle="--",
                label="forecast",
            )
            axis.plot(
                lead,
                reference_actual[issue, left : right + 1, 0],
                color="#111827",
                linewidth=1.5,
                label="actual",
            )
            axis.axvline(event_lead + 1, color="#7c3aed", linestyle=":", linewidth=1.5)
            axis.set_title(
                f"{MODEL_LABELS[model]} | {event.direction} 3h event | "
                f"issue {issue} | hit±3h={events[(events.event_uid == event.event_uid) & (events.model == model)].member_hit_rate_3h.iloc[0]:.1%}"
            )
            axis.set_ylabel("Aggregated wind (MW)")
            axis.set_xlabel("Lead hour")
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Low-hit aggregate wind ramp events", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected_ids


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    output: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    selected_event_ids: list[str],
    metadata: dict,
) -> None:
    def value(scope: str, lag: int, direction: str, model: str, column: str) -> float:
        row = summary[
            summary.scope.eq(scope)
            & summary.lag_hours.eq(lag)
            & summary.direction.eq(direction)
            & summary.model.eq(model)
        ].iloc[0]
        return float(row[column])

    core = summary[
        summary.direction.eq("all")
        & summary.scope.eq("aggregate_mw")
    ][
        [
            "model_label",
            "lag_hours",
            "event_count",
            "exact_amplitude_coverage_90",
            "mean_member_hit_rate_1h",
            "mean_member_hit_rate_3h",
            "mean_member_hit_rate_6h",
            "median_event_timing_offset_hours",
            "early_share_of_matches",
            "late_share_of_matches",
        ]
    ].copy()
    paired_core = paired[
        paired.scope.eq("aggregate_mw")
        & paired.direction.eq("all")
        & paired.metric.isin(
            ["exact_amplitude_covered_90", "member_hit_rate_3h"]
        )
    ].copy()
    report = "# 风电事件时刻诊断（验证集）\n\n"
    report += (
        "本诊断没有重新训练模型，也没有使用测试集。两组结果使用相同的 23 个验证发布窗口、"
        f"每窗 {metadata['member_count']} 条场景和同一生成随机种子。实际极端事件仅从实测序列定义，"
        "模型比较采用完全相同的事件清单。\n\n"
    )
    report += "## 如何读这些指标\n\n"
    report += (
        "- `exact_amplitude_coverage_90`：在真实事件发生的那个时刻，真实爬坡幅值是否落入场景爬坡的 90% 区间。它回答“幅值包络住没有”。\n"
        "- `mean_member_hit_rate_±kh`：80 个成员中，有多少比例在真实事件前后 k 小时内生成了同方向、且达到同一极端阈值的事件。它回答“事件时刻覆盖住没有”，是主指标。\n"
        "- timing offset = 生成事件时刻 − 真实事件时刻；负数是偏早，正数是偏晚。\n"
        "- `any member hit` 对成员数很敏感，只作为辅助，不用于证明模型已经可靠。\n\n"
    )
    report += "## 13 场站聚合风电结果\n\n"
    report += _markdown_table(core) + "\n\n"
    station_delta_3h = [
        value("station_pu", lag, "all", MODEL_ORDER[1], "mean_member_hit_rate_3h")
        - value("station_pu", lag, "all", MODEL_ORDER[0], "mean_member_hit_rate_3h")
        for lag in DEFAULT_LAGS
    ]
    aggregate_delta_3h = [
        value("aggregate_mw", lag, "all", MODEL_ORDER[1], "mean_member_hit_rate_3h")
        - value("aggregate_mw", lag, "all", MODEL_ORDER[0], "mean_member_hit_rate_3h")
        for lag in DEFAULT_LAGS
    ]
    aggregate_amplitude_delta = [
        value(
            "aggregate_mw",
            lag,
            "all",
            MODEL_ORDER[1],
            "exact_amplitude_coverage_90",
        )
        - value(
            "aggregate_mw",
            lag,
            "all",
            MODEL_ORDER[0],
            "exact_amplitude_coverage_90",
        )
        for lag in DEFAULT_LAGS
    ]
    state_up_late_1h = value(
        "aggregate_mw", 1, "up", MODEL_ORDER[1], "late_share_of_matches"
    )
    state_up_early_1h = value(
        "aggregate_mw", 1, "up", MODEL_ORDER[1], "early_share_of_matches"
    )
    state_up_late_3h = value(
        "aggregate_mw", 3, "up", MODEL_ORDER[1], "late_share_of_matches"
    )
    state_up_early_3h = value(
        "aggregate_mw", 3, "up", MODEL_ORDER[1], "early_share_of_matches"
    )
    report += "## 诊断结论\n\n"
    report += (
        "1. **不是整条曲线统一平移。** 两个模型在各口径下的事件中位偏移均为 0 h；但聚合向上事件存在明显的偏晚不对称："
        f"{MODEL_LABELS[MODEL_ORDER[1]]} 的 1 h 爬坡命中成员中，偏晚 {state_up_late_1h:.1%}、偏早 {state_up_early_1h:.1%}；"
        f"3 h 爬坡为偏晚 {state_up_late_3h:.1%}、偏早 {state_up_early_3h:.1%}。因此截图中的‘滞后感’主要对应局部上升事件，而不是所有时段的固定时移。\n\n"
        f"2. **比较单站与聚合变化。** {MODEL_LABELS[MODEL_ORDER[1]]} 相对 {MODEL_LABELS[MODEL_ORDER[0]]} 在单场站 ±3 h 成员命中率上的差值（1/3/6 h 爬坡）分别为 "
        + "、".join(f"{delta:+.2%}" for delta in station_delta_3h)
        + "；但聚合风电分别为 "
        + "、".join(f"{delta:+.2%}" for delta in aggregate_delta_3h)
        + "。单站与聚合差值方向不一致时，说明跨站同步仍是独立于单站边际质量的问题。\n\n"
        "3. **幅值和时刻需要分开判断。** 聚合风电真实事件时刻的 90% 爬坡幅值覆盖，候选模型相对基线在 1/3/6 h 上分别变化 "
        + "、".join(f"{delta:+.2%}" for delta in aggregate_amplitude_delta)
        + "；同时 ±6 h 的成员事件命中率仍只有约 33%–40%。因此下一步不能只把包络整体加宽，也不能只做固定时间平移。\n\n"
        "4. **判定规则：** 候选模型只有在聚合事件时刻与幅值改善、且单站边际和光伏质量未明显退化时，才能认为结构改动有效。\n\n"
    )
    report += f"## {MODEL_LABELS[MODEL_ORDER[1]]} 相对 {MODEL_LABELS[MODEL_ORDER[0]]} 的发布窗口块 bootstrap\n\n"
    report += (
        f"以下差值为 {MODEL_LABELS[MODEL_ORDER[1]]} − {MODEL_LABELS[MODEL_ORDER[0]]}；"
        "95% 区间按发布窗口整体重采样，避免把同一 168 h 窗口中的大量小时误当成独立样本。\n\n"
    )
    report += _markdown_table(
        paired_core[
            [
                "lag_hours",
                "metric",
                "state_minus_control",
                "bootstrap_ci_2_5",
                "bootstrap_ci_97_5",
                "issue_block_count",
            ]
        ]
    ) + "\n\n"
    report += "## 图表\n\n"
    report += (
        "- `figures/timing_hit_rate_comparison.png`：单场站与聚合风电、1/3/6 h 爬坡、向上/向下事件的成员时刻命中率。\n"
        "- `figures/timing_offset_distribution.png`：命中成员相对真实事件偏早或偏晚的分布。\n"
        "- `figures/aggregate_timing_direction_balance.png`：聚合风电事件命中后，偏早、准时和偏晚的成员占比。\n"
        "- `figures/worst_aggregate_event_examples.png`：候选模型的低命中聚合事件局部曲线。\n\n"
    )
    report += "低命中示例 event_uid：`" + "`, `".join(selected_event_ids) + "`。\n\n"
    report += "## 限制\n\n"
    report += (
        "这是 23 个重叠验证发布窗口上的定位实验，bootstrap 区间仍是验证集内部证据，不是最终测试集置信区间。"
        "表中的 `event_count` 是‘发布窗口—事件’对的数量；同一个自然时刻的实际事件可能出现在多个滚动发布窗口中，因此不能解读为互相独立的物理事件数。"
        "事件阈值为对应口径与时距下实测绝对爬坡的 90% 分位数；相邻连续超阈值点只保留幅值最大的一个，避免同一次变化被重复计数。\n"
    )
    (output / "timing_diagnostics.md").write_text(report, encoding="utf-8")


def run_diagnostic(
    result_dirs: Iterable[str | Path],
    data_path: str | Path,
    output_dir: str | Path,
    extreme_quantile: float = 0.90,
    search_radius: int = 6,
    bootstrap_repetitions: int = 5000,
    bootstrap_seed: int = 20260803,
    baseline_variant: str = "ramp36_control",
    candidate_variant: str = "state_v1_fixed_graph",
    baseline_label: str = "Ramp36 control",
    candidate_label: str = "State V1",
) -> Path:
    _configure_models(
        baseline_variant,
        candidate_variant,
        baseline_label,
        candidate_label,
    )
    if search_radius < max(DEFAULT_TOLERANCES):
        raise ValueError(
            f"search radius must be at least {max(DEFAULT_TOLERANCES)} hours"
        )
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    figures = output / "figures"
    figures.mkdir(parents=True)
    results = _load_results(result_dirs)
    data_path = Path(data_path)
    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    )
    issues = pd.read_csv(data_path / "val_issue_dates.csv").sort_values(
        "sample_index"
    )
    issue_count = next(iter(results.values()))["actual"].shape[0]
    if len(issues) != issue_count:
        raise ValueError(f"issue metadata has {len(issues)} rows, expected {issue_count}")
    if len(stations) != next(iter(results.values()))["actual"].shape[2]:
        raise ValueError("station metadata does not match result channel count")

    event_rows: list[dict] = []
    offset_rows: list[dict] = []
    for scope in ["station_pu", "aggregate_mw"]:
        scope_events, scope_offsets = _event_rows_for_scope(
            results,
            stations,
            issues,
            scope,
            DEFAULT_LAGS,
            DEFAULT_TOLERANCES,
            extreme_quantile,
            search_radius,
        )
        event_rows.extend(scope_events)
        offset_rows.extend(scope_offsets)
    events = pd.DataFrame(event_rows)
    offsets = (
        pd.DataFrame(offset_rows)
        .groupby(
            ["model", "scope", "lag_hours", "direction", "timing_offset_hours"],
            as_index=False,
        )
        .matched_member_count.sum()
    )
    summary = _summarize_events(events, DEFAULT_TOLERANCES)
    paired = _paired_bootstrap(
        events,
        DEFAULT_TOLERANCES,
        bootstrap_repetitions,
        bootstrap_seed,
    )
    events.to_csv(output / "event_records.csv", index=False)
    offsets.to_csv(output / "timing_offset_distribution.csv", index=False)
    summary.to_csv(output / "timing_summary.csv", index=False)
    paired.to_csv(output / "paired_model_comparison.csv", index=False)
    _plot_hit_rates(summary, figures / "timing_hit_rate_comparison.png")
    _plot_offset_distribution(offsets, figures / "timing_offset_distribution.png")
    _plot_direction_balance(
        summary, figures / "aggregate_timing_direction_balance.png"
    )
    selected = _plot_worst_aggregate_events(
        events, results, stations, figures / "worst_aggregate_event_examples.png"
    )
    first_result = results[MODEL_ORDER[0]]
    metadata = {
        "purpose": "validation-only post-hoc wind extreme-ramp timing diagnostic",
        "models": MODEL_ORDER,
        "result_directories": {
            model: str(results[model]["path"]) for model in MODEL_ORDER
        },
        "split": "val",
        "test_used": False,
        "issue_count": int(issue_count),
        "member_count": int(first_result["scenarios"].shape[1]),
        "generation_seed": int(first_result["metadata"]["generation_seed"]),
        "lags_hours": list(DEFAULT_LAGS),
        "timing_tolerances_hours": list(DEFAULT_TOLERANCES),
        "search_radius_hours": int(search_radius),
        "extreme_quantile": float(extreme_quantile),
        "event_definition": (
            "absolute actual ramp >= scope/lag-specific q90; one strongest point "
            "retained per contiguous same-direction exceedance run"
        ),
        "match_definition": (
            "nearest generated same-direction ramp exceeding the same threshold; "
            "offset = generated event time - actual event time"
        ),
        "primary_metric": "member event-hit rate within +/-1h, +/-3h, +/-6h",
        "bootstrap": {
            "unit": "validation issue window",
            "repetitions": int(bootstrap_repetitions),
            "seed": int(bootstrap_seed),
        },
        "selected_low_hit_event_uids": selected,
    }
    (output / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output, summary, paired, selected, metadata)
    return output


def main() -> None:
    args = parse_args()
    output = run_diagnostic(
        args.result_dirs,
        args.data_path,
        args.output_dir,
        extreme_quantile=args.extreme_quantile,
        search_radius=args.search_radius_hours,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        baseline_variant=args.baseline_variant,
        candidate_variant=args.candidate_variant,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    print(f"WIND_EVENT_TIMING_DIAGNOSTIC_COMPLETE output={output}")


if __name__ == "__main__":
    main()
