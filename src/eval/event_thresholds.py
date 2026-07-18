#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Leak-free threshold review helpers for event-level evaluation.

This module deliberately stops before model/event matching. It reconstructs
unique hourly timelines from overlapping 168-hour windows, fits thresholds on
the training timeline only, and reports candidate counts on each split.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


CHANNELS = ("wind", "solar", "load")
DURATIONS = (6, 12, 24, 48)


def reconstruct_sliding_windows(
    windows: np.ndarray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """Reconstruct ``[hours, channels]`` from stride-1 windows.

    Returns the reconstructed timeline and the maximum absolute overlap error.
    The input is not modified.
    """
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError(f"windows must be [N, L, C], got {windows.shape}")
    if windows.shape[0] == 0 or windows.shape[1] < 2:
        raise ValueError(f"windows must contain data and L >= 2, got {windows.shape}")

    left = windows[:-1, 1:, :]
    right = windows[1:, :-1, :]
    max_error = float(np.max(np.abs(left - right))) if left.size else 0.0
    if not np.allclose(left, right, atol=atol, rtol=rtol):
        raise ValueError(
            "Adjacent windows do not agree on their 167-hour overlap; "
            f"max_abs_error={max_error:.6g}"
        )

    hourly = np.concatenate([windows[0], windows[1:, -1, :]], axis=0)
    return hourly, max_error


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_hourly_split(data_path: str | Path, split: str) -> dict:
    """Load one normalized split, verify overlaps, and denormalize channels."""
    data_path = Path(data_path)
    metadata = _read_json(data_path / "export_metadata.json")
    params = _read_json(data_path / "normalization_params.json")
    split_meta = metadata["splits"][split]

    windows = np.load(data_path / f"{split}_actual.npy").astype(np.float64)
    hourly_norm, max_error = reconstruct_sliding_windows(windows)
    expected_hours = int(split_meta["hours"])
    if hourly_norm.shape != (expected_hours, 3):
        raise ValueError(
            f"{split} reconstructed shape {hourly_norm.shape} != {(expected_hours, 3)}"
        )

    scales = np.asarray(
        [
            params["wind_total_capacity"],
            params["solar_total_capacity"],
            params["load_denominator"],
        ],
        dtype=np.float64,
    )
    hourly = hourly_norm * scales.reshape(1, 3)

    start = datetime.fromisoformat(split_meta["start_local"])
    timestamps = [start + timedelta(hours=i) for i in range(expected_hours)]
    expected_end = datetime.fromisoformat(split_meta["end_local"])
    if timestamps[-1] != expected_end:
        raise ValueError(
            f"{split} reconstructed end {timestamps[-1].isoformat()} "
            f"!= metadata end {expected_end.isoformat()}"
        )

    return {
        "split": split,
        "hourly": hourly,
        "hourly_normalized": hourly_norm,
        "timestamps": timestamps,
        "hours": expected_hours,
        "windows": int(windows.shape[0]),
        "window_length": int(windows.shape[1]),
        "start_local": split_meta["start_local"],
        "end_local": split_meta["end_local"],
        "overlap_max_abs_error_normalized": max_error,
        "missing_hours": 0,
        # Window timestamps are implied by stride-one starts in export_metadata.
        # Report both common meanings of "duplicate timestamp count" explicitly.
        "repeated_distinct_timestamps": int(max(expected_hours - 2, 0)),
        "redundant_timestamp_occurrences": int(
            windows.shape[0] * windows.shape[1] - expected_hours
        ),
        "scales": {
            "wind_capacity_mw": float(scales[0]),
            "solar_capacity_mw": float(scales[1]),
            "load_denominator_mw": float(scales[2]),
        },
    }


def calculate_features(hourly: np.ndarray, timestamps: Sequence[datetime]) -> dict:
    """Calculate hourly power-risk features from a unique physical timeline."""
    wind = hourly[:, 0]
    solar = hourly[:, 1]
    load = hourly[:, 2]
    renewable = wind + solar
    net_load = load - renewable

    hours = np.asarray([t.hour for t in timestamps])
    daylight = (hours >= 7) & (hours <= 18)

    return {
        "wind_mw": wind,
        "solar_mw": solar,
        "load_mw": load,
        "renewable_mw": renewable,
        "net_load_mw": net_load,
        "daylight_mask": daylight,
        "renewable_drop_1h_mw": np.maximum(renewable[:-1] - renewable[1:], 0.0),
        "wind_drop_1h_mw": np.maximum(wind[:-1] - wind[1:], 0.0),
        "solar_drop_1h_mw": np.maximum(solar[:-1] - solar[1:], 0.0),
        "load_rise_1h_mw": np.maximum(load[1:] - load[:-1], 0.0),
    }


def daily_daylight_energy_ratio(
    solar_mw: np.ndarray,
    timestamps: Sequence[datetime],
    solar_capacity_mw: float,
) -> Tuple[List[datetime], np.ndarray]:
    """Return one daylight-energy capacity factor per local calendar day.

    Daylight is fixed at 07:00--18:59 (12 hourly observations).  A complete
    day has ratio = sum(solar MW over those hours) / (capacity MW * 12 h).
    Incomplete boundary days are rejected rather than silently rescaled.
    """
    by_day: Dict[object, List[int]] = {}
    for idx, timestamp in enumerate(timestamps):
        if 7 <= timestamp.hour <= 18:
            by_day.setdefault(timestamp.date(), []).append(idx)

    day_starts: List[datetime] = []
    ratios: List[float] = []
    for day, indices in sorted(by_day.items()):
        if len(indices) != 12:
            continue
        first = timestamps[indices[0]]
        day_starts.append(first.replace(hour=7, minute=0, second=0, microsecond=0))
        ratios.append(float(np.sum(solar_mw[indices]) / (solar_capacity_mw * 12.0)))
    return day_starts, np.asarray(ratios, dtype=np.float64)


def _percentiles(values: np.ndarray, levels: Iterable[int]) -> Dict[str, float]:
    return {f"p{level:02d}": float(np.percentile(values, level)) for level in levels}


def _positive_percentiles(values: np.ndarray, levels: Iterable[int]) -> Dict[str, float]:
    positive = values[values > 0]
    if positive.size == 0:
        return {f"p{level:02d}": 0.0 for level in levels}
    return _percentiles(positive, levels)


def fit_event_thresholds(train_split: Mapping[str, object]) -> dict:
    """Fit all review thresholds on the unique training timeline only."""
    hourly = np.asarray(train_split["hourly"])
    timestamps = train_split["timestamps"]
    scales = train_split["scales"]
    features = calculate_features(hourly, timestamps)

    wind_ratio = features["wind_mw"] / float(scales["wind_capacity_mw"])
    solar_ratio = features["solar_mw"] / float(scales["solar_capacity_mw"])
    renewable_ratio = features["renewable_mw"] / (
        float(scales["wind_capacity_mw"]) + float(scales["solar_capacity_mw"])
    )
    daylight = features["daylight_mask"]
    _, solar_daily_ratio = daily_daylight_energy_ratio(
        features["solar_mw"], timestamps, float(scales["solar_capacity_mw"])
    )

    thresholds = {
        "fit_split": "train",
        "fit_start_local": train_split["start_local"],
        "fit_end_local": train_split["end_local"],
        "fit_unique_hours": int(train_split["hours"]),
        "daylight_rule": "local hour 07:00-18:59",
        "low_wind_capacity_ratio": _percentiles(wind_ratio, (5, 10)),
        "low_solar_daily_daylight_energy_ratio": _percentiles(
            solar_daily_ratio, (5, 10)
        ),
        "low_renewable_capacity_ratio": _percentiles(renewable_ratio, (5, 10)),
        "high_load_mw": _percentiles(features["load_mw"], (90, 95)),
        "high_net_load_mw": _percentiles(features["net_load_mw"], (90, 95)),
        "renewable_drop_1h_mw_positive": _positive_percentiles(
            features["renewable_drop_1h_mw"], (90, 95)
        ),
        "wind_drop_1h_mw_positive": _positive_percentiles(
            features["wind_drop_1h_mw"], (90, 95)
        ),
        "solar_drop_1h_mw_positive": _positive_percentiles(
            features["solar_drop_1h_mw"], (90, 95)
        ),
        "load_rise_1h_mw_positive": _positive_percentiles(
            features["load_rise_1h_mw"], (90, 95)
        ),
        "absolute_net_load_ramp_mw": {},
    }

    net_load = features["net_load_mw"]
    for horizon in (1, 6, 12):
        ramp = np.abs(net_load[horizon:] - net_load[:-horizon])
        thresholds["absolute_net_load_ramp_mw"][f"{horizon}h"] = _percentiles(
            ramp, (90, 95)
        )
    return thresholds


def bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill bounded False runs of length ``<= max_gap`` in a boolean mask."""
    result = np.asarray(mask, dtype=bool).copy()
    if max_gap <= 0 or result.size < 3:
        return result
    false_runs = true_runs(~result)
    for start, end in false_runs:
        bounded = start > 0 and end < result.size - 1
        if bounded and end - start + 1 <= max_gap:
            result[start : end + 1] = True
    return result


def true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive ``(start, end)`` ranges for contiguous True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def duration_counts(mask: np.ndarray, *, gap_hours: int = 1) -> Dict[str, int]:
    runs = true_runs(bridge_short_gaps(mask, gap_hours))
    return {
        f"at_least_{duration}h": int(
            sum(end - start + 1 >= duration for start, end in runs)
        )
        for duration in DURATIONS
    }


def merge_intervals(
    intervals: Sequence[Tuple[int, int]], *, max_gap: int = 0
) -> List[Tuple[int, int]]:
    """Merge overlapping intervals, optionally allowing a small gap."""
    if not intervals:
        return []
    ordered = sorted((int(a), int(b)) for a, b in intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + max_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def ramp_event_summary(
    net_load: np.ndarray,
    *,
    horizon: int,
    threshold: float,
    merge_gap_hours: int = 0,
) -> dict:
    ramp = np.abs(net_load[horizon:] - net_load[:-horizon])
    exceedance_end = np.flatnonzero(ramp >= threshold) + horizon
    candidates = [(int(end - horizon), int(end)) for end in exceedance_end]
    events = merge_intervals(candidates, max_gap=merge_gap_hours)
    return {
        "exceedance_points": int(exceedance_end.size),
        "merged_candidate_events": int(len(events)),
    }


def summarize_candidate_counts(split: Mapping[str, object], thresholds: Mapping[str, object]) -> dict:
    """Apply train-fitted thresholds and return review-only candidate counts."""
    hourly = np.asarray(split["hourly"])
    timestamps = split["timestamps"]
    scales = split["scales"]
    features = calculate_features(hourly, timestamps)
    daylight = features["daylight_mask"]

    wind_ratio = features["wind_mw"] / float(scales["wind_capacity_mw"])
    solar_ratio = features["solar_mw"] / float(scales["solar_capacity_mw"])
    renewable_ratio = features["renewable_mw"] / (
        float(scales["wind_capacity_mw"]) + float(scales["solar_capacity_mw"])
    )

    out = {
        "unique_hours": int(split["hours"]),
        "low_wind": {},
        "low_solar_daily_energy": {},
        "low_renewable": {},
        "high_load": {},
        "high_net_load": {},
        "net_load_ramp": {},
        "wind_drop_load_rise": {},
        "solar_drop_load_rise": {},
        "compound_low_renewable_high_net_load": {},
    }

    for level in (90, 95):
        low_key = "p10" if level == 90 else "p05"
        high_key = f"p{level}"

        out["low_wind"][low_key] = duration_counts(
            wind_ratio <= thresholds["low_wind_capacity_ratio"][low_key]
        )

        _, solar_daily_ratio = daily_daylight_energy_ratio(
            features["solar_mw"], timestamps, float(scales["solar_capacity_mw"])
        )
        out["low_solar_daily_energy"][low_key] = {
            "candidate_days": int(np.sum(
                solar_daily_ratio
                <= thresholds["low_solar_daily_daylight_energy_ratio"][low_key]
            )),
            "total_complete_days": int(solar_daily_ratio.size),
        }

        low_renewable_mask = (
            renewable_ratio <= thresholds["low_renewable_capacity_ratio"][low_key]
        )
        out["low_renewable"][low_key] = duration_counts(low_renewable_mask)

        high_load_mask = features["load_mw"] >= thresholds["high_load_mw"][high_key]
        out["high_load"][high_key] = duration_counts(high_load_mask)

        high_net_load_mask = (
            features["net_load_mw"] >= thresholds["high_net_load_mw"][high_key]
        )
        out["high_net_load"][high_key] = duration_counts(high_net_load_mask)

        compound_mask = low_renewable_mask & high_net_load_mask
        out["compound_low_renewable_high_net_load"][f"{low_key}_{high_key}"] = duration_counts(
            compound_mask
        )

        load_rise_threshold = thresholds["load_rise_1h_mw_positive"][high_key]
        wind_opposition = (
            (features["wind_drop_1h_mw"] >= thresholds["wind_drop_1h_mw_positive"][high_key])
            & (features["load_rise_1h_mw"] >= load_rise_threshold)
        )
        solar_opposition = (
            (features["solar_drop_1h_mw"] >= thresholds["solar_drop_1h_mw_positive"][high_key])
            & (features["load_rise_1h_mw"] >= load_rise_threshold)
        )
        out["wind_drop_load_rise"][high_key] = {
            "simultaneous_change_points": int(np.sum(wind_opposition)),
        }
        out["solar_drop_load_rise"][high_key] = {
            "simultaneous_change_points": int(np.sum(solar_opposition)),
        }

        for horizon in (1, 6, 12):
            out["net_load_ramp"][f"{horizon}h_{high_key}"] = ramp_event_summary(
                features["net_load_mw"],
                horizon=horizon,
                threshold=thresholds["absolute_net_load_ramp_mw"][f"{horizon}h"][high_key],
            )

    return out


def _event_row(
    split: str,
    event_type: str,
    sequence: int,
    timestamps: Sequence[datetime],
    start_idx: int,
    end_idx: int,
    values: np.ndarray,
    threshold: float,
    direction: str,
    unit: str,
) -> dict:
    segment = np.asarray(values[start_idx : end_idx + 1], dtype=float)
    relative_peak = int(np.argmin(segment) if direction == "low" else np.argmax(segment))
    peak_idx = start_idx + relative_peak
    peak_value = float(segment[relative_peak])
    if direction == "low":
        severity = float(threshold - peak_value)
    else:
        severity = float(peak_value - threshold)
    return {
        "event_id": f"{split}_{event_type}_{sequence:04d}",
        "split": split,
        "event_type": event_type,
        "start_time": timestamps[start_idx].isoformat(),
        "end_time": timestamps[end_idx].isoformat(),
        "duration_hours": int(end_idx - start_idx + 1),
        "peak_time": timestamps[peak_idx].isoformat(),
        "peak_value": peak_value,
        "threshold": float(threshold),
        "severity_beyond_threshold": severity,
        "unit": unit,
    }


def _events_from_mask(
    split: Mapping[str, object],
    event_type: str,
    mask: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str,
    unit: str,
    *,
    min_duration: int = 6,
    max_gap_hours: int = 1,
) -> List[dict]:
    bridged = bridge_short_gaps(mask, max_gap_hours)
    runs = [(a, b) for a, b in true_runs(bridged) if b - a + 1 >= min_duration]
    return [
        _event_row(
            str(split["split"]), event_type, sequence, split["timestamps"], start, end,
            values, threshold, direction, unit,
        )
        for sequence, (start, end) in enumerate(runs, 1)
    ]


def build_event_catalog(
    split: Mapping[str, object],
    thresholds: Mapping[str, object],
    *,
    low_level: str = "p10",
    high_level: str = "p90",
    max_gap_hours: int = 1,
) -> List[dict]:
    """Build a review catalog from hourly points, never from 168h labels.

    Persistent states require six hours and bridge at most one missing hour by
    default. Daily solar events use daylight energy. Ramp events are intervals
    of their physical horizon. Wind/solar opposition are separate 1h events.
    """
    hourly = np.asarray(split["hourly"])
    timestamps = split["timestamps"]
    scales = split["scales"]
    f = calculate_features(hourly, timestamps)
    wind_ratio = f["wind_mw"] / float(scales["wind_capacity_mw"])
    renewable_ratio = f["renewable_mw"] / (
        float(scales["wind_capacity_mw"]) + float(scales["solar_capacity_mw"])
    )
    specs = [
        ("low_wind", wind_ratio, thresholds["low_wind_capacity_ratio"][low_level], "low", "capacity_ratio"),
        ("low_renewable", renewable_ratio, thresholds["low_renewable_capacity_ratio"][low_level], "low", "capacity_ratio"),
        ("high_load", f["load_mw"], thresholds["high_load_mw"][high_level], "high", "MW"),
        ("high_net_load", f["net_load_mw"], thresholds["high_net_load_mw"][high_level], "high", "MW"),
    ]
    events: List[dict] = []
    for event_type, values, threshold, direction, unit in specs:
        mask = values <= threshold if direction == "low" else values >= threshold
        events.extend(_events_from_mask(
            split, event_type, mask, values, threshold, direction, unit,
            min_duration=6, max_gap_hours=max_gap_hours,
        ))

    # Compound stress requires low total renewable and high net load together.
    renewable_threshold = thresholds["low_renewable_capacity_ratio"][low_level]
    net_load_threshold = thresholds["high_net_load_mw"][high_level]
    compound_mask = (
        (renewable_ratio <= renewable_threshold)
        & (f["net_load_mw"] >= net_load_threshold)
    )
    compound_score = np.minimum(
        renewable_threshold / np.maximum(renewable_ratio, 1e-12),
        f["net_load_mw"] / max(net_load_threshold, 1e-12),
    )
    events.extend(_events_from_mask(
        split, "compound_low_renewable_high_net_load", compound_mask,
        compound_score, 1.0, "high", "joint_tail_ratio",
        min_duration=6, max_gap_hours=max_gap_hours,
    ))

    # Daily daylight solar energy: each selected local day is one 12h event.
    day_starts, day_ratios = daily_daylight_energy_ratio(
        f["solar_mw"], timestamps, float(scales["solar_capacity_mw"])
    )
    solar_threshold = thresholds["low_solar_daily_daylight_energy_ratio"][low_level]
    timestamp_to_idx = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
    solar_sequence = 0
    for day_start, ratio in zip(day_starts, day_ratios):
        if ratio > solar_threshold:
            continue
        solar_sequence += 1
        start_idx = timestamp_to_idx[day_start]
        end_idx = start_idx + 11
        row = _event_row(
            str(split["split"]), "low_solar_daily_energy", solar_sequence,
            timestamps, start_idx, end_idx,
            np.full(len(timestamps), math.nan), solar_threshold, "low", "daylight_capacity_factor",
        )
        row["peak_time"] = day_start.isoformat()
        row["peak_value"] = float(ratio)
        row["severity_beyond_threshold"] = float(solar_threshold - ratio)
        events.append(row)

    # Net-load ramp intervals.  Overlap is merged; no extra gap is introduced.
    for horizon in (1, 6, 12):
        ramp = np.abs(f["net_load_mw"][horizon:] - f["net_load_mw"][:-horizon])
        threshold = thresholds["absolute_net_load_ramp_mw"][f"{horizon}h"][high_level]
        intervals = [(int(end - horizon), int(end)) for end in np.flatnonzero(ramp >= threshold) + horizon]
        for sequence, (start, end) in enumerate(merge_intervals(intervals, max_gap=0), 1):
            # Place each horizon change at its ending hour for peak selection.
            aligned = np.full(len(timestamps), -np.inf)
            aligned[horizon:] = ramp
            events.append(_event_row(
                str(split["split"]), f"high_ramp_{horizon}h", sequence,
                timestamps, start, end, aligned, threshold, "high", "MW",
            ))

    # Separate source/load opposition: either wind OR solar may fall while load rises.
    for source in ("wind", "solar"):
        drop = f[f"{source}_drop_1h_mw"]
        drop_threshold = thresholds[f"{source}_drop_1h_mw_positive"][high_level]
        rise_threshold = thresholds["load_rise_1h_mw_positive"][high_level]
        mask = (drop >= drop_threshold) & (f["load_rise_1h_mw"] >= rise_threshold)
        for sequence, change_idx in enumerate(np.flatnonzero(mask), 1):
            start, end = int(change_idx), int(change_idx + 1)
            # Peak value is a dimensionless minimum tail ratio; >=1 passes both rules.
            joint_score = min(
                float(drop[change_idx] / max(drop_threshold, 1e-12)),
                float(f["load_rise_1h_mw"][change_idx] / max(rise_threshold, 1e-12)),
            )
            row = _event_row(
                str(split["split"]), f"{source}_drop_load_rise_1h", sequence,
                timestamps, start, end, np.ones(len(timestamps)), 1.0, "high", "joint_tail_ratio",
            )
            row["peak_time"] = timestamps[end].isoformat()
            row["peak_value"] = joint_score
            row["severity_beyond_threshold"] = joint_score - 1.0
            row["source_drop_mw"] = float(drop[change_idx])
            row["load_rise_mw"] = float(f["load_rise_1h_mw"][change_idx])
            events.append(row)

    events.sort(key=lambda row: (row["start_time"], row["event_type"], row["event_id"]))
    return events


def map_windows_to_events(split: Mapping[str, object], events: Sequence[Mapping[str, object]]) -> List[dict]:
    """Retain every event/window pair with at least one overlapping hour."""
    timestamps = split["timestamps"]
    window_length = int(split["window_length"])
    num_windows = int(split["windows"])
    index_by_iso = {timestamp.isoformat(): idx for idx, timestamp in enumerate(timestamps)}
    rows: List[dict] = []
    for event in events:
        event_start_idx = index_by_iso[str(event["start_time"])]
        event_end_idx = index_by_iso[str(event["end_time"])]
        first_window = max(0, event_start_idx - window_length + 1)
        last_window = min(num_windows - 1, event_end_idx)
        for window_id in range(first_window, last_window + 1):
            window_end_idx = window_id + window_length - 1
            overlap_start = max(window_id, event_start_idx)
            overlap_end = min(window_end_idx, event_end_idx)
            lead_hours = event_start_idx - window_id
            rows.append({
                "window_id": f"{split['split']}_window_{window_id:05d}",
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "window_start": timestamps[window_id].isoformat(),
                "window_end": timestamps[window_end_idx].isoformat(),
                "event_start": event["start_time"],
                "event_end": event["end_time"],
                "lead_hours": int(lead_hours),
                "event_start_index": int(lead_hours),
                "event_end_index": int(event_end_idx - window_id),
                "overlap_hours": int(overlap_end - overlap_start + 1),
                "contains_event_start": bool(window_id <= event_start_idx <= window_end_idx),
                "fully_contains_event": bool(window_id <= event_start_idx and event_end_idx <= window_end_idx),
                "post_onset": bool(lead_hours < 0),
            })
    return rows
