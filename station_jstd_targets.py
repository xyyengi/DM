"""Train-fitted continuous event targets for JSTD-Tail.

Event duration is measured from contiguous physical support.  The 1/3/6 hour
lags are observation scales for fast supervision; 12/24 hour filters are
projection scales for slow supervision.  None of those scales is an event
class or a minimum-duration rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


EXPECTED_STATIONS = 24
EXPECTED_HOURS = 168


@dataclass(frozen=True)
class JSTDTargetArrays:
    split: str
    event_active: np.ndarray
    time_support: np.ndarray
    station_support: np.ndarray
    event_hypothesis: np.ndarray
    hypothesis_time_support: np.ndarray
    hypothesis_station_support: np.ndarray
    sample_weights: np.ndarray
    catalog: tuple[dict[str, object], ...]
    audit: dict[str, object]


def _validate_data_dir(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    required = [
        root / "train_residual.npy",
        root / "train_fill_mask.npy",
        root / "train_issue_dates.csv",
        root / "station_order.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Station-24 artifacts: {missing}")
    return root


def _bridge_short_gaps(active: np.ndarray, maximum_gap: int) -> np.ndarray:
    active = np.asarray(active, dtype=bool).copy()
    if maximum_gap <= 0:
        return active
    true_index = np.flatnonzero(active)
    for left, right in zip(true_index[:-1], true_index[1:]):
        if 1 <= right - left - 1 <= maximum_gap:
            active[left : right + 1] = True
    return active


def _segments_with_seed(active: np.ndarray, seed: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(active, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if bool(np.any(seed[start:stop]))
    ]


def _positive_quantile(values: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("cannot fit a positive event threshold from empty data")
    return float(np.quantile(values, quantile))


def fit_station_jstd_event_thresholds(
    data_dir: str | Path,
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit all event thresholds on train only."""

    root = _validate_data_dir(data_dir)
    config = dict(config or {})
    enter_quantile = float(config.get("jstd_event_enter_quantile", 0.90))
    keep_fraction = float(config.get("jstd_event_keep_fraction", 0.50))
    station_quantile = float(config.get("jstd_station_support_quantile", 0.80))
    bridge_hours = int(config.get("jstd_event_bridge_hours", 1))
    if not 0.5 <= enter_quantile < 1.0:
        raise ValueError("jstd_event_enter_quantile must be in [0.5,1)")
    if not 0.0 < keep_fraction < 1.0:
        raise ValueError("jstd_event_keep_fraction must be in (0,1)")
    if not 0.5 <= station_quantile < 1.0:
        raise ValueError("jstd_station_support_quantile must be in [0.5,1)")
    if not 0 <= bridge_hours <= 3:
        raise ValueError("jstd_event_bridge_hours must be in [0,3]")

    residual = np.asarray(np.load(root / "train_residual.npy", mmap_mode="r"), dtype=np.float64)
    valid = np.asarray(np.load(root / "train_fill_mask.npy", mmap_mode="r")) == 0
    stations = (
        pd.read_csv(root / "station_order.csv")
        .sort_values("channel_index")
        .reset_index(drop=True)
    )
    capacity = stations.capacity_mw.to_numpy(dtype=np.float64)
    type_thresholds: dict[str, dict[str, float]] = {}
    station_thresholds = np.zeros((EXPECTED_STATIONS, 2), dtype=np.float64)
    for type_name in ("wind", "solar"):
        indices = stations.index[stations.data_type.eq(type_name)].to_numpy(dtype=int)
        weights = capacity[indices] / capacity[indices].sum()
        aggregate = np.einsum("nts,s->nt", residual[:, :, indices], weights)
        complete = valid[:, :, indices].all(axis=-1)
        negative = np.where(complete, np.maximum(-aggregate, 0.0), np.nan)
        positive = np.where(complete, np.maximum(aggregate, 0.0), np.nan)
        # A 168 h window almost always contains an hourly q90 exceedance. Fit
        # the entry threshold on each issue's maximum severity instead, so the
        # issue-level tail target remains genuinely sparse. Duration is still
        # recovered from contiguous hourly support below this entry threshold.
        negative_issue_max = np.nanmax(negative, axis=1)
        positive_issue_max = np.nanmax(positive, axis=1)
        type_thresholds[type_name] = {
            "negative": _positive_quantile(negative_issue_max, enter_quantile),
            "positive": _positive_quantile(positive_issue_max, enter_quantile),
        }
        for station in indices:
            values = residual[:, :, station][valid[:, :, station]]
            station_thresholds[station, 0] = _positive_quantile(
                np.maximum(-values, 0.0), station_quantile
            )
            station_thresholds[station, 1] = _positive_quantile(
                np.maximum(values, 0.0), station_quantile
            )
    return {
        "method": "train_only_continuous_signed_power_residual_events_v1",
        "fit_split": "train",
        "future_actual_used_as_condition": False,
        "event_duration_definition": "contiguous_hysteresis_support_without_minimum_duration",
        "event_type_uses_fixed_hours": False,
        "fast_observation_lags_hours": [1, 3, 6],
        "slow_projection_widths_hours": [12, 24],
        "enter_quantile": enter_quantile,
        "enter_quantile_population": "per_issue_maximum_severity",
        "keep_fraction": keep_fraction,
        "station_support_quantile": station_quantile,
        "bridge_hours": bridge_hours,
        "type_thresholds": type_thresholds,
        "station_thresholds": station_thresholds.tolist(),
    }


def validate_station_jstd_event_thresholds(
    specification: Mapping[str, object],
) -> dict[str, object]:
    result = dict(specification)
    if result.get("fit_split") != "train":
        raise ValueError("JSTD event thresholds must be fitted on train")
    if bool(result.get("future_actual_used_as_condition", True)):
        raise ValueError("JSTD target residuals cannot be generation conditions")
    if bool(result.get("event_type_uses_fixed_hours", True)):
        raise ValueError("JSTD event identity cannot use fixed duration classes")
    if list(result.get("fast_observation_lags_hours", [])) != [1, 3, 6]:
        raise ValueError("JSTD fast supervision must use 1/3/6 h observation lags")
    if list(result.get("slow_projection_widths_hours", [])) != [12, 24]:
        raise ValueError("JSTD slow constraints must use 12/24 h projections")
    thresholds = np.asarray(result.get("station_thresholds"), dtype=np.float64)
    if thresholds.shape != (EXPECTED_STATIONS, 2) or np.any(thresholds <= 0):
        raise ValueError("invalid JSTD station thresholds")
    return result


def build_station_jstd_target_arrays(
    data_dir: str | Path,
    split: str,
    thresholds: Mapping[str, object],
) -> JSTDTargetArrays:
    """Apply train-fitted thresholds to one split and retain actual duration."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split={split!r}")
    root = _validate_data_dir(data_dir)
    fitted = validate_station_jstd_event_thresholds(thresholds)
    residual = np.asarray(np.load(root / f"{split}_residual.npy", mmap_mode="r"), dtype=np.float64)
    valid = np.asarray(np.load(root / f"{split}_fill_mask.npy", mmap_mode="r")) == 0
    issues = pd.read_csv(root / f"{split}_issue_dates.csv")
    stations = (
        pd.read_csv(root / "station_order.csv")
        .sort_values("channel_index")
        .reset_index(drop=True)
    )
    capacity = stations.capacity_mw.to_numpy(dtype=np.float64)
    sample_count = residual.shape[0]
    time_support = np.zeros((sample_count, EXPECTED_HOURS), dtype=np.float32)
    station_support = np.zeros(
        (sample_count, EXPECTED_STATIONS, EXPECTED_HOURS), dtype=np.float32
    )
    event_hypothesis = np.zeros((sample_count, 6), dtype=np.float32)
    hypothesis_time_support = np.zeros(
        (sample_count, EXPECTED_HOURS), dtype=np.float32
    )
    hypothesis_station_support = np.zeros(
        (sample_count, EXPECTED_STATIONS, EXPECTED_HOURS), dtype=np.float32
    )
    catalog: list[dict[str, object]] = []
    candidates: list[list[dict[str, object]]] = [
        [] for _ in range(sample_count)
    ]
    bridge = int(fitted["bridge_hours"])
    keep_fraction = float(fitted["keep_fraction"])
    station_threshold = np.asarray(fitted["station_thresholds"], dtype=np.float64)
    target_start_column = "target_start" if "target_start" in issues else "issue_date"
    target_starts = pd.to_datetime(issues[target_start_column])

    for type_name in ("wind", "solar"):
        indices = stations.index[stations.data_type.eq(type_name)].to_numpy(dtype=int)
        weights = capacity[indices] / capacity[indices].sum()
        aggregate = np.einsum("nts,s->nt", residual[:, :, indices], weights)
        complete = valid[:, :, indices].all(axis=-1)
        for direction_index, direction_name in enumerate(("negative", "positive")):
            sign = -1.0 if direction_name == "negative" else 1.0
            magnitude = np.maximum(sign * aggregate, 0.0)
            enter = float(fitted["type_thresholds"][type_name][direction_name])
            keep = keep_fraction * enter
            for sample in range(sample_count):
                seed = complete[sample] & (magnitude[sample] >= enter)
                active = _bridge_short_gaps(
                    complete[sample] & (magnitude[sample] >= keep), bridge
                )
                for start, stop in _segments_with_seed(active, seed):
                    severity = np.clip(magnitude[sample, start:stop] / enter, 0.0, 4.0)
                    time_support[sample, start:stop] = np.maximum(
                        time_support[sample, start:stop], severity / 4.0
                    )
                    for station in indices:
                        local = np.maximum(sign * residual[sample, start:stop, station], 0.0)
                        support = np.clip(
                            local / station_threshold[station, direction_index], 0.0, 4.0
                        ) / 4.0
                        support *= valid[sample, start:stop, station]
                        station_support[sample, station, start:stop] = np.maximum(
                            station_support[sample, station, start:stop], support
                        )
                    onset_time = target_starts.iloc[sample] + pd.Timedelta(hours=start)
                    stop_time = target_starts.iloc[sample] + pd.Timedelta(hours=stop)
                    segment = aggregate[sample, start:stop]
                    catalog.append(
                        {
                            "sample_index": int(sample),
                            "source": type_name,
                            "direction": direction_name,
                            "lead_onset": int(start),
                            "lead_stop_exclusive": int(stop),
                            "actual_duration_hours": int(stop - start),
                            "physical_onset": onset_time.isoformat(),
                            "physical_stop_exclusive": stop_time.isoformat(),
                            "depth": float(np.max(magnitude[sample, start:stop])),
                            "mean_signed_residual": float(np.mean(segment)),
                            "integrated_absolute_residual": float(np.sum(np.abs(segment))),
                        }
                    )
                    candidates[sample].append(catalog[-1])

    # H1 is an explicit controllability upper bound.  Each issuance receives
    # one compact event hypothesis rather than the full future residual.  For
    # issues containing several events, use the strongest normalized physical
    # event so the hypothesis and localized supervision describe one coherent
    # onset/duration pair.
    type_aggregate: dict[str, np.ndarray] = {}
    for type_name in ("wind", "solar"):
        indices = stations.index[stations.data_type.eq(type_name)].to_numpy(dtype=int)
        weights = capacity[indices] / capacity[indices].sum()
        type_aggregate[type_name] = np.einsum(
            "nts,s->nt", residual[:, :, indices], weights
        )
    selected_catalog: list[dict[str, object]] = []
    for sample, sample_candidates in enumerate(candidates):
        if not sample_candidates:
            continue
        def event_score(item: Mapping[str, object]) -> float:
            threshold = float(
                fitted["type_thresholds"][str(item["source"])][str(item["direction"])]
            )
            duration = max(int(item["actual_duration_hours"]), 1)
            return float(item["depth"]) / max(threshold, 1e-8) * np.sqrt(duration)

        chosen = max(sample_candidates, key=event_score)
        selected_catalog.append(dict(chosen))
        start = int(chosen["lead_onset"])
        stop = int(chosen["lead_stop_exclusive"])
        duration = stop - start
        hypothesis_time_support[sample, start:stop] = time_support[sample, start:stop]
        hypothesis_station_support[sample, :, start:stop] = station_support[
            sample, :, start:stop
        ]
        signed_amplitudes: dict[str, float] = {}
        for type_name in ("wind", "solar"):
            segment = type_aggregate[type_name][sample, start:stop]
            extreme = float(segment[np.argmax(np.abs(segment))])
            direction = "positive" if extreme >= 0.0 else "negative"
            threshold = float(fitted["type_thresholds"][type_name][direction])
            signed_amplitudes[type_name] = float(
                np.clip(extreme / max(threshold, 1e-8), -4.0, 4.0) / 4.0
            )
        source = str(chosen["source"])
        source_indices = stations.index[
            stations.data_type.eq(source)
        ].to_numpy(dtype=int)
        source_weights = capacity[source_indices]
        source_weights = source_weights / source_weights.sum()
        source_support = hypothesis_station_support[
            sample, source_indices, start:stop
        ].max(axis=1)
        synchrony = float(np.sum(source_weights * source_support))
        event_hypothesis[sample] = np.asarray(
            [
                1.0,
                start / float(EXPECTED_HOURS - 1),
                duration / float(EXPECTED_HOURS),
                signed_amplitudes["wind"],
                signed_amplitudes["solar"],
                np.clip(synchrony, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    event_active = (time_support.max(axis=1) > 0).astype(np.float32)
    # Moderate replay only changes how often event-bearing issuance windows are
    # drawn. The per-sample inverse weight still calibrates issue-level gates.
    sample_weights = 1.0 + 2.0 * event_active
    durations = np.asarray(
        [row["actual_duration_hours"] for row in catalog], dtype=np.float64
    )
    physical_keys = {
        (row["source"], row["direction"], row["physical_onset"], row["physical_stop_exclusive"])
        for row in catalog
    }
    audit = {
        "method": "continuous_signed_event_targets_v1",
        "split": split,
        "fit_split": "train",
        "sample_count": int(sample_count),
        "event_bearing_issue_count": int(event_active.sum()),
        "window_event_count": int(len(catalog)),
        "exact_physical_event_count": int(len(physical_keys)),
        "duration_min_hours": float(durations.min()) if durations.size else None,
        "duration_median_hours": float(np.median(durations)) if durations.size else None,
        "duration_max_hours": float(durations.max()) if durations.size else None,
        "contains_sub_6h_events": bool(np.any(durations < 6)) if durations.size else False,
        "fixed_duration_event_classes": False,
        "future_actual_used_as_condition": False,
        "h1_event_hypothesis_dimension": 6,
        "h1_event_hypothesis_fields": [
            "active",
            "onset_fraction",
            "duration_fraction",
            "signed_wind_depth",
            "signed_solar_depth",
            "source_synchrony",
        ],
        "h1_selected_event_count": int(len(selected_catalog)),
        "h1_hypothesis_source": "split_actual_residual_for_upper_bound_only",
    }
    return JSTDTargetArrays(
        split=split,
        event_active=event_active,
        time_support=time_support,
        station_support=station_support,
        event_hypothesis=event_hypothesis,
        hypothesis_time_support=hypothesis_time_support,
        hypothesis_station_support=hypothesis_station_support,
        sample_weights=sample_weights.astype(np.float64),
        catalog=tuple(catalog),
        audit=audit,
    )
