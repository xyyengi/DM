"""Train-only discrete wind-event memory for the 24-station model.

Unlike :mod:`station_retrieval_memory`, this module never averages retrieved
histories.  It builds a bank of local, joint 13-wind-station residual events
and returns a forecast-only candidate set for every issue.  A generated member
must later select exactly one candidate (prototype + lead-time support).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HOURS = 168
STATIONS = 24
EVENT_DURATIONS = (6, 12, 24)
EVENT_TYPES = ("sustained_drop", "down_ramp", "up_ramp", "large_mismatch")


@dataclass(frozen=True)
class DiscreteEventArrays:
    residual: np.ndarray
    distance: np.ndarray
    prior_weight: np.ndarray
    train_index: np.ndarray
    time_mask: np.ndarray
    event_type: np.ndarray
    duration: np.ndarray
    target_start: np.ndarray
    source_start: np.ndarray
    audit: dict[str, object]


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    return np.convolve(values, np.ones(width, dtype=np.float64) / width, mode="valid")


def _resample_patch(values: np.ndarray, bins: int = 6) -> np.ndarray:
    """Resample [time,station] to a fixed forecast-only descriptor."""

    edges = np.linspace(0, len(values), bins + 1).round().astype(int)
    blocks = []
    for left, right in zip(edges[:-1], edges[1:]):
        right = max(right, left + 1)
        blocks.append(values[left:right].mean(axis=0))
    return np.stack(blocks, axis=0)


def _local_key(patch: np.ndarray, capacity_weight: np.ndarray) -> np.ndarray:
    shape = _resample_patch(np.asarray(patch, dtype=np.float64), bins=6)
    aggregate = shape @ capacity_weight
    ramp = np.diff(aggregate, prepend=aggregate[:1])
    station_mean = shape.mean(axis=0)
    station_std = shape.std(axis=0)
    return np.concatenate(
        [shape.reshape(-1), aggregate, ramp, station_mean, station_std]
    )


def _event_candidates(
    train_forecast: np.ndarray,
    train_residual: np.ndarray,
    train_valid: np.ndarray,
    issues: pd.DataFrame,
    wind_indices: np.ndarray,
    wind_weight: np.ndarray,
    quantile: float,
) -> list[dict[str, object]]:
    aggregate_residual = np.einsum(
        "nts,s->nt", train_residual[:, :, wind_indices], wind_weight
    )
    target_column = "target_start" if "target_start" in issues else "issue_date"
    target_starts = pd.to_datetime(issues[target_column])
    records: list[dict[str, object]] = []
    raw: dict[tuple[int, int], list[dict[str, object]]] = {}
    for duration in EVENT_DURATIONS:
        for issue_index in range(len(train_forecast)):
            valid = train_valid[issue_index][:, wind_indices].all(axis=1)
            valid_window = (
                np.convolve(valid.astype(np.int64), np.ones(duration, dtype=np.int64), mode="valid")
                == duration
            )
            residual = aggregate_residual[issue_index]
            level = _rolling_mean(residual, duration)
            change = residual[duration - 1 :] - residual[: 1 - duration]
            score_sets = (
                -level,
                -change,
                change,
                _rolling_mean(np.abs(residual), duration),
            )
            for type_index, scores in enumerate(score_sets):
                scores = np.where(valid_window, scores, -np.inf)
                if not np.any(np.isfinite(scores)):
                    continue
                start = int(np.argmax(scores))
                score = float(scores[start])
                raw.setdefault((duration, type_index), []).append(
                    {
                        "train_index": issue_index,
                        "source_start": start,
                        "duration": duration,
                        "event_type": type_index,
                        "score": score,
                        "timestamp": target_starts.iloc[issue_index]
                        + pd.Timedelta(hours=start),
                    }
                )

    # Keep only the upper train tail of every morphology/duration, then remove
    # repeated views of the same physical event caused by rolling issue windows.
    for key, candidates in raw.items():
        threshold = float(np.quantile([row["score"] for row in candidates], quantile))
        selected = [row for row in candidates if row["score"] >= threshold and row["score"] > 0]
        selected.sort(key=lambda row: (row["timestamp"], -row["score"]))
        clusters: list[list[dict[str, object]]] = []
        for row in selected:
            if not clusters or row["timestamp"] - clusters[-1][-1]["timestamp"] > pd.Timedelta(hours=6):
                clusters.append([row])
            else:
                clusters[-1].append(row)
        records.extend(max(group, key=lambda row: row["score"]) for group in clusters)
    if not records:
        raise ValueError("no train-only wind event prototypes survived selection")
    records.sort(key=lambda row: (row["duration"], row["event_type"], row["timestamp"]))
    # Normalize event magnitude within morphology/duration so a quota can retain
    # genuinely severe downside prototypes without comparing incomparable raw
    # ramp and level scores.
    for duration in EVENT_DURATIONS:
        for type_index in range(len(EVENT_TYPES)):
            group = [
                row
                for row in records
                if int(row["duration"]) == duration
                and int(row["event_type"]) == type_index
            ]
            if not group:
                continue
            order = sorted(group, key=lambda row: float(row["score"]))
            denominator = max(len(order) - 1, 1)
            for rank, row in enumerate(order):
                row["severity_rank"] = float(rank / denominator)
    return records


def build_discrete_event_arrays(
    data_dir: str | Path,
    split: str,
    top_k: int = 48,
    exclusion_days: int = 6,
    event_quantile: float = 0.75,
    target_stride_hours: int = 3,
    severe_downside_fraction: float = 0.0,
) -> DiscreteEventArrays:
    """Return leakage-safe local event candidates for ``split``.

    Retrieval keys use only the issued forecast available at generation.  Train
    residuals are values, not query features.  For training queries, source
    issues inside the overlap exclusion radius are removed.
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split={split!r}")
    if top_k < 8:
        raise ValueError("discrete event top_k must be at least 8")
    if not 0.5 <= event_quantile < 1.0:
        raise ValueError("event_quantile must be in [0.5,1)")
    if target_stride_hours not in {1, 2, 3, 4, 6, 12}:
        raise ValueError("unsupported target_stride_hours")
    if not 0.0 <= severe_downside_fraction <= 0.5:
        raise ValueError("severe_downside_fraction must be in [0,0.5]")
    data_dir = Path(data_dir)
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    wind_capacity = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
    wind_weight = wind_capacity / wind_capacity.sum()
    train_forecast = np.asarray(
        np.load(data_dir / "train_forecast.npy", mmap_mode="r"), dtype=np.float32
    )
    train_residual = np.asarray(
        np.load(data_dir / "train_residual.npy", mmap_mode="r"), dtype=np.float32
    )
    train_valid = np.asarray(np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")) == 0
    train_issues = pd.read_csv(data_dir / "train_issue_dates.csv")
    query_forecast = np.asarray(
        np.load(data_dir / f"{split}_forecast.npy", mmap_mode="r"), dtype=np.float32
    )
    query_issues = pd.read_csv(data_dir / f"{split}_issue_dates.csv")
    records = _event_candidates(
        train_forecast,
        train_residual,
        train_valid,
        train_issues,
        wind_indices,
        wind_weight,
        event_quantile,
    )
    train_dates = pd.to_datetime(train_issues["issue_date"]).dt.normalize().to_numpy()
    query_dates = pd.to_datetime(query_issues["issue_date"]).dt.normalize().to_numpy()

    bank_keys = []
    for row in records:
        source = int(row["train_index"])
        start = int(row["source_start"])
        duration = int(row["duration"])
        bank_keys.append(
            _local_key(
                train_forecast[source, start : start + duration][:, wind_indices],
                wind_weight,
            )
        )
    bank_keys = np.stack(bank_keys)
    feature_mean = bank_keys.mean(axis=0)
    feature_std = bank_keys.std(axis=0)
    feature_std[feature_std < 1e-5] = 1.0
    bank_keys = np.nan_to_num((bank_keys - feature_mean) / feature_std)

    query_count = len(query_forecast)
    residual = np.zeros((query_count, top_k, STATIONS, HOURS), dtype=np.float32)
    time_mask = np.zeros((query_count, top_k, HOURS), dtype=np.float32)
    distance = np.zeros((query_count, top_k), dtype=np.float32)
    prior_weight = np.zeros((query_count, top_k), dtype=np.float32)
    train_index = np.full((query_count, top_k), -1, dtype=np.int64)
    event_type = np.full((query_count, top_k), -1, dtype=np.int64)
    duration_array = np.zeros((query_count, top_k), dtype=np.int64)
    target_start_array = np.zeros((query_count, top_k), dtype=np.int64)
    source_start_array = np.zeros((query_count, top_k), dtype=np.int64)

    for query_index in range(query_count):
        allowed = np.ones(len(records), dtype=bool)
        if split == "train":
            source_dates = train_dates[[int(row["train_index"]) for row in records]]
            separation = np.abs(
                (source_dates - query_dates[query_index])
                .astype("timedelta64[D]")
                .astype(np.int64)
            )
            allowed &= separation > int(exclusion_days)
        pool: list[tuple[float, int, int]] = []
        for duration in EVENT_DURATIONS:
            record_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(records)
                    if int(row["duration"]) == duration and allowed[index]
                ],
                dtype=np.int64,
            )
            if not len(record_indices):
                continue
            for target_start in range(0, HOURS - duration + 1, target_stride_hours):
                key = _local_key(
                    query_forecast[query_index, target_start : target_start + duration][
                        :, wind_indices
                    ],
                    wind_weight,
                )
                key = np.nan_to_num((key - feature_mean) / feature_std)
                local_distance = np.mean(
                    (bank_keys[record_indices] - key[None]) ** 2, axis=1
                )
                # Retain one nearest source for each event morphology at this
                # possible lead time.  These remain separate candidates.
                for type_index in range(len(EVENT_TYPES)):
                    typed = np.asarray(
                        [
                            pos
                            for pos, record_index in enumerate(record_indices)
                            if int(records[int(record_index)]["event_type"]) == type_index
                        ],
                        dtype=np.int64,
                    )
                    if len(typed):
                        best = int(typed[np.argmin(local_distance[typed])])
                        pool.append(
                            (
                                float(local_distance[best]),
                                int(record_indices[best]),
                                int(target_start),
                            )
                        )
        if len(pool) < top_k:
            raise ValueError(
                f"query {query_index} has only {len(pool)} discrete event candidates"
            )
        pool.sort(key=lambda value: value[0])
        # First retain the best candidate for each lead-day/type/duration cell.
        # This prevents all K candidates collapsing onto one visually similar hour.
        chosen: list[tuple[float, int, int]] = []
        seen_cells: set[tuple[int, int, int]] = set()
        seen_exact: set[tuple[int, int]] = set()
        severe_quota = int(round(top_k * severe_downside_fraction))
        if severe_quota:
            downside = [
                item
                for item in pool
                if int(records[item[1]]["event_type"]) in {0, 1}
            ]
            downside.sort(
                key=lambda item: (
                    -float(records[item[1]].get("severity_rank", 0.0)),
                    float(item[0]),
                )
            )
            severe_cells: set[tuple[int, int, int]] = set()
            for item in downside:
                _, record_index, target_start = item
                row = records[record_index]
                cell = (
                    target_start // 24,
                    int(row["event_type"]),
                    int(row["duration"]),
                )
                exact = (record_index, target_start)
                if cell in severe_cells or exact in seen_exact:
                    continue
                chosen.append(item)
                severe_cells.add(cell)
                seen_cells.add(cell)
                seen_exact.add(exact)
                if len(chosen) == severe_quota:
                    break
        for item in pool:
            _, record_index, target_start = item
            row = records[record_index]
            cell = (
                target_start // 24,
                int(row["event_type"]),
                int(row["duration"]),
            )
            exact = (record_index, target_start)
            if cell not in seen_cells and exact not in seen_exact:
                chosen.append(item)
                seen_cells.add(cell)
                seen_exact.add(exact)
            if len(chosen) == top_k:
                break
        if len(chosen) < top_k:
            for item in pool:
                exact = (item[1], item[2])
                if exact not in seen_exact:
                    chosen.append(item)
                    seen_exact.add(exact)
                if len(chosen) == top_k:
                    break
        selected_distance = np.asarray([item[0] for item in chosen], dtype=np.float64)
        scale = max(float(np.median(selected_distance)), 1e-6)
        logits = -(selected_distance - selected_distance.min()) / (0.5 * scale)
        weights = np.exp(logits - logits.max())
        weights = 0.9 * weights / weights.sum() + 0.1 / top_k
        distance[query_index] = (selected_distance / scale).astype(np.float32)
        prior_weight[query_index] = weights.astype(np.float32)
        for candidate_index, (_, record_index, target_start) in enumerate(chosen):
            row = records[record_index]
            source = int(row["train_index"])
            source_start = int(row["source_start"])
            duration = int(row["duration"])
            patch = train_residual[source, source_start : source_start + duration][
                :, wind_indices
            ].T
            residual[
                query_index,
                candidate_index,
                wind_indices,
                target_start : target_start + duration,
            ] = patch
            time_mask[
                query_index,
                candidate_index,
                target_start : target_start + duration,
            ] = 1.0
            train_index[query_index, candidate_index] = source
            event_type[query_index, candidate_index] = int(row["event_type"])
            duration_array[query_index, candidate_index] = duration
            target_start_array[query_index, candidate_index] = target_start
            source_start_array[query_index, candidate_index] = source_start

    effective_k = 1.0 / np.sum(prior_weight**2, axis=1)
    type_counts = {
        name: int(sum(int(row["event_type"]) == index for row in records))
        for index, name in enumerate(EVENT_TYPES)
    }
    return DiscreteEventArrays(
        residual=residual,
        distance=distance,
        prior_weight=prior_weight,
        train_index=train_index,
        time_mask=time_mask,
        event_type=event_type,
        duration=duration_array,
        target_start=target_start_array,
        source_start=source_start_array,
        audit={
            "method": "train_only_stratified_joint_wind_discrete_event_memory_v2",
            "split": split,
            "query_count": int(query_count),
            "event_bank_count": int(len(records)),
            "event_bank_type_counts": type_counts,
            "durations_hours": list(EVENT_DURATIONS),
            "event_types": list(EVENT_TYPES),
            "event_quantile": float(event_quantile),
            "top_k_candidate_pool": int(top_k),
            "target_stride_hours": int(target_stride_hours),
            "severe_downside_fraction": float(severe_downside_fraction),
            "severe_downside_quota": int(
                round(top_k * severe_downside_fraction)
            ),
            "mean_effective_k": float(effective_k.mean()),
            "selection_contract": "one_discrete_prototype_per_generated_event_member",
            "topk_averaging": False,
            "key_source": "issued_forecast_local_patch_only",
            "value_source": "train_residual_local_joint_wind_patch_only",
            "future_query_actual_used": False,
            "test_target_used": False,
            "train_overlap_exclusion_days": int(exclusion_days),
            "wind_station_indices": wind_indices.tolist(),
        },
    )
