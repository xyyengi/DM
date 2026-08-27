"""Leakage-safe historical retrieval memory for Station24 experiments.

The retrieval key uses issued wind forecasts only.  Values are complete joint
wind residual trajectories from the training split.  Validation/test targets
are never inspected while neighbours are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HOURS = 168
STATIONS = 24
BLOCK_HOURS = 6
RAMP_LAGS = (1, 3, 6)


def _block_mean(values: np.ndarray, width: int) -> np.ndarray:
    shape = (values.shape[0], values.shape[1] // width, width) + values.shape[2:]
    return values.reshape(shape).mean(axis=2)


def _forecast_features(
    forecast: np.ndarray,
    issue_dates: pd.DataFrame,
    wind_indices: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    """R0 forecast-only key: 6 h node shapes, ramps, daily moments, calendar."""

    wind = np.asarray(forecast[:, :, wind_indices], dtype=np.float64)
    weights = capacities / capacities.sum()
    aggregate = np.einsum("nts,s->nt", wind, weights)
    ramps = []
    for lag in RAMP_LAGS:
        values = np.zeros_like(aggregate)
        values[:, lag:] = aggregate[:, lag:] - aggregate[:, :-lag]
        ramps.append(_block_mean(values[:, :, None], BLOCK_HOURS)[:, :, 0])
    daily = wind.reshape(len(wind), 7, 24, len(wind_indices))
    dates = pd.to_datetime(issue_dates["issue_date"])
    calendar = np.stack(
        [
            np.sin(2 * np.pi * (dates.dt.month.to_numpy() - 1) / 12.0),
            np.cos(2 * np.pi * (dates.dt.month.to_numpy() - 1) / 12.0),
            np.sin(2 * np.pi * dates.dt.dayofweek.to_numpy() / 7.0),
            np.cos(2 * np.pi * dates.dt.dayofweek.to_numpy() / 7.0),
        ],
        axis=1,
    )
    return np.concatenate(
        [
            _block_mean(wind, BLOCK_HOURS).reshape(len(wind), -1),
            *ramps,
            daily.mean(axis=2).reshape(len(wind), -1),
            daily.std(axis=2).reshape(len(wind), -1),
            calendar,
        ],
        axis=1,
    )


@dataclass(frozen=True)
class RetrievalArrays:
    residual: np.ndarray
    distance: np.ndarray
    prior_weight: np.ndarray
    train_index: np.ndarray
    audit: dict[str, object]


def build_retrieval_arrays(
    data_dir: str | Path,
    split: str,
    top_k: int = 40,
    exclusion_days: int = 6,
) -> RetrievalArrays:
    """Build Top-K train-memory tensors for one query split.

    Training queries exclude themselves and overlapping +/- ``exclusion_days``
    issue windows.  Validation and test queries can retrieve from train only.
    """

    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported retrieval split={split!r}")
    data_dir = Path(data_dir)
    stations = pd.read_csv(data_dir / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    wind_indices = stations.index[stations.data_type.eq("wind")].to_numpy(int)
    capacities = stations.loc[wind_indices, "capacity_mw"].to_numpy(float)
    train_forecast = np.load(data_dir / "train_forecast.npy", mmap_mode="r")
    train_residual = np.load(data_dir / "train_residual.npy", mmap_mode="r")
    train_fill = np.load(data_dir / "train_fill_mask.npy", mmap_mode="r")
    train_issues = pd.read_csv(data_dir / "train_issue_dates.csv")
    query_forecast = np.load(data_dir / f"{split}_forecast.npy", mmap_mode="r")
    query_issues = pd.read_csv(data_dir / f"{split}_issue_dates.csv")

    # The new expert is wind-only.  Requiring complete wind values keeps every
    # retrieved trajectory physically coherent without discarding histories for
    # a missing solar value that the expert never consumes.
    bank_indices = np.flatnonzero(
        ~np.any(np.asarray(train_fill[:, :, wind_indices]) != 0, axis=(1, 2))
    )
    if len(bank_indices) <= top_k:
        raise ValueError(
            f"retrieval bank has {len(bank_indices)} complete windows, needs > {top_k}"
        )
    train_features = _forecast_features(
        train_forecast, train_issues, wind_indices, capacities
    )
    query_features = _forecast_features(
        query_forecast, query_issues, wind_indices, capacities
    )
    mean = train_features[bank_indices].mean(axis=0)
    std = train_features[bank_indices].std(axis=0)
    std[std < 1e-6] = 1.0
    bank_standard = np.nan_to_num((train_features - mean) / std)
    query_standard = np.nan_to_num((query_features - mean) / std)
    train_dates = pd.to_datetime(train_issues["issue_date"]).dt.normalize().to_numpy()
    query_dates = pd.to_datetime(query_issues["issue_date"]).dt.normalize().to_numpy()

    chosen_indices: list[np.ndarray] = []
    chosen_distances: list[np.ndarray] = []
    chosen_weights: list[np.ndarray] = []
    for query_index, query in enumerate(query_standard):
        candidates = bank_indices
        if split == "train":
            separation = np.abs(
                (train_dates[candidates] - query_dates[query_index])
                .astype("timedelta64[D]")
                .astype(np.int64)
            )
            candidates = candidates[separation > int(exclusion_days)]
        if len(candidates) < top_k:
            raise ValueError(
                f"query {query_index} has only {len(candidates)} leakage-safe histories"
            )
        difference = bank_standard[candidates] - query[None]
        distance = np.mean(difference * difference, axis=1)
        local = np.argsort(distance, kind="stable")[:top_k]
        selected_distance = distance[local]
        temperature = max(float(np.median(selected_distance)), 1e-6)
        logits = -(selected_distance - selected_distance.min()) / temperature
        weight = np.exp(logits)
        weight /= weight.sum()
        chosen_indices.append(candidates[local])
        chosen_distances.append(selected_distance)
        chosen_weights.append(weight)

    train_index = np.asarray(chosen_indices, dtype=np.int64)
    residual = np.zeros(
        (len(train_index), top_k, STATIONS, HOURS), dtype=np.float32
    )
    # Source arrays are [issue,hour,station]; network tensors are [K,S,L].
    selected = np.asarray(train_residual[train_index], dtype=np.float32)
    selected = selected.transpose(0, 1, 3, 2)
    residual[:, :, wind_indices, :] = selected[:, :, wind_indices, :]
    distance = np.asarray(chosen_distances, dtype=np.float32)
    distance /= np.maximum(np.median(distance, axis=1, keepdims=True), 1e-6)
    prior_weight = np.asarray(chosen_weights, dtype=np.float32)
    return RetrievalArrays(
        residual=residual,
        distance=distance,
        prior_weight=prior_weight,
        train_index=train_index,
        audit={
            "method": "forecast_only_topk_joint_wind_residual_memory_v1",
            "split": split,
            "query_count": int(len(train_index)),
            "train_bank_count": int(len(bank_indices)),
            "top_k": int(top_k),
            "exclusion_days_for_train": int(exclusion_days),
            "key_source": "issued_forecast_only",
            "value_source": "train_residual_only",
            "future_query_actual_used": False,
            "test_target_used": False,
            "wind_station_indices": wind_indices.tolist(),
        },
    )
