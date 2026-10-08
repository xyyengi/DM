"""Causal V2 conditions for the Shandong91 faithful-24 Raw Body.

The base data contract is never changed.  Forecast-state thresholds are fitted
from unique hourly training actuals only.  Future node-state inputs are computed
from forecast only; recent errors use only hours strictly before window start.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.shandong91_reliable import Shandong91ReliableDataset


def _timestamps(path: Path, column: str) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row[column] for row in csv.DictReader(stream)]


def fit_faithful24_state_thresholds(
    root: str | Path,
    *,
    low_quantile: float = 0.20,
    high_quantile: float = 0.90,
    ramp_quantile: float = 0.90,
    ramp_lags: tuple[int, ...] = (3, 6),
    epsilon: float = 1e-4,
) -> dict[str, Any]:
    root = Path(root)
    if not 0 < low_quantile < high_quantile < 1:
        raise ValueError("state quantiles must satisfy 0 < low < high < 1")
    actual = np.load(root / "hourly/actual_normalized.npy", mmap_mode="r")
    valid = np.load(root / "hourly/actual_valid_mask.npy", mmap_mode="r").astype(bool)
    node_type = np.load(root / "static/node_type_mask.npy").astype(bool)
    train_enabled = np.load(root / "static/channel_train_mask.npy").astype(bool)
    metadata = json.loads((root / "preprocessing_metadata.json").read_text("utf-8"))
    train_end = metadata["splits"]["train"]["split_end"]
    times = np.asarray(_timestamps(root / "hourly/timestamps.csv", "timestamp"))
    train_rows = times <= train_end
    active = node_type & train_enabled
    low = np.zeros((91, 3), dtype=np.float32)
    high = np.ones((91, 3), dtype=np.float32)
    counts = np.zeros((91, 3), dtype=np.int64)
    ramp_scales = {str(lag): np.ones((91, 3), dtype=np.float32) for lag in ramp_lags}
    for node in range(91):
        for resource in range(3):
            if not active[node, resource]:
                continue
            mask = train_rows & valid[:, node, resource]
            values = np.asarray(actual[:, node, resource], dtype=np.float64)
            selected = values[mask]
            if selected.size < 2:
                raise ValueError(f"insufficient state data node={node + 1} resource={resource}")
            low[node, resource] = np.quantile(selected, low_quantile)
            high[node, resource] = np.quantile(selected, high_quantile)
            if high[node, resource] - low[node, resource] < epsilon:
                high[node, resource] = low[node, resource] + epsilon
            counts[node, resource] = selected.size
            for lag in ramp_lags:
                pair = mask[lag:] & mask[:-lag]
                differences = np.abs(values[lag:] - values[:-lag])[pair]
                ramp_scales[str(lag)][node, resource] = max(
                    float(np.quantile(differences, ramp_quantile)), epsilon
                )
    return {
        "method": "train_unique_hourly_actual_quantiles_per_node_resource",
        "fit_split": "train",
        "future_state_source": "forecast_only",
        "future_actual_used_as_condition": False,
        "lead_condition": "disabled_no_forecast_horizon_semantics",
        "low_quantile": low_quantile,
        "high_quantile": high_quantile,
        "ramp_quantile": ramp_quantile,
        "ramp_lags": list(ramp_lags),
        "epsilon": epsilon,
        "low_threshold": low,
        "high_threshold": high,
        "ramp_abs_scale": ramp_scales,
        "train_valid_count": counts,
    }


def threshold_document(thresholds: Mapping[str, Any]) -> dict[str, Any]:
    def serializable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {name: serializable(item) for name, item in value.items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value
    return {key: serializable(item) for key, item in thresholds.items()}


def threshold_sha256(thresholds: Mapping[str, Any]) -> str:
    payload = json.dumps(threshold_document(thresholds), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


class Shandong91Faithful24Dataset(Dataset):
    """V2 view adding causal recent-error and forecast-state conditions."""

    def __init__(self, root: str | Path, split: str,
                 thresholds: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.base = Shandong91ReliableDataset(root, split)
        self.split = split
        self.thresholds = dict(thresholds or fit_faithful24_state_thresholds(root))
        self.window_starts = _timestamps(
            self.root / "windows168" / split / "window_starts.csv", "window_start"
        )
        hourly_times = _timestamps(self.root / "hourly/timestamps.csv", "timestamp")
        self.hour_index = {value: index for index, value in enumerate(hourly_times)}
        self.hourly_residual = np.load(
            self.root / "hourly/residual_normalized.npy", mmap_mode="r"
        )
        self.hourly_valid = np.load(
            self.root / "hourly/residual_valid_mask.npy", mmap_mode="r"
        ).astype(bool)
        self.state_clip = 3.0

    def __len__(self) -> int:
        return len(self.base)

    @property
    def node_features(self): return self.base.node_features
    @property
    def node_type_mask(self): return self.base.node_type_mask
    @property
    def channel_train_mask(self): return self.base.channel_train_mask
    @property
    def adjacency_with_self(self): return self.base.adjacency_with_self
    @property
    def scales(self): return self.base.scales

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        forecast = sample["forecast"].numpy().astype(np.float32, copy=False)
        forecast_valid = sample["forecast_valid_mask"].numpy().astype(bool, copy=False)
        active = (self.base.node_type_mask & self.base.channel_train_mask).numpy()
        low = np.asarray(self.thresholds["low_threshold"], dtype=np.float32)
        high = np.asarray(self.thresholds["high_threshold"], dtype=np.float32)
        epsilon = float(self.thresholds["epsilon"])
        state = np.zeros((168, 91, 3, 4), dtype=np.float32)
        valid = forecast_valid & active[None]
        low_scale = np.maximum(np.abs(low), epsilon)
        high_scale = np.maximum(1.0 - high, epsilon)
        state[..., 0] = np.maximum(0.0, (low[None] - forecast) / low_scale[None])
        state[..., 1] = np.maximum(0.0, (forecast - high[None]) / high_scale[None])
        state[..., :2] *= valid[..., None]
        for lag in self.thresholds["ramp_lags"]:
            lag = int(lag)
            ramp = forecast[lag:] - forecast[:-lag]
            pair = valid[lag:] & valid[:-lag]
            scale = np.asarray(self.thresholds["ramp_abs_scale"][str(lag)], dtype=np.float32)
            state[lag:, ..., 2] = np.maximum(
                state[lag:, ..., 2], np.where(pair, np.maximum(ramp, 0.0) / scale, 0.0)
            )
            state[lag:, ..., 3] = np.maximum(
                state[lag:, ..., 3], np.where(pair, np.maximum(-ramp, 0.0) / scale, 0.0)
            )
        np.clip(state, 0.0, self.state_clip, out=state)
        sample["node_state"] = torch.from_numpy(state.reshape(168, 91, 12))

        recent = np.zeros((24, 91, 3), dtype=np.float32)
        recent_valid = np.zeros((24, 91, 3), dtype=bool)
        start = self.hour_index[self.window_starts[index]]
        if start >= 24:
            recent[:] = np.asarray(self.hourly_residual[start - 24:start], dtype=np.float32)
            recent_valid[:] = np.asarray(self.hourly_valid[start - 24:start], dtype=bool)
            recent_valid &= active[None]
            recent *= recent_valid
        sample["recent_error"] = torch.from_numpy(recent)
        sample["recent_error_valid_mask"] = torch.from_numpy(recent_valid)
        return sample

    def denormalize_mw(self, values: torch.Tensor) -> torch.Tensor:
        return self.base.denormalize_mw(values)

    def condition_manifest(self) -> dict[str, Any]:
        return {
            "state_threshold_sha256": threshold_sha256(self.thresholds),
            "state_thresholds": threshold_document(self.thresholds),
            "recent_error": "24 hours strictly before window start; actual-forecast residual",
            "recent_error_future_actual_used": False,
            "lead": "disabled",
        }
