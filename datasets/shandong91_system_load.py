"""Train-only system-Load representation for the Shandong91 Route-A probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


LOAD = 2


def _sha256(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def fit_system_load_contract(root: str | Path) -> dict:
    """Fit fixed node shares and system scale from unique hourly train rows only."""
    root = Path(root)
    meta = json.loads((root / "preprocessing_metadata.json").read_text("utf-8"))
    import csv
    with (root / "hourly/timestamps.csv").open(encoding="utf-8-sig", newline="") as f:
        times = np.asarray([row["timestamp"] for row in csv.DictReader(f)])
    train = times <= meta["splits"]["train"]["split_end"]
    actual = np.asarray(np.load(root / "hourly/actual_mw.npy", mmap_mode="r")[:, :, LOAD])
    forecast = np.asarray(np.load(root / "hourly/forecast_mw.npy", mmap_mode="r")[:, :, LOAD])
    valid_a = np.asarray(np.load(root / "hourly/actual_valid_mask.npy", mmap_mode="r")[:, :, LOAD], bool)
    valid_f = np.asarray(np.load(root / "hourly/forecast_valid_mask.npy", mmap_mode="r")[:, :, LOAD], bool)
    node_type = np.load(root / "static/node_type_mask.npy").astype(bool)[:, LOAD]
    enabled = np.load(root / "static/channel_train_mask.npy").astype(bool)[:, LOAD]
    active = node_type & enabled
    joint = train[:, None] & valid_a & valid_f & active[None]
    complete = joint[:, active].all(axis=1)
    if complete.sum() < 2:
        raise ValueError("insufficient complete train-only Load rows")
    train_actual = actual[complete][:, active]
    system_actual = train_actual.sum(axis=1)
    if np.any(system_actual <= 0):
        raise ValueError("system Load must be positive when fitting fixed shares")
    shares_matrix = train_actual / system_actual[:, None]
    shares = np.median(shares_matrix, axis=0)
    shares /= shares.sum()
    reconstructed = system_actual[:, None] * shares[None]
    abs_error = np.abs(reconstructed - train_actual)
    system_forecast = forecast[complete][:, active].sum(axis=1)
    scale = float(np.quantile(system_actual, 0.995))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid train-only system Load scale")
    full_shares = np.zeros(91, np.float64); full_shares[active] = shares
    document = {
        "version": "shandong91_system_load_train_only_v1",
        "fit_split": "train_unique_hourly_complete_cases",
        "active_node_indices_zero_based": np.flatnonzero(active).tolist(),
        "active_node_ids": (np.flatnonzero(active) + 1).tolist(),
        "coefficients": full_shares.tolist(),
        "coefficient_sum": float(full_shares.sum()),
        "complete_train_hours": int(complete.sum()),
        "system_scale_mw": scale,
        "scale_rule": "train-only q99.5 of system actual MW",
        "residual_definition": "system_actual - system_forecast",
        "node_reconstruction_max_abs_mw": float(abs_error.max()),
        "node_reconstruction_rmse_mw": float(np.sqrt(np.mean(abs_error ** 2))),
        "node_reconstruction_max_relative_to_system": float(abs_error.max() / np.max(system_actual)),
        "system_forecast_finite": bool(np.isfinite(system_forecast).all()),
        "source": "existing 77-node Load arrays; mathematical structure only, not proof of business provenance",
    }
    document["sha256"] = _sha256(document)
    return document


class SystemLoadTransform:
    """Collapse Load to one normalized coordinate and map it back by fixed shares."""
    def __init__(self, contract: dict, node_scales: torch.Tensor):
        self.contract = contract
        self.shares = torch.tensor(contract["coefficients"], dtype=torch.float32)
        self.system_scale = float(contract["system_scale_mw"])
        self.node_scales = node_scales[:, LOAD].float()
        self.active = self.shares > 0

    def collapse(self, node_normalized: torch.Tensor) -> torch.Tensor:
        mw = node_normalized[..., LOAD] * self.node_scales.to(node_normalized)
        return mw[..., self.active.to(mw.device)].sum(dim=-1) / self.system_scale

    def expand(self, system_normalized: torch.Tensor) -> torch.Tensor:
        mw = system_normalized[..., None] * self.system_scale * self.shares.to(system_normalized)
        result = torch.zeros_like(mw)
        active = self.active.to(mw.device)
        result[..., active] = mw[..., active] / self.node_scales.to(mw.device)[active]
        return result

