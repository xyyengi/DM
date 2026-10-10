"""Train-only PCA contract for the Shandong91 low-rank V3 experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from datasets.shandong91_reliable import CHANNELS, _load_scales


FACTOR_COUNTS = {"Wind": 5, "Solar": 1, "Load": 1}


def _timestamps(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return np.asarray([row["timestamp"] for row in csv.DictReader(stream)])


def _canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def factor_sha256(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "sha256"}
    return hashlib.sha256(_canonical(payload)).hexdigest()


def fit_train_only_pca(
    root: str | Path,
    factor_counts: Mapping[str, int] = FACTOR_COUNTS,
) -> dict[str, Any]:
    """Fit deterministic MW-space PCA using unique hourly training rows only."""
    root = Path(root)
    metadata = json.loads((root / "preprocessing_metadata.json").read_text("utf-8"))
    train_end = metadata["splits"]["train"]["split_end"]
    times = _timestamps(root / "hourly/timestamps.csv")
    train_rows = times <= train_end
    residual = np.load(root / "hourly/residual_mw.npy", mmap_mode="r")
    valid = np.load(root / "hourly/residual_valid_mask.npy", mmap_mode="r").astype(bool)
    node_type = np.load(root / "static/node_type_mask.npy").astype(bool)
    train_enabled = np.load(root / "static/channel_train_mask.npy").astype(bool)
    scales = _load_scales(root).astype(np.float64)
    resources: dict[str, Any] = {}
    for channel, name in enumerate(CHANNELS):
        active = node_type[:, channel] & train_enabled[:, channel]
        nodes = np.flatnonzero(active)
        complete = train_rows & valid[:, nodes, channel].all(axis=1)
        values = np.asarray(residual[complete][:, nodes, channel], dtype=np.float64)
        if values.shape[0] < 2:
            raise ValueError(f"insufficient complete training rows for {name}")
        mean = values.mean(axis=0)
        centered = values - mean
        _, singular, components = np.linalg.svd(centered, full_matrices=False)
        count = int(factor_counts[name])
        components = components[:count].copy()
        singular = singular[:count].copy()
        for index in range(count):
            pivot = int(np.argmax(np.abs(components[index])))
            if components[index, pivot] < 0:
                components[index] *= -1
        factor_std = singular / np.sqrt(values.shape[0] - 1)
        scores = centered @ components.T / factor_std
        reconstruction = (scores * factor_std) @ components
        total_variance = np.square(centered).sum()
        explained = np.square(singular).sum() / max(total_variance, np.finfo(float).eps)
        normalized_mean = mean / scales[nodes, channel]
        normalized_loadings = (
            factor_std[:, None] * components / scales[nodes, channel][None]
        )
        error = centered - reconstruction
        resources[name] = {
            "channel": channel,
            "factor_count": count,
            "active_nodes_zero_based": nodes.tolist(),
            "complete_train_rows": int(complete.sum()),
            "train_start": str(times[train_rows][0]),
            "train_end": str(times[train_rows][-1]),
            "mw_mean": mean.tolist(),
            "mw_components": components.tolist(),
            "factor_std_mw": factor_std.tolist(),
            "normalized_mean": normalized_mean.tolist(),
            "normalized_loadings": normalized_loadings.tolist(),
            "explained_variance_ratio_cumulative": float(explained),
            "train_reconstruction_rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
            "train_reconstruction_max_abs_mw": float(np.max(np.abs(error))),
        }
    document: dict[str, Any] = {
        "version": "shandong91_train_only_pca_v1",
        "data_version": metadata.get("version"),
        "fit_scope": "unique hourly rows with timestamp <= train split_end; complete active-node rows per resource",
        "factor_order": ["Wind_PC1", "Wind_PC2", "Wind_PC3", "Wind_PC4", "Wind_PC5", "Solar_PC1", "Load_PC1"],
        "resources": resources,
        "load_contract": "strict rank-1 centered reconstruction; no Load local latent",
        "solar_80_83_policy": "retained separately; duplicate-source mapping risk unresolved",
    }
    document["sha256"] = factor_sha256(document)
    return document


class FixedPCAFactorTransform(nn.Module):
    """Mask-aware fixed PCA decomposition in normalized residual coordinates."""

    def __init__(self, document: Mapping[str, Any], ridge: float = 1e-10):
        super().__init__()
        if factor_sha256(document) != document.get("sha256"):
            raise ValueError("PCA factor document hash mismatch")
        means = torch.zeros(91, 3)
        rows: list[torch.Tensor] = []
        resources: list[int] = []
        active = torch.zeros(91, 3, dtype=torch.bool)
        slices: dict[str, tuple[int, int]] = {}
        cursor = 0
        for channel, name in enumerate(CHANNELS):
            item = document["resources"][name]
            nodes = torch.tensor(item["active_nodes_zero_based"], dtype=torch.long)
            active[nodes, channel] = True
            means[nodes, channel] = torch.tensor(item["normalized_mean"], dtype=torch.float32)
            loading = torch.tensor(item["normalized_loadings"], dtype=torch.float32)
            for row in loading:
                full = torch.zeros(91, 3)
                full[nodes, channel] = row
                rows.append(full)
                resources.append(channel)
            slices[name] = (cursor, cursor + loading.shape[0])
            cursor += loading.shape[0]
        self.register_buffer("mean", means)
        self.register_buffer("basis", torch.stack(rows))
        self.register_buffer("factor_resource", torch.tensor(resources, dtype=torch.long))
        self.register_buffer("active", active)
        self.ridge = float(ridge)
        self.slices = slices
        self.document_sha256 = str(document["sha256"])

    @property
    def factor_count(self) -> int:
        return int(self.basis.shape[0])

    def _solve(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length = values.shape[:2]
        # CUDA autocast can supply FP16 denoiser outputs while the fixed PCA
        # buffers remain FP32.  Least-squares solves are both unsupported and
        # numerically unsafe in FP16, so keep this small K<=5 operation in
        # FP32.  The casts remain differentiable back to the AMP backbone.
        solve_dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        output = torch.zeros(
            batch, length, self.factor_count,
            device=values.device, dtype=solve_dtype,
        )
        for channel, name in enumerate(CHANNELS):
            start, stop = self.slices[name]
            basis = self.basis[start:stop, :, channel].to(solve_dtype)
            weight = mask[..., channel].to(solve_dtype)
            current = values[..., channel].to(solve_dtype) * weight
            gram = torch.einsum("btn,kn,ln->btkl", weight, basis, basis)
            rhs = torch.einsum("btn,kn->btk", current, basis)
            eye = torch.eye(stop - start, device=values.device, dtype=solve_dtype)
            output[..., start:stop] = torch.linalg.solve(
                gram + self.ridge * eye, rhs.unsqueeze(-1)
            ).squeeze(-1)
        return output

    def common(self, factors: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btk,knr->btnr", factors, self.basis)

    def decompose(self, residual: torch.Tensor, mask: torch.Tensor):
        centered = (residual - self.mean) * mask.to(residual.dtype)
        factors = self._solve(centered, mask)
        common = self.common(factors)
        local = (centered - common) * mask.to(residual.dtype)
        local[..., 2] = 0
        local = self.project_local(local, mask)
        return factors, local

    def project_factor(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._solve(values * mask.to(values.dtype), mask)

    def project_local(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        projected = values - self.common(self._solve(values, mask))
        projected = projected * mask.to(values.dtype)
        projected[..., 2] = 0
        return projected

    def reconstruct(self, factors: torch.Tensor, local: torch.Tensor,
                    mask: torch.Tensor, *, include_mean: bool = True) -> torch.Tensor:
        value = self.common(factors) + local
        if include_mean:
            value = value + self.mean
        return value * mask.to(value.dtype)
