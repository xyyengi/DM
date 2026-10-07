"""Isolated loader and masked objectives for the Shandong 91-node data."""

from __future__ import annotations

import hashlib
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = ("Wind", "Solar", "Load")
SPLITS = ("train", "validation", "test")
EXPECTED_SHAPES = {
    "train": (298, 168, 91, 3),
    "validation": (24, 168, 91, 3),
    "test": (25, 168, 91, 3),
}
WINDOW_ARRAYS = (
    "actual", "forecast", "residual", "actual_valid_mask",
    "forecast_valid_mask", "residual_valid_mask", "time_mark",
    "window_position",
)


def validate_data_contract(root: str | Path, verify_hashes: bool = False) -> dict[str, Any]:
    """Validate the published quality gate and manifest metadata."""
    root = Path(root)
    quality = json.loads((root / "quality_report.json").read_text(encoding="utf-8"))
    checks = quality.get("checks", [])
    if not quality.get("all_checks_passed") or len(checks) != 20 or not all(
        check.get("passed") for check in checks
    ):
        raise ValueError("quality_report.json does not contain 20 passing checks")
    manifest_document = json.loads(
        (root / "output_manifest.json").read_text(encoding="utf-8")
    )
    manifest = manifest_document.get("files", manifest_document)
    split_summary = json.loads((root / "split_summary.json").read_text(encoding="utf-8"))
    required = [
        "preprocessing_metadata.json", "normalization_params.json",
        "split_summary.json", "quality_report.json", "output_manifest.json",
        "static/node_order.csv", "static/node_features.npy",
        "static/node_type_mask.npy", "static/channel_train_mask.npy",
        "static/adjacency_binary_no_self_loop.npy",
        "static/adjacency_binary_with_self_loop.npy", "static/edge_index.npy",
        "static/edge_features.npy", "static/node_feature_columns.json",
    ]
    for split in SPLITS:
        required.extend(f"windows168/{split}/{name}.npy" for name in WINDOW_ARRAYS)
        required.append(f"windows168/{split}/window_starts.csv")
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Shandong91 artifacts missing: {missing}")
    with (root / "static/node_order.csv").open(encoding="utf-8-sig", newline="") as stream:
        node_ids = [int(row["node_id"]) for row in csv.DictReader(stream)]
    if node_ids != list(range(1, 92)):
        raise ValueError("node_order.csv must remain exactly node_id 1..91")
    static_shapes = {
        "static/node_features.npy": (91, 10),
        "static/node_type_mask.npy": (91, 3),
        "static/channel_train_mask.npy": (91, 3),
        "static/adjacency_binary_no_self_loop.npy": (91, 91),
        "static/adjacency_binary_with_self_loop.npy": (91, 91),
        "static/edge_index.npy": (2, 140),
        "static/edge_features.npy": (140, 8),
    }
    for relative, expected in static_shapes.items():
        array = np.load(root / relative, mmap_mode="r")
        entry = manifest.get(relative)
        if tuple(array.shape) != expected or entry is None:
            raise ValueError(f"manifest/shape mismatch for {relative}")
        if list(array.shape) != entry.get("shape") or str(array.dtype) != entry.get("dtype"):
            raise ValueError(f"manifest metadata mismatch for {relative}")
    for split, expected in EXPECTED_SHAPES.items():
        if tuple(split_summary[split]["array_shapes"]["actual"]) != expected:
            raise ValueError(f"unexpected {split} shape in split_summary.json")
        for name in ("actual", "forecast", "residual", "actual_valid_mask",
                     "forecast_valid_mask", "residual_valid_mask"):
            relative = f"windows168/{split}/{name}.npy"
            array = np.load(root / relative, mmap_mode="r")
            entry = manifest.get(relative)
            if tuple(array.shape) != expected or entry is None:
                raise ValueError(f"manifest/shape mismatch for {relative}")
            if list(array.shape) != entry.get("shape") or str(array.dtype) != entry.get("dtype"):
                raise ValueError(f"manifest metadata mismatch for {relative}")
    if verify_hashes:
        for relative, entry in manifest.items():
            if "sha256" not in entry:
                continue
            digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {relative}")
    return {"quality_checks": 20, "splits": EXPECTED_SHAPES.copy()}


def _load_scales(root: Path) -> np.ndarray:
    document = json.loads((root / "normalization_params.json").read_text(encoding="utf-8"))
    scales = np.zeros((91, 3), dtype=np.float32)
    channel_index = {name: index for index, name in enumerate(CHANNELS)}
    for record in document["records"]:
        scale = record.get("scale")
        if scale is not None:
            scales[int(record["node_id"]) - 1, channel_index[record["resource"]]] = scale
    return scales


class Shandong91ReliableDataset(Dataset):
    """Memory-mapped 168-hour samples retaining node/channel semantics."""

    def __init__(self, root: str | Path, split: str, validate: bool = True) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        if validate:
            validate_data_contract(self.root)
        split_dir = self.root / "windows168" / split
        self.arrays = {
            name: np.load(split_dir / f"{name}.npy", mmap_mode="r")
            for name in WINDOW_ARRAYS
        }
        self.node_features = torch.from_numpy(
            np.load(self.root / "static/node_features.npy").astype(np.float32)
        )
        self.node_type_mask = torch.from_numpy(
            np.load(self.root / "static/node_type_mask.npy").astype(bool)
        )
        self.channel_train_mask = torch.from_numpy(
            np.load(self.root / "static/channel_train_mask.npy").astype(bool)
        )
        self.adjacency_no_self = torch.from_numpy(
            np.load(self.root / "static/adjacency_binary_no_self_loop.npy").astype(bool)
        )
        self.adjacency_with_self = torch.from_numpy(
            np.load(self.root / "static/adjacency_binary_with_self_loop.npy").astype(bool)
        )
        self.edge_index = torch.from_numpy(np.load(self.root / "static/edge_index.npy"))
        self.edge_features = torch.from_numpy(
            np.load(self.root / "static/edge_features.npy").astype(np.float32)
        )
        self.scales = torch.from_numpy(_load_scales(self.root))

    def __len__(self) -> int:
        return int(self.arrays["actual"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        for name, array in self.arrays.items():
            value = np.array(array[index], copy=True)
            sample[name] = torch.from_numpy(value)
        valid = sample["residual_valid_mask"].bool()
        effective = valid & self.channel_train_mask.unsqueeze(0) & self.node_type_mask.unsqueeze(0)
        sample["residual_valid_mask"] = valid
        sample["actual_valid_mask"] = sample["actual_valid_mask"].bool()
        sample["forecast_valid_mask"] = sample["forecast_valid_mask"].bool()
        sample["effective_mask"] = effective
        return sample

    def static_inputs(self) -> Mapping[str, torch.Tensor]:
        return {
            "node_features": self.node_features,
            "node_type_mask": self.node_type_mask,
            "channel_train_mask": self.channel_train_mask,
            "adjacency_no_self": self.adjacency_no_self,
            "adjacency_with_self": self.adjacency_with_self,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
        }

    def denormalize_mw(self, values: torch.Tensor) -> torch.Tensor:
        """Apply node/channel scales without clipping normalized values."""
        return values * self.scales.to(device=values.device, dtype=values.dtype)


def effective_mask(
    residual_valid_mask: torch.Tensor,
    channel_train_mask: torch.Tensor,
    node_type_mask: torch.Tensor,
) -> torch.Tensor:
    return (
        residual_valid_mask.bool()
        & channel_train_mask.bool().view(*([1] * (residual_valid_mask.ndim - 2)), 91, 3)
        & node_type_mask.bool().view(*([1] * (residual_valid_mask.ndim - 2)), 91, 3)
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((prediction - target).square(), mask)


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_mean((prediction - target).abs(), mask)

