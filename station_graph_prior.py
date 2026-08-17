"""Load, freeze, and audit optional Station24 fixed dual-graph priors."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_adjacency(path: Path, station_count: int) -> np.ndarray:
    adjacency = np.load(path).astype(np.float32)
    expected = (station_count, station_count)
    if adjacency.shape != expected:
        raise ValueError(f"adjacency {path} expected {expected}, got {adjacency.shape}")
    if not np.isfinite(adjacency).all():
        raise ValueError(f"adjacency {path} contains non-finite values")
    if np.min(adjacency) < 0:
        raise ValueError(f"adjacency {path} contains negative weights")
    if not np.allclose(adjacency, adjacency.T, atol=1e-6, rtol=0.0):
        raise ValueError(f"adjacency {path} must be symmetric")
    if np.any(adjacency.sum(axis=1) <= 0):
        raise ValueError(f"adjacency {path} has an empty row")
    return adjacency


def prepare_training_graphs(
    data_path: Path,
    run_dir: Path,
    model_config: dict[str, object],
    secondary_override: str | None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object]]:
    """Freeze the exact matrices used by training inside the run directory."""

    station_count = int(model_config.get("station_count", 24))
    primary_source = data_path / "station_adjacency.npy"
    primary = _load_adjacency(primary_source, station_count)
    enabled = bool(model_config.get("use_dual_fixed_graph", False))
    if not enabled:
        return torch.from_numpy(primary), None, {
            "mode": "single_geographic",
            "primary_adjacency_source": str(primary_source),
            "primary_sha256": sha256_file(primary_source),
            "secondary_adjacency_source": None,
            "secondary_sha256": None,
        }

    configured = model_config.get("secondary_adjacency_path")
    selected_source = secondary_override or (str(configured) if configured else None)
    if not selected_source:
        raise ValueError(
            "dual fixed graph requires --secondary-adjacency or "
            "model.secondary_adjacency_path"
        )
    secondary_source = Path(selected_source)
    if not secondary_source.is_file():
        raise FileNotFoundError(f"secondary adjacency not found: {secondary_source}")
    secondary = _load_adjacency(secondary_source, station_count)

    graph_dir = run_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=False)
    frozen_primary = graph_dir / "primary_adjacency.npy"
    frozen_secondary = graph_dir / "secondary_adjacency.npy"
    shutil.copyfile(primary_source, frozen_primary)
    shutil.copyfile(secondary_source, frozen_secondary)
    manifest = {
        "mode": "dual_fixed_shared_projection",
        "primary_adjacency_role": "geographic",
        "primary_adjacency_source": str(primary_source),
        "primary_frozen_path": str(frozen_primary.relative_to(run_dir)),
        "primary_sha256": sha256_file(frozen_primary),
        "secondary_adjacency_role": str(
            model_config.get("secondary_adjacency_role", "unspecified_historical")
        ),
        "secondary_adjacency_source": str(secondary_source),
        "secondary_frozen_path": str(frozen_secondary.relative_to(run_dir)),
        "secondary_sha256": sha256_file(frozen_secondary),
        "fit_split": str(model_config.get("secondary_adjacency_fit_split", "train")),
        "validation_actual_used": False,
        "test_actual_used": False,
    }
    if manifest["fit_split"] != "train":
        raise ValueError("secondary adjacency must be fitted on train")
    (graph_dir / "graph_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_config["secondary_adjacency_path"] = str(
        frozen_secondary.relative_to(run_dir)
    )
    return torch.from_numpy(primary), torch.from_numpy(secondary), manifest


def load_generation_graphs(
    data_path: Path,
    run_dir: Path,
    model_config: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object]]:
    """Reload the frozen training matrices and reject provenance drift."""

    station_count = int(model_config.get("station_count", 24))
    enabled = bool(model_config.get("use_dual_fixed_graph", False))
    if not enabled:
        source = data_path / "station_adjacency.npy"
        return torch.from_numpy(_load_adjacency(source, station_count)), None, {
            "mode": "single_geographic",
            "primary_adjacency_source": str(source),
            "primary_sha256": sha256_file(source),
        }

    graph_dir = run_dir / "graphs"
    manifest_path = graph_dir / "graph_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("dual-graph run lacks graphs/graph_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    primary_path = run_dir / str(manifest["primary_frozen_path"])
    secondary_path = run_dir / str(manifest["secondary_frozen_path"])
    observed = {
        "primary_sha256": sha256_file(primary_path),
        "secondary_sha256": sha256_file(secondary_path),
    }
    for key, value in observed.items():
        if value != manifest.get(key):
            raise ValueError(f"frozen graph hash mismatch for {key}")
    checkpoint_manifest = checkpoint.get("graph_manifest")
    if checkpoint_manifest != manifest:
        raise ValueError("checkpoint graph manifest does not match frozen graph files")
    return (
        torch.from_numpy(_load_adjacency(primary_path, station_count)),
        torch.from_numpy(_load_adjacency(secondary_path, station_count)),
        manifest,
    )
