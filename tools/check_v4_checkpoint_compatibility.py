#!/usr/bin/env python
"""Fail-fast compatibility check against a real historical V4 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import build_model, load_model_checkpoint, resolve_architecture


ALLOWED_LEGACY_MISSING = {
    "diffusion.beta",
    "diffusion.alpha",
    "diffusion.alpha_hat",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-used", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.config_used.is_file():
        raise FileNotFoundError(args.config_used)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    with args.config_used.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    architecture = resolve_architecture(config["model"])
    if architecture != "v4_legacy":
        raise ValueError(
            f"compatibility artifact must resolve to v4_legacy, got {architecture!r}"
        )

    device = torch.device("cpu")
    model = build_model(config["model"], device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = load_model_checkpoint(
        model, checkpoint, architecture=architecture
    )
    critical_missing = sorted(set(missing) - ALLOWED_LEGACY_MISSING)
    if critical_missing or unexpected:
        raise RuntimeError(
            "V4 checkpoint incompatibility: "
            f"critical_missing={critical_missing}, unexpected={sorted(unexpected)}"
        )
    print(
        "V4_CHECKPOINT_COMPATIBLE "
        f"checkpoint={args.checkpoint} allowed_missing={sorted(missing)}"
    )


if __name__ == "__main__":
    main()
