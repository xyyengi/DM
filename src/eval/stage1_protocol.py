"""Shared protocol checks for paired V5 Stage-1 experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ALLOWED_STAGE1_TRAINING_SEEDS = (2026, 2027)
STAGE1_CONFIG_TEMPLATE_SEED = 2026
STAGE1_VALIDATION_SEED = 314159
STAGE1_GENERATION_SEED = 424242
STAGE1_ENSEMBLE_SIZE = 20


def failed_checks(checks: Mapping[str, bool]) -> list[str]:
    return [name for name, passed in checks.items() if not passed]


def training_protocol_checks(
    config: Mapping[str, Any],
    expected_architecture: str,
    runtime_seed: int,
    runtime_batch_size: int,
    runtime_epochs: int,
    runtime_patience: int,
) -> dict[str, bool]:
    """Validate the immutable template plus the explicit runtime seed override."""
    model = config["model"]
    target = config["target"]
    sampling = config["sampling"]
    train = config["train"]
    return {
        "architecture": (
            model.get("architecture", "v4_legacy") == expected_architecture
        ),
        "length_168": int(config["data"]["length"]) == 168,
        "residual_target": target["type"] == "residual",
        "residual_standardization": bool(
            target["residual_standardization"]["enabled"]
        ),
        "num_steps_500": int(model["num_steps"]) == 500,
        "linear_schedule": model["schedule"] == "linear",
        "posterior_variance": (
            sampling["reverse_variance_type"] == "posterior"
        ),
        "validation_seed": (
            int(train["validation_seed"]) == STAGE1_VALIDATION_SEED
        ),
        "top_k_three": int(train["top_k_checkpoints"]) == 3,
        "config_template_seed": (
            int(train["seed"]) == STAGE1_CONFIG_TEMPLATE_SEED
        ),
        "runtime_seed_allowed": (
            int(runtime_seed) in ALLOWED_STAGE1_TRAINING_SEEDS
        ),
        "runtime_batch_positive": int(runtime_batch_size) > 0,
        "runtime_epochs_positive": int(runtime_epochs) > 0,
        "runtime_patience_positive": int(runtime_patience) > 0,
    }


def validation_protocol_checks(
    config: Mapping[str, Any],
) -> dict[str, bool]:
    """Validate the saved run config before top-3 scenario generation."""
    train = config["train"]
    evaluation = config.get("evaluation", {})
    return {
        "training_seed_allowed": (
            int(train["seed"]) in ALLOWED_STAGE1_TRAINING_SEEDS
        ),
        "validation_seed": (
            int(train["validation_seed"]) == STAGE1_VALIDATION_SEED
        ),
        "top_k": int(train["top_k_checkpoints"]) == 3,
        "ensemble": (
            int(evaluation.get("n_samples", STAGE1_ENSEMBLE_SIZE))
            == STAGE1_ENSEMBLE_SIZE
        ),
        "generation_seed": (
            int(evaluation.get("generation_seed", STAGE1_GENERATION_SEED))
            == STAGE1_GENERATION_SEED
        ),
        "posterior": config["sampling"]["reverse_variance_type"] == "posterior",
    }
