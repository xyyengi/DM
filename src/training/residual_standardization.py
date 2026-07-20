"""Leak-free channel-wise standardization for diffusion residual targets."""

from __future__ import annotations

from typing import Mapping

import numpy as np


CHANNEL_ORDER = ("wind", "solar", "load")
SUPPORTED_RESIDUAL_DEFINITIONS = {"forecast_minus_actual", "actual_minus_forecast"}


def to_internal_residual(values: np.ndarray, residual_definition: str) -> np.ndarray:
    """Return residuals in the project convention: forecast - actual."""
    if residual_definition not in SUPPORTED_RESIDUAL_DEFINITIONS:
        raise ValueError(f"Unsupported residual_definition={residual_definition!r}")
    result = np.asarray(values, dtype=np.float64)
    return -result if residual_definition == "actual_minus_forecast" else result


def recover_stride_one_unique_hours(windows: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """Recover one continuous series from stride-one overlapping windows.

    The overlap is verified before fitting statistics so duplicated hours cannot
    silently receive extra weight.
    """
    values = np.asarray(windows)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"Expected [windows,time,channels], got {values.shape}")
    if values.shape[0] > 1 and not np.allclose(
        values[:-1, 1:, :], values[1:, :-1, :], atol=atol, rtol=0.0
    ):
        raise ValueError("Training residual windows are not consistent stride-one overlaps")
    return np.concatenate([values[0], values[1:, -1, :]], axis=0)


def fit_residual_standardizer(
    train_residual_windows: np.ndarray,
    residual_definition: str,
    normalization_divisors: np.ndarray,
    epsilon: float = 1e-6,
) -> dict:
    """Fit three-channel statistics on train unique hours only."""
    raw = np.asarray(train_residual_windows)[..., :3]
    internal = to_internal_residual(raw, residual_definition)
    divisors = np.asarray(normalization_divisors, dtype=np.float64)
    if divisors.shape != (3,) or not np.isfinite(divisors).all() or np.any(divisors <= 0):
        raise ValueError(f"Invalid normalization_divisors={divisors}")
    unique = recover_stride_one_unique_hours(internal) / divisors.reshape(1, 3)
    mean = np.mean(unique, axis=0)
    std = np.std(unique, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < epsilon):
        raise ValueError(f"Invalid fitted residual statistics mean={mean}, std={std}")
    return {
        "enabled": True,
        "fit_split": "train",
        "fit_scope": "unique_stride_one_hours",
        "n_unique_hours": int(unique.shape[0]),
        "channel_order": list(CHANNEL_ORDER),
        "residual_definition": "forecast_minus_actual",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epsilon": float(epsilon),
    }


def validate_standardizer(stats: Mapping[str, object]) -> dict:
    """Validate and normalize a serialized residual standardizer."""
    if not bool(stats.get("enabled", False)):
        raise ValueError("Residual standardizer must have enabled=true")
    mean = np.asarray(stats.get("mean"), dtype=np.float64)
    std = np.asarray(stats.get("std"), dtype=np.float64)
    epsilon = float(stats.get("epsilon", 1e-6))
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError(f"Residual standardizer expects three channels, got mean={mean}, std={std}")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < epsilon):
        raise ValueError(f"Invalid residual standardizer mean={mean}, std={std}")
    channel_order = tuple(stats.get("channel_order", CHANNEL_ORDER))
    if channel_order != CHANNEL_ORDER:
        raise ValueError(f"Unexpected residual channel order {channel_order}")
    return {
        **dict(stats),
        "enabled": True,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "epsilon": epsilon,
        "channel_order": list(CHANNEL_ORDER),
        "residual_definition": "forecast_minus_actual",
    }


def standardize_residual(values: np.ndarray, stats: Mapping[str, object], channel_axis: int) -> np.ndarray:
    checked = validate_standardizer(stats)
    mean = np.asarray(checked["mean"], dtype=np.float64)
    std = np.asarray(checked["std"], dtype=np.float64)
    shape = [1] * np.ndim(values)
    shape[channel_axis] = 3
    return (np.asarray(values) - mean.reshape(shape)) / std.reshape(shape)


def inverse_standardize_residual(values: np.ndarray, stats: Mapping[str, object], channel_axis: int) -> np.ndarray:
    checked = validate_standardizer(stats)
    mean = np.asarray(checked["mean"], dtype=np.float64)
    std = np.asarray(checked["std"], dtype=np.float64)
    shape = [1] * np.ndim(values)
    shape[channel_axis] = 3
    return np.asarray(values) * std.reshape(shape) + mean.reshape(shape)
