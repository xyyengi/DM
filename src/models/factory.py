"""Explicit model factory with a compatibility-only default for legacy configs."""

from __future__ import annotations

from typing import Mapping


LEGACY_ARCHITECTURE = "v4_legacy"
V5_ARCHITECTURES = {"v5_t", "v5_tf"}
SUPPORTED_ARCHITECTURES = {LEGACY_ARCHITECTURE, *V5_ARCHITECTURES}


def resolve_architecture(model_config: Mapping[str, object]) -> str:
    """Resolve an explicit architecture without inferring it from channel count.

    Missing architecture is the sole compatibility exception: historical V4
    configs predate the factory, so they continue to select the legacy model.
    """
    architecture = str(model_config.get("architecture", LEGACY_ARCHITECTURE))
    aliases = {
        "legacy": LEGACY_ARCHITECTURE,
        "multichannel_csdi": LEGACY_ARCHITECTURE,
    }
    architecture = aliases.get(architecture, architecture)
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported model architecture={architecture!r}; "
            f"expected one of {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    return architecture


def build_model(model_config: Mapping[str, object], device):
    """Instantiate the architecture selected by ``model.architecture``."""
    architecture = resolve_architecture(model_config)
    if architecture == LEGACY_ARCHITECTURE:
        from diff_models_multivariate import MultiChannelCSDI

        model = MultiChannelCSDI(dict(model_config), device)
        model.architecture = LEGACY_ARCHITECTURE
        return model

    from .v5_conditioned_diffusion import V5Stage1Model

    config = dict(model_config)
    expected_sequence_condition = architecture == "v5_tf"
    configured = bool(config.get("use_sequence_condition", expected_sequence_condition))
    if configured != expected_sequence_condition:
        raise ValueError(
            f"architecture={architecture!r} requires "
            f"use_sequence_condition={expected_sequence_condition}"
        )
    if "use_network_condition" in config:
        network_condition = bool(config["use_network_condition"])
        if network_condition != expected_sequence_condition:
            raise ValueError(
                f"architecture={architecture!r} requires "
                f"use_network_condition={expected_sequence_condition}"
            )
    if bool(config.get("use_guidance", False)):
        raise ValueError("V5 stage-1 architectures require use_guidance=False")
    config["architecture"] = architecture
    config["use_sequence_condition"] = expected_sequence_condition
    return V5Stage1Model(config, device)


def load_model_checkpoint(model, checkpoint: Mapping[str, object], architecture=None):
    """Load a checkpoint while preserving the historical V4 loading contract.

    V5 checkpoints are strict because they are versioned by an explicit
    architecture. V4 keeps ``strict=False`` for old diffusion-buffer
    compatibility and returns the incompatibility lists to the caller.
    """
    architecture = architecture or getattr(model, "architecture", LEGACY_ARCHITECTURE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if architecture in V5_ARCHITECTURES:
        model.load_state_dict(state_dict, strict=True)
        return [], []
    incompatible = model.load_state_dict(state_dict, strict=False)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
