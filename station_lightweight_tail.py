"""Versioned training wiring only; the existing JMRT network is unchanged."""
from copy import deepcopy
from pathlib import Path
import yaml

VERSION = "v2_fair_v1"
RAMP_SELECTION_VERSION = "source_direction_daylight_pooled_v1"
AUX_TIMESTEP_COMPENSATION_VERSION = "clipped_sqrt_snr_mean1_v1"
FROZEN_AUXILIARY_ALPHAS = {
    0.65: (0.117, 0.091, 0.065),
    1.00: (0.180, 0.140, 0.100),
    1.35: (0.243, 0.189, 0.135),
}
REFERENCE = Path(__file__).resolve().parent / "configs/station24_independent_joint_tail_v2_event_balanced_168h.yaml"
ADDITIONS = {
    "joint_multiresidual_training_version": VERSION,
    "use_body_tail_experts": True, "independent_joint_tail_training": False,
    "use_joint_multiresidual_tail": True,
    "tail_expert_channels": 16, "tail_epsilon_context_hours": 0,
    "tail_common_gate_init": -1.0, "tail_gate_channels": 16,
    "tail_gate_prior_probability": 0.076, "tail_gate_loss_weight": 0.0,
    "joint_multiresidual_channels": 24, "joint_multiresidual_haar_levels": 3,
    # Historical compatibility field; new version generates a tail-only pool.
    "joint_multiresidual_tail_fraction": 0.20,
    "joint_multiresidual_decomposition_loss_weight": 0.0,
    "joint_multiresidual_structure_loss_weight": 0.0,
    "jstd_decomposition_loss_weight": 0.0, "jstd_structure_loss_weight": 0.0,
    "jstd_mask_loss_weight": 0.0, "jstd_issue_loss_weight": 0.0,
    "jstd_outside_zero_loss_weight": 0.0,
}


def expected_config(
    ramp_selection=False,
    auxiliary_alpha=1.0,
    auxiliary_timestep_compensation=False,
):
    auxiliary_alpha = float(auxiliary_alpha)
    if auxiliary_alpha not in FROZEN_AUXILIARY_ALPHAS:
        raise ValueError(
            f"alpha is outside the frozen Stage 1A set: {auxiliary_alpha}"
        )
    value = yaml.safe_load(REFERENCE.read_text(encoding="utf-8"))
    value["experiment"].update(
        name="station24_lightweight_joint_tail_v2_fair_168h",
        variant="geo_history_actual_lightweight_joint_tail_v2_fair",
        description="Existing frozen-Raw JMRT network with the unchanged V2 objective",
    )
    value["model"].update(ADDITIONS)
    ramp, shape, slow = FROZEN_AUXILIARY_ALPHAS[auxiliary_alpha]
    value["model"].update(
        event_balanced_ramp_loss_weight=ramp,
        event_balanced_shape_loss_weight=shape,
        event_balanced_slow_loss_weight=slow,
    )
    if auxiliary_alpha != 1.0:
        tag = str(auxiliary_alpha).replace(".", "p")
        value["experiment"].update(
            name=f"station24_lightweight_joint_tail_v2_alpha_{tag}_168h",
            variant=f"geo_history_actual_lightweight_joint_tail_v2_alpha_{tag}",
            description=(
                "Frozen Stage 1A auxiliary-strength candidate; architecture, "
                "sampling, selector, and relative loss proportions unchanged"
            ),
        )
        value["model"]["auxiliary_strength_alpha"] = auxiliary_alpha
    if ramp_selection:
        value["experiment"].update(
            name="station24_lightweight_joint_tail_v2_ramp_selection_168h",
            variant="geo_history_actual_lightweight_joint_tail_v2_ramp_selection",
            description=(
                "Lightweight V2 ramp-selection-only ablation with source, "
                "direction, and daylight separation"
            ),
        )
        value["model"]["event_balanced_ramp_selection_version"] = (
            RAMP_SELECTION_VERSION
        )
    if auxiliary_timestep_compensation:
        value["experiment"].update(
            name="station24_lightweight_joint_tail_aux_timestep_compensation_168h",
            variant="geo_history_actual_lightweight_joint_tail_aux_timestep_compensation",
            description=(
                "Lightweight Tail with fixed clipped mean-one inverse-Jacobian "
                "compensation on ramp/shape/slow only"
            ),
        )
        value["model"].update(
            auxiliary_timestep_compensation=True,
            auxiliary_timestep_compensation_version=(
                AUX_TIMESTEP_COMPENSATION_VERSION
            ),
        )
    return value


def validate_model_version(config):
    version = config.get("joint_multiresidual_training_version", "legacy_v1")
    if version == "legacy_v1":
        return False
    if version != VERSION:
        raise ValueError(f"unknown joint multiresidual training version: {version}")
    for key, expected in ADDITIONS.items():
        if config.get(key) != expected:
            raise ValueError(f"lightweight V2 requires {key}={expected!r}")
    alpha = float(config.get("auxiliary_strength_alpha", 1.0))
    if alpha not in FROZEN_AUXILIARY_ALPHAS:
        raise ValueError(f"alpha is outside the frozen Stage 1A set: {alpha}")
    ramp, shape, slow = FROZEN_AUXILIARY_ALPHAS[alpha]
    for key, expected in {
        "independent_tail_event_sampling_fraction": .60,
        "event_balanced_ramp_loss_weight": ramp,
        "event_balanced_shape_loss_weight": shape,
        "event_balanced_slow_loss_weight": slow,
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"lightweight V2 requires {key}={expected}")
    for key in ("use_jstd_tail", "use_jstd_event_hypothesis", "use_jstd_segment_prior",
                "use_discrete_event_memory", "use_retrieval_mismatch_expert",
                "use_tail_time_localizer", "use_event_replay_x0", "use_extreme_event_weighting",
                "train_sampler_energy_score_only", "independent_tail_natural_sampling"):
        if config.get(key, False):
            raise ValueError(f"lightweight V2 forbids {key}")
    return True


def validate_config(config, resolved=False):
    ramp_selection = (
        config["model"].get(
            "event_balanced_ramp_selection_version", "legacy_abs_topk_v1"
        )
        == RAMP_SELECTION_VERSION
    )
    auxiliary_alpha = float(
        config["model"].get("auxiliary_strength_alpha", 1.0)
    )
    auxiliary_timestep_compensation = bool(
        config["model"].get("auxiliary_timestep_compensation", False)
    )
    if auxiliary_timestep_compensation and config["model"].get(
        "auxiliary_timestep_compensation_version"
    ) != AUX_TIMESTEP_COMPENSATION_VERSION:
        raise ValueError("unknown auxiliary timestep compensation recipe")
    expected = expected_config(
        ramp_selection=ramp_selection,
        auxiliary_alpha=auxiliary_alpha,
        auxiliary_timestep_compensation=auxiliary_timestep_compensation,
    )
    actual = deepcopy(config)
    if resolved:
        if Path(actual["data"]["data_path"]).resolve() != Path(expected["data"]["data_path"]).resolve():
            raise ValueError("different lightweight dataset")
        actual["data"]["data_path"] = expected["data"]["data_path"]
        actual["model"]["secondary_adjacency_path"] = None
    for section in ("experiment", "data", "target", "model", "train", "evaluation"):
        for key, value in expected[section].items():
            if actual[section].get(key) != value:
                raise ValueError(f"lightweight mismatch: {section}.{key}")
        if not resolved and actual[section] != expected[section]:
            raise ValueError(f"unexpected lightweight settings: {section}")
    validate_model_version(config["model"])
