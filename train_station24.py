"""Train one 24-station conditional diffusion ablation on train/validation only."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_graph_prior import prepare_training_graphs
from station_dataset import (
    fit_station_forecast_mismatch_replay,
    fit_station_unified_event_replay,
    fit_station_event_replay,
    fit_station_event_weighting,
    fit_station_residual_scale,
    fit_station_state_thresholds,
    get_station_dataloader,
    load_station_static_data,
    write_residual_scale,
    write_station_event_weighting,
    write_station_event_replay,
    write_station_state_thresholds,
)
from station_retrieval_memory import build_retrieval_arrays
from station_discrete_event_memory import build_discrete_event_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-root", default="outputs_shandong/station24")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--secondary-adjacency", default=None)
    parser.add_argument(
        "--initialize-checkpoint",
        default=None,
        help="Initialize a parameter-isolated expert from an existing checkpoint",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "sample_index"
    }


def create_ema(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def update_ema(
    ema_state: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
    trainable_state_names: set[str] | None = None,
) -> None:
    for name, value in model.state_dict().items():
        if trainable_state_names is not None and name not in trainable_state_names:
            # Preserve a frozen source model bit-for-bit. Repeated EMA arithmetic
            # on an unchanged tensor can otherwise accumulate round-off drift.
            ema_state[name].copy_(value.detach())
        elif value.is_floating_point():
            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value.detach())


def ema_decay_for_step(
    max_decay: float,
    optimization_step: int,
    warmup: Mapping[str, object] | None = None,
) -> float:
    """Return a fixed or warm-up EMA decay for one optimizer update.

    The warm-up schedule follows the power-law form used by modern diffusion
    training utilities.  It prevents a newly initialized, short-trained adapter
    from being dominated by its initialization while retaining ``max_decay`` as
    the long-run smoothing limit.
    """

    if not 0.0 <= max_decay < 1.0:
        raise ValueError("ema_decay must be in [0,1)")
    if not warmup or not bool(warmup.get("enabled", False)):
        return max_decay
    inv_gamma = float(warmup.get("inv_gamma", 1.0))
    power = float(warmup.get("power", 0.75))
    min_decay = float(warmup.get("min_decay", 0.0))
    update_after_step = int(warmup.get("update_after_step", 0))
    if inv_gamma <= 0.0 or power <= 0.0:
        raise ValueError("EMA warm-up inv_gamma and power must be positive")
    if not 0.0 <= min_decay <= max_decay:
        raise ValueError("EMA warm-up min_decay must be in [0, ema_decay]")
    adjusted_step = max(0, int(optimization_step) - update_after_step)
    if adjusted_step == 0:
        return min_decay
    decay = 1.0 - (1.0 + adjusted_step / inv_gamma) ** (-power)
    return min(max_decay, max(min_decay, decay))


def state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in state.items()}


@torch.no_grad()
def build_body_tail_validation_events(
    batch: Mapping[str, torch.Tensor],
    model: Station24DiffusionModel,
    event_replay: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Label validation events with train-fitted thresholds for model selection.

    These labels are used only to choose a checkpoint. They are never passed to
    generation or to the causal risk gate as inputs.
    """

    if event_replay.get("method") == "train_unified_wind_event_replay_v1":
        deep_active, deep_window, deep_severity = build_body_tail_validation_events(
            batch, model, event_replay["deep_replay"]
        )
        mismatch_active, mismatch_window, mismatch_severity = (
            build_body_tail_validation_events(
                batch, model, event_replay["mismatch_replay"]
            )
        )
        deep_scale = max(
            float(event_replay["deep_replay"]["severity_thresholds"][0]), 1e-6
        )
        mismatch_scale = max(
            float(event_replay["mismatch_replay"]["severity_thresholds"][0]),
            1e-6,
        )
        choose_mismatch = (mismatch_active > 0) & (
            (deep_active == 0)
            | (mismatch_severity / mismatch_scale > deep_severity / deep_scale)
        )
        active = torch.maximum(deep_active, mismatch_active)
        window = torch.where(
            choose_mismatch[:, None], mismatch_window, deep_window
        )
        severity = torch.where(
            choose_mismatch,
            mismatch_severity / mismatch_scale,
            deep_severity / deep_scale,
        )
        return active, window, severity

    if event_replay.get("method") == "train_independent_forecast_missed_ramp_replay_v1":
        window = int(event_replay["event_window_hours"])
        forecast = batch["forecast"]
        actual = batch["actual"]
        valid = batch["valid_mask"]
        wind_weight = model.denoiser.wind_capacity_weight.to(forecast.dtype)
        wind_mask = model.denoiser.wind_station_mask.bool()
        forecast_wind = torch.einsum("s,bst->bt", wind_weight, forecast)
        actual_wind = torch.einsum("s,bst->bt", wind_weight, actual)
        complete = valid[:, wind_mask].amin(dim=1) > 0.5
        score = torch.zeros_like(forecast_wind)
        fraction = float(event_replay["forecast_magnitude_fraction"])
        for lag in (int(value) for value in event_replay["ramp_lags"]):
            actual_ramp = actual_wind[:, lag:] - actual_wind[:, :-lag]
            forecast_ramp = forecast_wind[:, lag:] - forecast_wind[:, :-lag]
            threshold = float(
                event_replay["actual_ramp_abs_q90_thresholds"][str(lag)]
            )
            pair_valid = complete[:, lag:] & complete[:, :-lag]
            missed = pair_valid & (actual_ramp.abs() >= threshold) & (
                (torch.sign(actual_ramp) != torch.sign(forecast_ramp))
                | (forecast_ramp.abs() < fraction * actual_ramp.abs())
            )
            current = torch.where(
                missed,
                (actual_ramp - forecast_ramp).abs() / max(threshold, 1e-6),
                torch.zeros_like(actual_ramp),
            )
            score[:, lag:] = torch.maximum(score[:, lag:], current)
        severity, center = score.max(dim=1)
        active = (
            severity >= float(event_replay["severity_thresholds"][0])
        ).to(forecast.dtype)
        start = torch.clamp(center - window // 3, min=0, max=score.shape[1] - window)
        event_window = torch.zeros_like(score)
        offsets = torch.arange(window, device=forecast.device)[None]
        event_window.scatter_(
            1, start[:, None] + offsets, active[:, None].expand(-1, window)
        )
        return active, event_window, severity

    window = int(event_replay["event_window_hours"])
    threshold = float(event_replay["severity_thresholds"][0])
    forecast = batch["forecast"]
    actual = batch["actual"]
    valid = batch["valid_mask"]
    wind_weight = model.denoiser.wind_capacity_weight.to(forecast.dtype)
    wind_mask = model.denoiser.wind_station_mask.bool()
    mismatch = torch.einsum(
        "s,bst->bt", wind_weight, forecast - actual
    )
    complete = valid[:, wind_mask].amin(dim=1)
    rolling = F.avg_pool1d(
        mismatch[:, None, :], kernel_size=window, stride=1
    )[:, 0]
    rolling_valid = F.avg_pool1d(
        complete[:, None, :].to(forecast.dtype),
        kernel_size=window,
        stride=1,
    )[:, 0] >= 1.0 - 1.0e-6
    rolling = rolling.masked_fill(~rolling_valid, float("-inf"))
    severity, start = rolling.max(dim=1)
    if torch.any(~torch.isfinite(severity)):
        raise ValueError("validation issue has no complete wind event window")
    active = (severity >= threshold).to(forecast.dtype)
    event_window = torch.zeros_like(mismatch)
    offsets = torch.arange(window, device=forecast.device)[None, :]
    indices = start[:, None] + offsets
    event_window.scatter_(1, indices, active[:, None].expand(-1, window))
    return active, event_window, severity


@torch.no_grad()
def validate(
    model: Station24DiffusionModel,
    loader,
    device: torch.device,
    seed: int,
    event_replay: Mapping[str, object] | None = None,
) -> tuple[float, dict[str, float]]:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total_loss = 0.0
    total_weight = 0.0
    gate_error_sum = 0.0
    gate_samples = 0
    tail_event_count = 0
    location_mass_sum = 0.0
    location_offset_sum = 0.0
    mismatch_time_error_sum = 0.0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        batch_size = batch["forecast"].shape[0]
        if model.train_tail_time_localizer_only:
            if event_replay is None:
                raise ValueError(
                    "tail time validation requires train event thresholds"
                )
            active, event_window, _ = build_body_tail_validation_events(
                batch, model, event_replay
            )
            batch["event_active"] = active
            batch["event_window_mask"] = event_window
            active_count = int(active.sum())
            if active_count:
                loss = model.tail_time_localization_loss(batch)
                probability = model.tail_time_probability(batch)
                event_mass = (probability * event_window).sum(dim=-1)
                target_index = (
                    event_window
                    * torch.arange(
                        event_window.shape[-1],
                        device=device,
                        dtype=event_window.dtype,
                    )[None]
                ).sum(dim=-1) / event_window.sum(dim=-1).clamp(min=1.0)
                predicted_index = probability.argmax(dim=-1).to(target_index.dtype)
                offset = torch.abs(predicted_index - target_index)
                total_loss += float(loss) * active_count
                total_weight += active_count
                location_mass_sum += float((event_mass * active).sum())
                location_offset_sum += float((offset * active).sum())
                tail_event_count += active_count
            continue
        timestep = torch.randint(
            0,
            model.diffusion.num_steps,
            (batch_size,),
            generator=generator,
            device=device,
        )
        noise = torch.randn(
            batch["residual_target"].shape,
            generator=generator,
            device=device,
            dtype=batch["residual_target"].dtype,
        )
        if model.use_body_tail_experts:
            if event_replay is None:
                raise ValueError("body-tail validation requires train event thresholds")
            active, event_window, _ = build_body_tail_validation_events(
                batch, model, event_replay
            )
            batch["event_active"] = active
            batch["event_window_mask"] = event_window
            support = model.body_tail_epsilon_weight(batch) * batch[
                "valid_mask"
            ].to(active.dtype)
            support_count = float(support.sum())
            loss = model(
                batch,
                timestep=timestep,
                noise=noise,
                include_auxiliary=False,
                body_tail_event_masking=True,
            )
            total_loss += float(loss) * support_count
            total_weight += support_count
            target = active
            if model.use_retrieval_mismatch_expert:
                context, _ = model.encode_retrieval_memory(batch)
                logits = model.mismatch_risk_logits(batch, context)
                time_logits = model.mismatch_time_logits(batch, context)
                time_error = F.binary_cross_entropy_with_logits(
                    time_logits, event_window, reduction="none"
                ).mean(dim=1)
                mismatch_time_error_sum += float(time_error.sum())
            else:
                logits = model.tail_risk_logits(batch)
            gate_error_sum += float(
                F.binary_cross_entropy_with_logits(
                    logits, target, reduction="sum"
                )
            )
            gate_samples += batch_size
            tail_event_count += int(active.sum())
        else:
            loss = model(
                batch,
                timestep=timestep,
                noise=noise,
                include_auxiliary=False,
            )
            total_loss += float(loss) * batch_size
            total_weight += batch_size
    if model.train_retrieval_mismatch_only:
        ema_trainable_state_names = set(model.retrieval_mismatch_state_dict_keys)
    elif model.train_tail_time_localizer_only:
        if total_weight <= 0.0:
            raise ValueError("validation split contains no tail localization event")
        objective = total_loss / total_weight
        return objective, {
            "val_tail_time_nll": objective,
            "val_tail_time_event_mass": location_mass_sum / total_weight,
            "val_tail_time_argmax_abs_offset_h": (
                location_offset_sum / total_weight
            ),
            "val_tail_event_count": float(tail_event_count),
        }
    if model.use_body_tail_experts:
        if total_weight <= 0.0 or tail_event_count <= 0:
            raise ValueError("validation split contains no train-threshold tail event")
        tail_epsilon = total_loss / total_weight
        gate_bce = gate_error_sum / max(gate_samples, 1)
        if model.use_retrieval_mismatch_expert:
            time_bce = mismatch_time_error_sum / max(gate_samples, 1)
            objective = (
                tail_epsilon
                + model.mismatch_gate_loss_weight * gate_bce
                + model.mismatch_time_loss_weight * time_bce
            )
            return objective, {
                "val_mismatch_epsilon_loss": tail_epsilon,
                "val_mismatch_gate_bce": gate_bce,
                "val_mismatch_time_bce": time_bce,
                "val_mismatch_event_count": float(tail_event_count),
            }
        objective = tail_epsilon + model.tail_gate_loss_weight * gate_bce
        return objective, {
            "val_tail_epsilon_loss": tail_epsilon,
            "val_tail_gate_bce": gate_bce,
            "val_tail_event_count": float(tail_event_count),
        }
    return total_loss / max(total_weight, 1.0), {}


def save_checkpoint(
    path: Path,
    model: Station24DiffusionModel,
    ema_state: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, object],
    residual_scale: Mapping[str, object],
    epoch: int,
    train_loss: float,
    val_loss: float,
    parameter_count: int,
    state_thresholds: Mapping[str, object] | None,
    event_weighting: Mapping[str, object] | None,
    event_replay: Mapping[str, object] | None,
    graph_manifest: Mapping[str, object],
    ema_metadata: Mapping[str, object] | None = None,
) -> None:
    payload = {
        "architecture": model.architecture,
        "spatial_mode": model.spatial_mode,
        "spatial_mix_levels": list(model.spatial_mix_levels),
        "parallel_spatial_fusion_levels": list(
            model.parallel_spatial_fusion_levels
        ),
        "parallel_spatial_adjacency_mode": (
            model.parallel_spatial_adjacency_mode
        ),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "parameter_count": int(parameter_count),
        "model_state_dict": state_to_cpu(model.state_dict()),
        "ema_model_state_dict": state_to_cpu(ema_state),
        "ema": copy.deepcopy(dict(ema_metadata)) if ema_metadata else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "residual_scale": dict(residual_scale),
        "config": copy.deepcopy(dict(config)),
        "spatial_gate_values": model.spatial_gate_values,
        "parallel_spatial_gate_statistics": (
            model.parallel_spatial_gate_statistics
        ),
        "condition_variant": str(config.get("experiment", {}).get("variant", "baseline")),
        "condition_gate_values": model.condition_gate_values,
        "forecast_condition_dropout_prob": float(
            model.denoiser.forecast_condition_dropout_prob
        ),
        "forecast_condition_dropout_statistics": (
            model.forecast_condition_dropout_statistics
        ),
        "forecast_correction_mode": model.forecast_correction_mode,
        "forecast_correction_loss_weight": float(
            model.forecast_correction_loss_weight
        ),
        "forecast_correction_huber_beta": float(
            model.forecast_correction_huber_beta
        ),
        "state_gate_values": model.state_gate_values,
        "wind_common_gate_value": model.wind_common_gate_value,
        "use_body_tail_experts": bool(model.use_body_tail_experts),
        "use_tail_time_localizer": bool(model.use_tail_time_localizer),
        "train_tail_time_localizer_only": bool(
            model.train_tail_time_localizer_only
        ),
        "use_retrieval_mismatch_expert": bool(
            model.use_retrieval_mismatch_expert
        ),
        "train_retrieval_mismatch_only": bool(
            model.train_retrieval_mismatch_only
        ),
        "use_discrete_event_memory": bool(model.use_discrete_event_memory),
        "train_discrete_event_memory_only": bool(
            model.train_discrete_event_memory_only
        ),
        "tail_gate_loss_weight": float(model.tail_gate_loss_weight),
        "tail_common_gate_value": model.tail_common_gate_value,
        "body_tail_trainable_parameter_names": list(
            model.body_tail_trainable_parameter_names
        ),
        "tail_time_trainable_parameter_names": list(
            model.tail_time_trainable_parameter_names
        ),
        "retrieval_mismatch_trainable_parameter_names": list(
            model.retrieval_mismatch_trainable_parameter_names
        ),
        "discrete_event_trainable_parameter_names": list(
            model.discrete_event_trainable_parameter_names
        ),
        "state_thresholds": (
            copy.deepcopy(dict(state_thresholds))
            if state_thresholds is not None
            else None
        ),
        "event_weighting": (
            copy.deepcopy(dict(event_weighting))
            if event_weighting is not None
            else None
        ),
        "event_replay": (
            copy.deepcopy(dict(event_replay))
            if event_replay is not None
            else None
        ),
        "graph_manifest": copy.deepcopy(dict(graph_manifest)),
    }
    torch.save(payload, path)


def plot_losses(
    history: list[dict[str, float]],
    output: Path,
    ylabel: str = "Fixed-noise epsilon MSE",
    title: str = "Station24 diffusion training",
) -> None:
    frame_epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_epochs = [row["epoch"] for row in history if "val_loss" in row]
    val_losses = [row["val_loss"] for row in history if "val_loss" in row]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(frame_epochs, train_losses, label="train")
    axis.plot(val_epochs, val_losses, marker="o", label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["model"].get("architecture") != "station24_resunet":
        raise ValueError("station trainer requires architecture=station24_resunet")
    train_config = config["train"]
    if args.epochs is not None:
        train_config["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_config["batch_size"] = int(args.batch_size)
        train_config["effective_batch_size"] = int(args.batch_size) * int(
            train_config.get("gradient_accumulation_steps", 1)
        )
    if args.num_workers is not None:
        train_config["num_workers"] = int(args.num_workers)
    if args.seed is not None:
        train_config["seed"] = int(args.seed)
    data_path = Path(args.data_path or config["data"]["data_path"])
    config["data"]["data_path"] = str(data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is required for full training; use --allow-cpu only for smoke tests")
    seed = int(train_config["seed"])
    set_seed(seed)

    experiment_name = args.exp_name or config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{timestamp}_{experiment_name}_seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run directory {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    static = load_station_static_data(data_path)
    primary_adjacency, secondary_adjacency, graph_manifest = (
        prepare_training_graphs(
            data_path,
            run_dir,
            config["model"],
            args.secondary_adjacency,
        )
    )

    scale_config = config["target"]["residual_scaling"]
    residual_scale = fit_station_residual_scale(
        data_path,
        epsilon=float(scale_config.get("epsilon", 1e-4)),
        method=str(scale_config.get("method", "per_station_std")),
        condition_config=scale_config,
    )
    write_residual_scale(run_dir / "residual_scale.json", residual_scale)
    state_thresholds = None
    if bool(config["model"].get("use_state_encoder", False)):
        state_thresholds = fit_station_state_thresholds(
            data_path,
            low_quantile=float(config["model"].get("state_low_quantile", 0.20)),
            high_quantile=float(config["model"].get("state_high_quantile", 0.90)),
            ramp_quantile=float(config["model"].get("state_ramp_quantile", 0.90)),
            ramp_lags=tuple(
                int(value)
                for value in config["model"].get("state_ramp_lags", [3, 6])
            ),
        )
        write_station_state_thresholds(
            run_dir / "state_thresholds.json", state_thresholds
        )
    event_weighting = None
    if bool(config["model"].get("use_extreme_event_weighting", False)):
        event_weighting = fit_station_event_weighting(
            data_path, config["model"]
        )
        write_station_event_weighting(
            run_dir / "event_weighting.json", event_weighting
        )
    event_replay = None
    if bool(config["model"].get("use_event_replay_x0", False)):
        if event_weighting is not None:
            raise ValueError(
                "B1 event replay cannot be combined with legacy event weighting"
            )
        if str(config["model"].get("forecast_correction_mode", "none")) != "none":
            raise ValueError(
                "B1 event replay must not use the A1/A2 forecast correction head"
            )
        if bool(config["model"].get("use_discrete_event_memory", False)):
            event_replay = fit_station_unified_event_replay(
                data_path, config["model"]
            )
        elif bool(config["model"].get("use_retrieval_mismatch_expert", False)):
            event_replay = fit_station_forecast_mismatch_replay(
                data_path, config["model"]
            )
        else:
            event_replay = fit_station_event_replay(data_path, config["model"])
        write_station_event_replay(
            run_dir / "event_replay.json", event_replay
        )
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    train_retrieval = None
    val_retrieval = None
    if bool(config["model"].get("use_discrete_event_memory", False)):
        retrieval_k = int(config["model"].get("event_memory_top_k", 48))
        exclusion_days = int(config["model"].get("retrieval_exclusion_days", 6))
        event_quantile = float(config["model"].get("event_memory_quantile", 0.75))
        stride = int(config["model"].get("event_memory_target_stride_hours", 3))
        train_retrieval = build_discrete_event_arrays(
            data_path, "train", retrieval_k, exclusion_days, event_quantile, stride
        )
        val_retrieval = build_discrete_event_arrays(
            data_path, "val", retrieval_k, exclusion_days, event_quantile, stride
        )
        (run_dir / "discrete_event_memory_audit.json").write_text(
            json.dumps(
                {"train": train_retrieval.audit, "val": val_retrieval.audit},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    elif bool(config["model"].get("use_retrieval_mismatch_expert", False)):
        retrieval_k = int(config["model"].get("retrieval_top_k", 40))
        exclusion_days = int(config["model"].get("retrieval_exclusion_days", 6))
        train_retrieval = build_retrieval_arrays(
            data_path, "train", retrieval_k, exclusion_days
        )
        val_retrieval = build_retrieval_arrays(
            data_path, "val", retrieval_k, exclusion_days
        )
        (run_dir / "retrieval_audit.json").write_text(
            json.dumps(
                {"train": train_retrieval.audit, "val": val_retrieval.audit},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    train_loader, train_dataset = get_station_dataloader(
        data_path,
        "train",
        residual_scale,
        batch_size=int(train_config["batch_size"]),
        seed=seed,
        num_workers=int(train_config.get("num_workers", 0)),
        persistent_workers=bool(
            train_config.get(
                "persistent_workers", int(train_config.get("num_workers", 0)) > 0
            )
        ),
        prefetch_factor=int(train_config.get("prefetch_factor", 2)),
        condition_config=config["model"],
        state_thresholds=state_thresholds,
        event_weighting=event_weighting,
        event_replay=event_replay,
        retrieval_arrays=train_retrieval,
    )
    val_loader, val_dataset = get_station_dataloader(
        data_path,
        "val",
        residual_scale,
        batch_size=int(train_config["batch_size"]),
        seed=int(train_config.get("validation_seed", 314159)),
        num_workers=int(train_config.get("num_workers", 0)),
        persistent_workers=bool(
            train_config.get(
                "persistent_workers", int(train_config.get("num_workers", 0)) > 0
            )
        ),
        prefetch_factor=int(train_config.get("prefetch_factor", 2)),
        condition_config=config["model"],
        state_thresholds=state_thresholds,
        event_weighting=event_weighting,
        event_replay=event_replay,
        retrieval_arrays=val_retrieval,
    )
    (run_dir / "condition_feature_audit.json").write_text(
        json.dumps(
            {
                "train": train_dataset.condition_audit,
                "val": val_dataset.condition_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    model = Station24DiffusionModel(
        config["model"],
        static["station_features"],
        primary_adjacency,
        static["station_capacities"],
        secondary_adjacency,
    ).to(device)
    initialization_manifest = None
    if model.use_body_tail_experts:
        if args.initialize_checkpoint is None:
            raise ValueError(
                "body-tail expert training requires --initialize-checkpoint"
            )
        initialization_path = Path(args.initialize_checkpoint)
        if not initialization_path.is_file():
            raise FileNotFoundError(
                f"initialization checkpoint not found: {initialization_path}"
            )
        initialization = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        if model.train_discrete_event_memory_only:
            if initialization.get("condition_variant") != (
                "geo_history_actual_body_tail_moe"
            ):
                raise ValueError(
                    "discrete event memory must initialize from the Raw "
                    "geo_history_actual_body_tail_moe checkpoint"
                )
            source_state = initialization["model_state_dict"]
            incompatible = model.load_state_dict(source_state, strict=False)
            expected_missing = set(model.discrete_event_new_state_dict_keys)
            if set(incompatible.missing_keys) != expected_missing:
                raise ValueError(
                    "unexpected discrete-event initialization gaps: "
                    f"{sorted(incompatible.missing_keys)}"
                )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "unexpected discrete-event initialization keys: "
                    f"{sorted(incompatible.unexpected_keys)}"
                )
            trainable_names = model.configure_discrete_event_training()
            initialization_manifest = {
                "method": "frozen_raw_body_plus_unified_discrete_event_expert",
                "checkpoint": str(initialization_path),
                "checkpoint_state_source": "raw",
                "source_condition_variant": str(initialization["condition_variant"]),
                "source_epoch": int(initialization["epoch"]),
                "body_frozen": True,
                "third_mismatch_expert_used": False,
                "topk_averaging": False,
                "event_memory_top_k": int(
                    config["model"].get("event_memory_top_k", 48)
                ),
                "trainable_parameter_names": list(trainable_names),
            }
        elif model.train_retrieval_mismatch_only:
            if initialization.get("condition_variant") != (
                "geo_history_actual_body_tail_moe"
            ):
                raise ValueError(
                    "retrieval mismatch expert must initialize from the Raw "
                    "geo_history_actual_body_tail_moe checkpoint"
                )
            source_state = initialization["model_state_dict"]
            incompatible = model.load_state_dict(source_state, strict=False)
            expected_missing = set(model.retrieval_mismatch_state_dict_keys)
            if set(incompatible.missing_keys) != expected_missing:
                raise ValueError(
                    "unexpected retrieval-mismatch initialization gaps: "
                    f"{sorted(incompatible.missing_keys)}"
                )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "unexpected retrieval-mismatch initialization keys: "
                    f"{sorted(incompatible.unexpected_keys)}"
                )
            trainable_names = model.configure_retrieval_mismatch_training()
            initialization_manifest = {
                "method": "frozen_raw_body_deep_tail_plus_retrieval_mismatch_expert",
                "checkpoint": str(initialization_path),
                "checkpoint_state_source": "raw",
                "source_condition_variant": str(initialization["condition_variant"]),
                "source_epoch": int(initialization["epoch"]),
                "body_frozen": True,
                "deep_tail_frozen": True,
                "retrieval_top_k": int(config["model"].get("retrieval_top_k", 40)),
                "trainable_parameter_names": list(trainable_names),
            }
        elif model.train_tail_time_localizer_only:
            if initialization.get("condition_variant") != (
                "geo_history_actual_body_tail_moe"
            ):
                raise ValueError(
                    "tail time localizer must initialize from the Raw "
                    "geo_history_actual_body_tail_moe checkpoint"
                )
            # The raw tail checkpoint is the experimental result being retained.
            # Do not silently fall back to the EMA state that suppressed it.
            source_state = initialization["model_state_dict"]
            incompatible = model.load_state_dict(source_state, strict=False)
            expected_missing = set(model.tail_time_state_dict_keys)
            if set(incompatible.missing_keys) != expected_missing:
                raise ValueError(
                    "unexpected tail-time initialization gaps: "
                    f"{sorted(incompatible.missing_keys)}"
                )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "unexpected tail-time initialization keys: "
                    f"{sorted(incompatible.unexpected_keys)}"
                )
            trainable_names = model.configure_tail_time_training()
            initialization_manifest = {
                "method": "frozen_raw_body_tail_plus_hourly_time_localizer",
                "checkpoint": str(initialization_path),
                "checkpoint_state_source": "raw",
                "checkpoint_state_key": "model_state_dict",
                "source_condition_variant": str(
                    initialization["condition_variant"]
                ),
                "source_epoch": int(initialization["epoch"]),
                "source_validation_objective": float(initialization["val_loss"]),
                "body_frozen": True,
                "tail_adapters_frozen": True,
                "issuance_gate_frozen": True,
                "trainable_parameter_names": list(trainable_names),
            }
        else:
            if initialization.get("condition_variant") != "geo_history_actual_dual":
                raise ValueError(
                    "body-tail expert must initialize from geo_history_actual_dual"
                )
            source_state = initialization.get(
                "ema_model_state_dict", initialization["model_state_dict"]
            )
            incompatible = model.load_state_dict(source_state, strict=False)
            expected_missing = set(model.body_tail_state_dict_keys)
            if set(incompatible.missing_keys) != expected_missing:
                raise ValueError(
                    "unexpected body-tail initialization gaps: "
                    f"{sorted(incompatible.missing_keys)}"
                )
            if incompatible.unexpected_keys:
                raise ValueError(
                    "unexpected body-tail initialization keys: "
                    f"{sorted(incompatible.unexpected_keys)}"
                )
            trainable_names = model.configure_body_tail_training()
            initialization_manifest = {
                "method": "frozen_historical_spatial_body_plus_tail_residual_adapter",
                "checkpoint": str(initialization_path),
                "source_condition_variant": str(
                    initialization["condition_variant"]
                ),
                "source_epoch": int(initialization["epoch"]),
                "source_validation_mse": float(initialization["val_loss"]),
                "body_frozen": True,
                "trainable_parameter_names": list(trainable_names),
            }
        (run_dir / "body_tail_initialization.json").write_text(
            json.dumps(initialization_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif args.initialize_checkpoint is not None:
        raise ValueError(
            "--initialize-checkpoint is reserved for parameter-isolated experts"
        )
    configured_event_x0_weight = sum(
        [
            model.diffusion.event_x0_magnitude_loss_weight,
            model.diffusion.event_x0_timing_loss_weight,
            model.diffusion.event_x0_sync_loss_weight,
        ]
    )
    if (event_replay is None) != (configured_event_x0_weight <= 0.0):
        raise ValueError(
            "event replay and positive event x0 loss weights must be enabled together"
        )
    if event_replay is not None and int(event_replay["event_window_hours"]) != int(
        model.diffusion.event_x0_window_hours
    ):
        raise ValueError(
            "event replay window and event x0 loss window must match"
        )
    if model.use_body_tail_experts and event_replay is None:
        raise ValueError("body-tail expert training requires event replay labels")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    amp_enabled = bool(train_config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    ema_decay = float(train_config.get("ema_decay", 0.999))
    ema_warmup = train_config.get("ema_warmup")
    if ema_warmup is not None and not isinstance(ema_warmup, Mapping):
        raise ValueError("train.ema_warmup must be a mapping")
    # Validate the schedule before the expensive training loop starts.
    ema_decay_for_step(ema_decay, 1, ema_warmup)
    ema_state = create_ema(model)
    if model.train_tail_time_localizer_only:
        ema_trainable_state_names = set(model.tail_time_state_dict_keys)
    elif model.train_discrete_event_memory_only:
        ema_trainable_state_names = set(model.discrete_event_state_dict_keys)
    elif model.use_body_tail_experts:
        ema_trainable_state_names = set(model.body_tail_state_dict_keys)
    else:
        ema_trainable_state_names = None
    accumulation = int(train_config.get("gradient_accumulation_steps", 1))
    clip_norm = float(train_config.get("gradient_clip_norm", 1.0))
    epochs = int(train_config["epochs"])
    validation_every = int(train_config.get("validation_every", 1))
    patience = int(train_config.get("patience", 40))
    min_delta = float(train_config.get("min_delta", 1e-5))
    save_every = int(train_config.get("save_every", 50))
    validation_seed = int(train_config.get("validation_seed", 314159))

    print(
        f"MODEL architecture={model.architecture} spatial_mode={model.spatial_mode} "
        f"spatial_levels={list(model.spatial_mix_levels)} "
        f"parallel_levels={list(model.parallel_spatial_fusion_levels)} "
        f"parallel_adjacency={model.parallel_spatial_adjacency_mode} "
        f"graph_mode={graph_manifest['mode']} "
        f"condition_variant={config.get('experiment', {}).get('variant', 'baseline')} "
        f"forecast_condition_dropout={model.denoiser.forecast_condition_dropout_prob} "
        f"forecast_correction={model.forecast_correction_mode} "
        f"correction_loss_weight={model.forecast_correction_loss_weight} "
        f"residual_scaling={residual_scale.get('method', 'per_station_std')} "
        f"ramp_aux_weight={model.diffusion.ramp_auxiliary_loss_weight} "
        f"common_event_weight={model.diffusion.wind_common_event_loss_weight} "
        f"event_weighting_method={event_weighting.get('method') if event_weighting else None} "
        f"event_replay_method={event_replay.get('method') if event_replay else None} "
        f"event_replay_count={event_replay.get('independent_event_count') if event_replay else 0} "
        f"event_x0_weights=({model.diffusion.event_x0_magnitude_loss_weight},"
        f"{model.diffusion.event_x0_timing_loss_weight},"
        f"{model.diffusion.event_x0_sync_loss_weight}) "
        f"body_tail_experts={model.use_body_tail_experts} "
        f"tail_time_localizer={model.use_tail_time_localizer} "
        f"tail_time_only={model.train_tail_time_localizer_only} "
        f"retrieval_mismatch={model.use_retrieval_mismatch_expert} "
        f"retrieval_mismatch_only={model.train_retrieval_mismatch_only} "
        f"discrete_event_memory={model.use_discrete_event_memory} "
        f"discrete_event_only={model.train_discrete_event_memory_only} "
        f"tail_gate_weight={model.tail_gate_loss_weight} "
        f"tail_common_gate={model.tail_common_gate_value} "
        f"common_gate={model.wind_common_gate_value} "
        f"condition_gates={model.condition_gate_values} parameters={parameter_count} "
        f"state_gates={model.state_gate_values} "
        f"trainable={trainable_count} device={device}"
    )
    print(
        f"TRAIN samples={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batch={train_config['batch_size']} accumulation={accumulation} epochs={epochs}"
    )
    print(
        f"INPUT_PIPELINE workers={train_config.get('num_workers', 0)} "
        f"persistent_workers={train_config.get('persistent_workers', int(train_config.get('num_workers', 0)) > 0)} "
        f"prefetch_factor={train_config.get('prefetch_factor', 2)} "
        f"pin_memory={torch.cuda.is_available()}"
    )
    print(
        f"EMA max_decay={ema_decay} warmup={dict(ema_warmup) if ema_warmup else None} "
        f"trainable_state_only={model.use_body_tail_experts}"
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    optimizer_updates = 0
    current_ema_decay = ema_decay_for_step(ema_decay, 0, ema_warmup)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_samples = 0
        event_draws = 0
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            batch_size = batch["forecast"].shape[0]
            if event_replay is not None:
                event_draws += int(batch["event_active"].sum().detach().cpu())
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = (
                    model.tail_time_localization_loss(batch)
                    if model.train_tail_time_localizer_only
                    else model(batch)
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % accumulation == 0 or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                current_ema_decay = ema_decay_for_step(
                    ema_decay, optimizer_updates, ema_warmup
                )
                update_ema(
                    ema_state,
                    model,
                    current_ema_decay,
                    trainable_state_names=ema_trainable_state_names,
                )
            batch_weight = (
                int(batch["event_active"].sum().detach().cpu())
                if model.train_tail_time_localizer_only
                else batch_size
            )
            total_loss += float(loss.detach()) * batch_weight
            total_samples += batch_weight
        train_loss = total_loss / max(total_samples, 1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        train_samples_per_second = len(train_loader.dataset) / max(train_seconds, 1e-9)
        row: dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": float(train_seconds),
            "train_samples_per_second": float(train_samples_per_second),
            "event_replay_draws": float(event_draws),
            "optimizer_updates": float(optimizer_updates),
            "ema_decay": float(current_ema_decay),
        }

        should_validate = epoch == 1 or epoch % validation_every == 0 or epoch == epochs
        if should_validate:
            val_loss, validation_details = validate(
                model,
                val_loader,
                device,
                validation_seed,
                event_replay=event_replay,
            )
            row["val_loss"] = val_loss
            row.update(validation_details)
            improved = val_loss < best_val - min_delta
            if improved:
                best_val = val_loss
                best_epoch = epoch
                save_checkpoint(
                    checkpoint_dir / "model_best.pt",
                    model,
                    ema_state,
                    optimizer,
                    config,
                    residual_scale,
                    epoch,
                    train_loss,
                    val_loss,
                    parameter_count,
                    state_thresholds,
                    event_weighting,
                    event_replay,
                    graph_manifest,
                    {
                        "max_decay": ema_decay,
                        "warmup": copy.deepcopy(dict(ema_warmup))
                        if ema_warmup
                        else None,
                        "optimizer_updates": optimizer_updates,
                        "current_decay": current_ema_decay,
                    },
                )
            print(
                f"epoch={epoch:04d} train={train_loss:.7f} val={val_loss:.7f} "
                f"best_epoch={best_epoch} spatial_gates={model.denoiser.spatial_block.gate_values()} "
                f"event_draws={event_draws} "
                f"train_s={train_seconds:.2f} samples_s={train_samples_per_second:.2f} "
                f"condition_gates={model.condition_gate_values}"
                f" state_gates={model.state_gate_values}"
            )
            if best_epoch and epoch - best_epoch >= patience:
                history.append(row)
                print(f"EARLY_STOP best_epoch={best_epoch} best_val={best_val:.7f}")
                break
        else:
            print(
                f"epoch={epoch:04d} train={train_loss:.7f} "
                f"event_draws={event_draws} train_s={train_seconds:.2f} "
                f"samples_s={train_samples_per_second:.2f}"
            )
        history.append(row)

        if epoch % save_every == 0:
            periodic_val = row.get("val_loss", float("nan"))
            save_checkpoint(
                checkpoint_dir / f"model_epoch_{epoch}.pt",
                model,
                ema_state,
                optimizer,
                config,
                residual_scale,
                epoch,
                train_loss,
                periodic_val,
                parameter_count,
                state_thresholds,
                event_weighting,
                event_replay,
                graph_manifest,
                {
                    "max_decay": ema_decay,
                    "warmup": copy.deepcopy(dict(ema_warmup))
                    if ema_warmup
                    else None,
                    "optimizer_updates": optimizer_updates,
                    "current_decay": current_ema_decay,
                },
            )

    if not (checkpoint_dir / "model_best.pt").is_file():
        raise RuntimeError("training ended without a best checkpoint")
    (log_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_losses(
        history,
        log_dir / "loss_curve.png",
        ylabel=(
            "Tail event-time NLL"
            if model.train_tail_time_localizer_only
            else "Fixed-noise epsilon MSE"
        ),
        title=(
            "Station24 tail time localization"
            if model.train_tail_time_localizer_only
            else "Station24 diffusion training"
        ),
    )
    final_record = {
        "architecture": model.architecture,
        "spatial_mode": model.spatial_mode,
        "spatial_mix_levels": list(model.spatial_mix_levels),
        "parallel_spatial_fusion_levels": list(
            model.parallel_spatial_fusion_levels
        ),
        "parallel_spatial_adjacency_mode": (
            model.parallel_spatial_adjacency_mode
        ),
        "condition_variant": str(
            config.get("experiment", {}).get("variant", "baseline")
        ),
        "condition_gate_values": model.condition_gate_values,
        "forecast_condition_dropout_prob": float(
            model.denoiser.forecast_condition_dropout_prob
        ),
        "forecast_condition_dropout_statistics": (
            model.forecast_condition_dropout_statistics
        ),
        "spatial_gate_values": model.spatial_gate_values,
        "parallel_spatial_gate_statistics": (
            model.parallel_spatial_gate_statistics
        ),
        "state_gate_values": model.state_gate_values,
        "use_body_tail_experts": bool(model.use_body_tail_experts),
        "use_tail_time_localizer": bool(model.use_tail_time_localizer),
        "train_tail_time_localizer_only": bool(
            model.train_tail_time_localizer_only
        ),
        "tail_gate_loss_weight": float(model.tail_gate_loss_weight),
        "tail_common_gate_value": model.tail_common_gate_value,
        "body_tail_initialization": initialization_manifest,
        "use_retrieval_mismatch_expert": bool(
            model.use_retrieval_mismatch_expert
        ),
        "train_retrieval_mismatch_only": bool(
            model.train_retrieval_mismatch_only
        ),
        "use_discrete_event_memory": bool(model.use_discrete_event_memory),
        "train_discrete_event_memory_only": bool(
            model.train_discrete_event_memory_only
        ),
        "body_tail_trainable_parameter_names": list(
            model.body_tail_trainable_parameter_names
        ),
        "tail_time_trainable_parameter_names": list(
            model.tail_time_trainable_parameter_names
        ),
        "retrieval_mismatch_trainable_parameter_names": list(
            model.retrieval_mismatch_trainable_parameter_names
        ),
        "discrete_event_trainable_parameter_names": list(
            model.discrete_event_trainable_parameter_names
        ),
        "residual_scaling_method": str(
            residual_scale.get("method", "per_station_std")
        ),
        "ramp_auxiliary_loss_weight": float(
            model.diffusion.ramp_auxiliary_loss_weight
        ),
        "ramp_auxiliary_lags": list(model.diffusion.ramp_auxiliary_lags),
        "ramp_auxiliary_lag_weights": list(
            model.diffusion.ramp_auxiliary_lag_weights
        ),
        "wind_common_event_loss_weight": float(
            model.diffusion.wind_common_event_loss_weight
        ),
        "event_weighting_method": (
            str(event_weighting.get("method"))
            if event_weighting is not None
            else None
        ),
        "event_replay_method": (
            str(event_replay.get("method"))
            if event_replay is not None
            else None
        ),
        "event_replay_independent_event_count": (
            int(event_replay.get("independent_event_count", 0))
            if event_replay is not None
            else 0
        ),
        "event_replay_expected_draws_per_epoch": (
            float(event_replay.get("expected_event_draws_per_epoch", 0.0))
            if event_replay is not None
            else 0.0
        ),
        "event_replay_observed_draws_mean": (
            float(np.mean([row["event_replay_draws"] for row in history]))
            if event_replay is not None and history
            else 0.0
        ),
        "event_x0_magnitude_loss_weight": float(
            model.diffusion.event_x0_magnitude_loss_weight
        ),
        "event_x0_timing_loss_weight": float(
            model.diffusion.event_x0_timing_loss_weight
        ),
        "event_x0_sync_loss_weight": float(
            model.diffusion.event_x0_sync_loss_weight
        ),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_fixed_noise_validation_mse": best_val,
        "training_seed": seed,
        "validation_seed": validation_seed,
        "ema": {
            "max_decay": ema_decay,
            "warmup": copy.deepcopy(dict(ema_warmup)) if ema_warmup else None,
            "optimizer_updates": optimizer_updates,
            "current_decay": current_ema_decay,
        },
        "test_used": False,
        "graph_manifest": graph_manifest,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(final_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TRAIN_COMPLETE spatial_mode={model.spatial_mode}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
