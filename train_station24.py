"""Train one 24-station conditional diffusion ablation on train/validation only."""

from __future__ import annotations

import argparse
import copy
import json
import random
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
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        batch_size = batch["forecast"].shape[0]
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
    if model.use_body_tail_experts:
        if total_weight <= 0.0 or tail_event_count <= 0:
            raise ValueError("validation split contains no train-threshold tail event")
        tail_epsilon = total_loss / total_weight
        gate_bce = gate_error_sum / max(gate_samples, 1)
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
        "tail_gate_loss_weight": float(model.tail_gate_loss_weight),
        "tail_common_gate_value": model.tail_common_gate_value,
        "body_tail_trainable_parameter_names": list(
            model.body_tail_trainable_parameter_names
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


def plot_losses(history: list[dict[str, float]], output: Path) -> None:
    frame_epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_epochs = [row["epoch"] for row in history if "val_loss" in row]
    val_losses = [row["val_loss"] for row in history if "val_loss" in row]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(frame_epochs, train_losses, label="train")
    axis.plot(val_epochs, val_losses, marker="o", label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Fixed-noise epsilon MSE")
    axis.set_title("Station24 diffusion training")
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
        event_replay = fit_station_event_replay(data_path, config["model"])
        write_station_event_replay(
            run_dir / "event_replay.json", event_replay
        )
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    train_loader, train_dataset = get_station_dataloader(
        data_path,
        "train",
        residual_scale,
        batch_size=int(train_config["batch_size"]),
        seed=seed,
        num_workers=int(train_config.get("num_workers", 0)),
        condition_config=config["model"],
        state_thresholds=state_thresholds,
        event_weighting=event_weighting,
        event_replay=event_replay,
    )
    val_loader, val_dataset = get_station_dataloader(
        data_path,
        "val",
        residual_scale,
        batch_size=int(train_config["batch_size"]),
        seed=int(train_config.get("validation_seed", 314159)),
        num_workers=int(train_config.get("num_workers", 0)),
        condition_config=config["model"],
        state_thresholds=state_thresholds,
        event_weighting=event_weighting,
        event_replay=event_replay,
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
            "source_condition_variant": str(initialization["condition_variant"]),
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema_decay = float(train_config.get("ema_decay", 0.999))
    ema_state = create_ema(model)
    ema_trainable_state_names = (
        set(model.body_tail_state_dict_keys)
        if model.use_body_tail_experts
        else None
    )
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

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        event_draws = 0
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            batch_size = batch["forecast"].shape[0]
            if event_replay is not None:
                event_draws += int(batch["event_active"].sum().detach().cpu())
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = model(batch)
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
                update_ema(
                    ema_state,
                    model,
                    ema_decay,
                    trainable_state_names=ema_trainable_state_names,
                )
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
        train_loss = total_loss / max(total_samples, 1)
        row: dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "event_replay_draws": float(event_draws),
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
                )
            print(
                f"epoch={epoch:04d} train={train_loss:.7f} val={val_loss:.7f} "
                f"best_epoch={best_epoch} spatial_gates={model.denoiser.spatial_block.gate_values()} "
                f"event_draws={event_draws} "
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
                f"event_draws={event_draws}"
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
            )

    if not (checkpoint_dir / "model_best.pt").is_file():
        raise RuntimeError("training ended without a best checkpoint")
    (log_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_losses(history, log_dir / "loss_curve.png")
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
        "tail_gate_loss_weight": float(model.tail_gate_loss_weight),
        "tail_common_gate_value": model.tail_common_gate_value,
        "body_tail_initialization": initialization_manifest,
        "body_tail_trainable_parameter_names": list(
            model.body_tail_trainable_parameter_names
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
