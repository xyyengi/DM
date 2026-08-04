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
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import (
    fit_station_event_weighting,
    fit_station_residual_scale,
    fit_station_state_thresholds,
    get_station_dataloader,
    load_station_static_data,
    write_residual_scale,
    write_station_event_weighting,
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
) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value.detach())


def state_to_cpu(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in state.items()}


@torch.no_grad()
def validate(
    model: Station24DiffusionModel,
    loader,
    device: torch.device,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total_loss = 0.0
    total_samples = 0
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
        loss = model(
            batch,
            timestep=timestep,
            noise=noise,
            include_auxiliary=False,
        )
        total_loss += float(loss) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


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
        "state_gate_values": model.state_gate_values,
        "wind_common_gate_value": model.wind_common_gate_value,
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
    static = load_station_static_data(data_path)
    model = Station24DiffusionModel(
        config["model"],
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    amp_enabled = bool(train_config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema_decay = float(train_config.get("ema_decay", 0.999))
    ema_state = create_ema(model)
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
        f"condition_variant={config.get('experiment', {}).get('variant', 'baseline')} "
        f"residual_scaling={residual_scale.get('method', 'per_station_std')} "
        f"ramp_aux_weight={model.diffusion.ramp_auxiliary_loss_weight} "
        f"common_event_weight={model.diffusion.wind_common_event_loss_weight} "
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
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            batch_size = batch["forecast"].shape[0]
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = model(batch)
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % accumulation == 0 or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema_state, model, ema_decay)
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
        train_loss = total_loss / max(total_samples, 1)
        row: dict[str, float] = {"epoch": epoch, "train_loss": train_loss}

        should_validate = epoch == 1 or epoch % validation_every == 0 or epoch == epochs
        if should_validate:
            val_loss = validate(model, val_loader, device, validation_seed)
            row["val_loss"] = val_loss
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
                )
            print(
                f"epoch={epoch:04d} train={train_loss:.7f} val={val_loss:.7f} "
                f"best_epoch={best_epoch} spatial_gates={model.denoiser.spatial_block.gate_values()} "
                f"condition_gates={model.condition_gate_values}"
                f" state_gates={model.state_gate_values}"
            )
            if best_epoch and epoch - best_epoch >= patience:
                history.append(row)
                print(f"EARLY_STOP best_epoch={best_epoch} best_val={best_val:.7f}")
                break
        else:
            print(f"epoch={epoch:04d} train={train_loss:.7f}")
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
        "spatial_gate_values": model.spatial_gate_values,
        "parallel_spatial_gate_statistics": (
            model.parallel_spatial_gate_statistics
        ),
        "state_gate_values": model.state_gate_values,
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
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_fixed_noise_validation_mse": best_val,
        "training_seed": seed,
        "validation_seed": validation_seed,
        "test_used": False,
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(final_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"TRAIN_COMPLETE spatial_mode={model.spatial_mode}")
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
