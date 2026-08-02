"""Generate and evaluate validation/test scenarios from a station24 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import (
    build_station_daylight_mask,
    get_station_dataloader,
    load_station_static_data,
)
from station_evaluation import evaluate_station_scenarios, save_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--issue-batch-size", type=int, default=None)
    parser.add_argument("--member-chunk-size", type=int, default=None)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "sample_index"
    }


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test split is sealed; pass --allow-test only after model lock")
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config_used.yaml"
    checkpoint_path = run_dir / "checkpoints" / "model_best.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("run directory lacks config_used.yaml or model_best.pt")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluation_config = config["evaluation"]
    data_path = Path(args.data_path or config["data"]["data_path"])
    n_samples = int(args.n_samples or evaluation_config.get("n_samples", 80))
    seed = int(args.seed or evaluation_config.get("generation_seed", 424242))
    issue_batch_size = int(
        args.issue_batch_size or evaluation_config.get("issue_batch_size", 1)
    )
    member_chunk_size = int(
        args.member_chunk_size or evaluation_config.get("member_chunk_size", 10)
    )
    if n_samples <= 0 or member_chunk_size <= 0:
        raise ValueError("n_samples and member_chunk_size must be positive")
    output_dir = Path(
        args.output_dir
        or run_dir / f"generation_{args.split}_n{n_samples}_seed{seed}"
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output_dir}")
    output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("CUDA is required for full generation; use --allow-cpu for smoke tests")
    set_seed(seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    residual_scale = checkpoint["residual_scale"]
    scale = np.asarray(residual_scale["scale"], dtype=np.float32)
    static = load_station_static_data(data_path)
    model = Station24DiffusionModel(
        config["model"], static["station_features"], static["station_adjacency"]
    ).to(device)
    if checkpoint.get("architecture") != model.architecture:
        raise ValueError("checkpoint architecture does not match config")
    if checkpoint.get("spatial_mode") != model.spatial_mode:
        raise ValueError("checkpoint spatial mode does not match config")
    state = checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()

    loader, dataset = get_station_dataloader(
        data_path,
        args.split,
        residual_scale,
        batch_size=issue_batch_size,
        seed=seed,
        num_workers=0,
        condition_config=config["model"],
    )
    daylight_mask, daylight_audit = build_station_daylight_mask(data_path, args.split)
    generated_standardized = []
    generated_residual = []
    raw_actual_scenarios = []
    projected_actual_scenarios = []
    actual_values = []
    forecast_values = []

    print(
        f"GENERATION split={args.split} issues={len(loader.dataset)} "
        f"members={n_samples} chunks={member_chunk_size} device={device}"
    )
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        chunks = []
        remaining = n_samples
        while remaining > 0:
            current = min(member_chunk_size, remaining)
            with torch.no_grad():
                chunks.append(model.generate(batch, n_samples=current).cpu())
            remaining -= current
        standardized = torch.cat(chunks, dim=1).numpy()  # [B,K,S,T]
        residual = standardized * scale[None, None, :, None]
        forecast = raw_batch["forecast"].numpy()  # [B,S,T]
        actual = raw_batch["actual"].numpy()
        raw_scenarios = forecast[:, None, :, :] + residual
        projected = np.clip(raw_scenarios, 0.0, 1.0)

        generated_standardized.append(standardized.transpose(0, 1, 3, 2))
        generated_residual.append(residual.transpose(0, 1, 3, 2))
        raw_actual_scenarios.append(raw_scenarios.transpose(0, 1, 3, 2))
        projected_actual_scenarios.append(projected.transpose(0, 1, 3, 2))
        actual_values.append(actual.transpose(0, 2, 1))
        forecast_values.append(forecast.transpose(0, 2, 1))
        print(f"generated issue batch {batch_index}/{len(loader)}")

    standardized_array = np.concatenate(generated_standardized, axis=0)
    residual_array = np.concatenate(generated_residual, axis=0)
    raw_array = np.concatenate(raw_actual_scenarios, axis=0)
    projected_array = np.concatenate(projected_actual_scenarios, axis=0)
    actual_array = np.concatenate(actual_values, axis=0)
    forecast_array = np.concatenate(forecast_values, axis=0)
    if projected_array.shape[0] != daylight_mask.shape[0]:
        raise ValueError("daylight mask issue count does not match generated scenarios")
    projected_array = np.where(
        daylight_mask[:, None, :, :], projected_array, 0.0
    ).astype(np.float32, copy=False)

    np.save(output_dir / "generated_residual_standardized.npy", standardized_array)
    np.save(output_dir / "generated_residual_normalized.npy", residual_array)
    np.save(output_dir / "actual_scenarios_raw_normalized.npy", raw_array)
    np.save(output_dir / "actual_scenarios_normalized.npy", projected_array)
    np.save(output_dir / "actual_data_normalized.npy", actual_array)
    np.save(output_dir / "forecast_data_normalized.npy", forecast_array)
    np.save(output_dir / "station_daylight_mask.npy", daylight_mask)

    stations = pd.read_csv(data_path / "station_order.csv").sort_values(
        "channel_index"
    ).reset_index(drop=True)
    adjacency = np.load(data_path / "station_adjacency.npy")
    metrics, station_frame, lead_frame = evaluate_station_scenarios(
        projected_array,
        raw_array,
        actual_array,
        forecast_array,
        stations,
        adjacency,
        daylight_mask=daylight_mask,
        interval_levels=tuple(
            float(value)
            for value in evaluation_config.get("quantiles", [0.80, 0.90, 0.95])
        ),
    )
    metrics["run"] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_mse": float(checkpoint["val_loss"]),
        "architecture": model.architecture,
        "spatial_mode": model.spatial_mode,
        "parameter_count": int(checkpoint["parameter_count"]),
        "spatial_gate_values": model.denoiser.spatial_block.gate_values(),
        "condition_variant": str(
            config.get("experiment", {}).get("variant", "baseline")
        ),
        "condition_gate_values": model.condition_gate_values,
        "condition_feature_audit": dataset.condition_audit,
        "split": args.split,
        "n_samples": n_samples,
        "generation_seed": seed,
        "physical_projection": "clip_0_1_and_station_astronomical_solar_night",
        "daylight_audit": daylight_audit,
        "test_used": args.split == "test",
    }
    solar_indices = stations.index[stations.data_type.eq("solar")].to_numpy()
    solar_night = ~daylight_mask[:, :, solar_indices]
    raw_solar = raw_array[..., solar_indices]
    projected_solar = projected_array[..., solar_indices]
    metrics["physical"]["raw_solar_night_nonzero_rate"] = float(
        np.mean(np.abs(raw_solar[solar_night[:, None, :, :].repeat(n_samples, axis=1)]) > 1e-6)
    )
    metrics["physical"]["projected_solar_night_nonzero_rate"] = float(
        np.mean(
            np.abs(
                projected_solar[
                    solar_night[:, None, :, :].repeat(n_samples, axis=1)
                ]
            )
            > 1e-6
        )
    )
    save_evaluation(output_dir, metrics, station_frame, lead_frame)
    (output_dir / "generation_metadata.json").write_text(
        json.dumps(metrics["run"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"GENERATION_COMPLETE spatial_mode={model.spatial_mode}")
    print(f"RESULT_DIR={output_dir}")


if __name__ == "__main__":
    main()
