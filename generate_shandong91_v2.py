"""Generate isolated Shandong91 faithful24 V2 Raw Body scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
import yaml

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, threshold_sha256,
)
from train_shandong91_v2 import MODEL_ID, build_model, move_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/shandong91/raw_body_v2_faithful24.yaml")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--member-chunk", type=int, default=2)
    parser.add_argument("--method", choices=("ddpm", "ddim"))
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--checkpoint-state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-windows", type=int,
                        help="bounded engineering check only; omit for formal generation")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 generation output: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal V2 generation requires available CUDA")
    config = yaml.safe_load(Path(args.config).read_text("utf-8"))
    if config["model"]["architecture"] != MODEL_ID:
        raise ValueError("wrong V2 config architecture")
    checkpoint = Path(args.run_dir) / "checkpoints" / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_identifier") != MODEL_ID:
        raise ValueError("checkpoint model identifier mismatch")
    if saved.get("config_snapshot", {}).get("model") != config["model"]:
        raise ValueError("checkpoint/config model mismatch")
    thresholds = saved["state_thresholds"]
    dataset = Shandong91Faithful24Dataset(config["data"]["data_path"], args.split, thresholds)
    if threshold_sha256(dataset.thresholds) != saved["state_threshold_sha256"]:
        raise ValueError("checkpoint state-threshold hash mismatch")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    model = build_model(config, dataset, device)
    state_key = "ema_state_dict" if args.checkpoint_state == "ema" else "model_state_dict"
    model.load_state_dict(saved[state_key], strict=True)
    model.eval()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    method = args.method or config["sampling"]["method"]
    steps = int(args.inference_steps or config["sampling"]["inference_steps"])
    if method == "ddpm" and steps != int(config["diffusion"]["num_steps"]):
        raise ValueError("DDPM must use the complete training diffusion schedule")
    count_windows = min(len(dataset), args.max_windows) if args.max_windows is not None else len(dataset)
    shape = (count_windows, args.n_samples, 168, 91, 3)
    scenarios = np.empty(shape, dtype=np.float32)
    actual = np.empty((count_windows, 168, 91, 3), dtype=np.float32)
    forecast = np.empty_like(actual)
    publication_mask = np.empty_like(actual, dtype=bool)
    effective_mask = np.empty_like(actual, dtype=bool)
    node = dataset.node_type_mask.to(device).view(1, 1, 91, 3)
    train = dataset.channel_train_mask.to(device).view(1, 1, 91, 3)

    for issue in range(count_windows):
        raw = dataset[issue]
        batch = move_batch({key: value.unsqueeze(0) for key, value in raw.items()}, device)
        publication = node & batch["forecast_valid_mask"].bool()
        generation = publication & train
        effective = batch["effective_mask"].bool()
        actual[issue] = dataset.denormalize_mw(batch["actual"])[0].cpu().numpy()
        forecast_mw = dataset.denormalize_mw(batch["forecast"])
        forecast[issue] = forecast_mw[0].cpu().numpy()
        publication_mask[issue] = publication[0].cpu().numpy()
        effective_mask[issue] = effective[0].cpu().numpy()
        for start in range(0, args.n_samples, args.member_chunk):
            members = min(args.member_chunk, args.n_samples - start)
            expanded = {
                key: value.expand(members, *value.shape[1:])
                for key, value in batch.items()
            }
            generation_expanded = generation.expand(members, -1, -1, -1)
            initial_noise = torch.randn_like(expanded["forecast"])
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                residual = model.sample(
                    expanded, generation_expanded, method=method,
                    inference_steps=steps, initial_noise=initial_noise,
                ).float()
            generated = dataset.denormalize_mw(expanded["forecast"] + residual)
            generated.masked_fill_(~publication.expand_as(generated), 0.0)
            scenarios[issue, start:start + members] = generated.cpu().numpy()
        print(json.dumps({"window": issue + 1, "total": count_windows}), flush=True)

    if not np.isfinite(scenarios).all():
        raise FloatingPointError("generated V2 scenarios contain NaN/Inf")
    output.mkdir(parents=True)
    np.save(output / "actual_scenarios_mw.npy", scenarios)
    np.save(output / "actual_mw.npy", actual)
    np.save(output / "forecast_mw.npy", forecast)
    np.save(output / "valid_mask.npy", publication_mask)
    np.save(output / "effective_mask.npy", effective_mask)
    metadata = {
        "model_identifier": MODEL_ID,
        "checkpoint": str(checkpoint), "checkpoint_epoch": int(saved["epoch"]),
        "checkpoint_state": args.checkpoint_state,
        "split": args.split, "n_samples": args.n_samples, "seed": args.seed,
        "shape": list(shape), "sampler": method, "inference_steps": steps,
        "residual_sign": "generated_actual = forecast + generated_residual",
        "state_threshold_sha256": saved["state_threshold_sha256"],
        "inactive_policy": config["sampling"]["inactive_policy"],
        "physical_clipping": False,
        "bounded_engineering_generation": args.max_windows is not None,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), "utf-8")


if __name__ == "__main__":
    main()
