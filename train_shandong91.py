"""Formal train/validation entry for Shandong91 heterogeneous Raw Body v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping

# Required by CUDA >= 10.2 for deterministic CuBLAS operations.  It must be
# present before the first CUDA BLAS call in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.shandong91_reliable import CHANNELS, Shandong91ReliableDataset
from src.models.shandong91_conditioned_diffusion import Shandong91HeterogeneousRawBody, Shandong91MaskedDiffusion


MODEL_ID = Shandong91HeterogeneousRawBody.architecture


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = False


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    for key in ("actual", "forecast", "residual", "time_mark"):
        result[key] = result[key].float()
    return result


def build_model(config: Mapping[str, Any], dataset: Shandong91ReliableDataset, device: torch.device) -> Shandong91MaskedDiffusion:
    denoiser = Shandong91HeterogeneousRawBody(
        config["model"], dataset.node_features.to(device), dataset.adjacency_with_self.to(device)
    ).to(device)
    return Shandong91MaskedDiffusion(denoiser, config["diffusion"]).to(device)


def atomic_save(document: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(document), temporary)
    os.replace(temporary, path)


def checkpoint_document(model, optimizer, scaler, epoch, global_step, best_val, config, seed, loader_generator):
    return {
        "checkpoint_version": "shandong91_formal_v1_epoch_boundary",
        "model_identifier": MODEL_ID, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": None,
        "amp_scaler_state_dict": scaler.state_dict(), "epoch": int(epoch),
        "global_step": int(global_step), "best_validation_metric": float(best_val),
        "config_snapshot": config, "random_seed": int(seed), "rng_state": rng_state(),
        "train_loader_generator_state": loader_generator.get_state(),
        "resume_level": "exact deterministic at completed epoch boundary",
    }


def load_checkpoint(path, model, optimizer, scaler, config, loader_generator, device):
    # RNG states must remain CPU ByteTensors. Model/optimizer loaders copy their
    # own tensors to the destination parameter devices.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("model_identifier") != MODEL_ID or saved.get("config_snapshot", {}).get("model") != config.get("model"):
        raise ValueError("resume checkpoint model/config is incompatible")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    scaler.load_state_dict(saved["amp_scaler_state_dict"])
    if saved.get("scheduler_state_dict") is not None:
        raise ValueError("v1 config declares scheduler=none")
    restore_rng(saved["rng_state"]); loader_generator.set_state(saved["train_loader_generator_state"].cpu())
    return saved


def epoch_pass(model, loader, device, *, optimizer=None, scaler=None, amp=False, seed=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    if not training and seed is not None: torch.manual_seed(seed)
    totals = {"overall": 0.0, **{name: 0.0 for name in CHANNELS}}
    counts = {"overall": 0, **{name: 0 for name in CHANNELS}}
    steps = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, raw in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches: break
            batch = move_batch(raw, device); mask = batch["effective_mask"].bool()
            timestep = torch.randint(0, model.alpha_hat.numel(), (batch["residual"].shape[0],), device=device)
            noise = torch.randn_like(batch["residual"])
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, element_loss = model.prediction_and_error(batch, timestep, noise)
                loss = model.masked_loss(element_loss, mask)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite masked diffusion loss")
            if training:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("missing or non-finite gradient")
                scaler.step(optimizer); scaler.update(); steps += 1
            total_count = int(mask.sum()); totals["overall"] += float((element_loss.detach() * mask).sum()); counts["overall"] += total_count
            for channel, name in enumerate(CHANNELS):
                current = mask[..., channel]; totals[name] += float((element_loss[..., channel].detach() * current).sum()); counts[name] += int(current.sum())
    if counts["overall"] == 0: raise RuntimeError("loader produced no supervised elements")
    return {name: totals[name] / max(counts[name], 1) for name in totals}, steps


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def write_manifest(path: Path, config: Mapping[str, Any], model, datasets, output_dir: Path, device, resume):
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    manifest = {
        "timestamp_unix": time.time(), "git_commit": git_value("rev-parse", "HEAD"),
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status.strip()),
        "working_tree_status": status.splitlines(), "dataset_identifier": config["data"]["dataset_identifier"],
        "split_sizes": {name: len(value) for name, value in datasets.items()}, "model_identifier": MODEL_ID,
        "model_parameter_count": sum(p.numel() for p in model.parameters()), "diffusion": config["diffusion"],
        "batch_size": config["training"]["batch_size"], "learning_rate": config["training"]["learning_rate"],
        "optimizer": config["training"]["optimizer"], "scheduler": config["training"]["scheduler"],
        "seed": config["training"]["seed"], "amp": config["training"]["mixed_precision"],
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "checkpoint_source": str(resume) if resume else None, "output_directory": str(output_dir.resolve()),
        "config_snapshot": config,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume"); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-dirty", action="store_true"); parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output_dir)
    if output.exists() and not args.resume: raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    if config["training"]["scheduler"] != "none": raise ValueError("only audited scheduler=none is allowed")
    if config["loss"]["formal_objective"] != "elementwise_effective_masked_mean": raise ValueError("formal objective changed")
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is required but unavailable")
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if tracked_status and not args.allow_dirty:
        raise RuntimeError("formal launcher requires a clean working tree")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    seed = int(config["training"]["seed"]); set_seed(seed)
    datasets = {split: Shandong91ReliableDataset(config["data"]["data_path"], split) for split in ("train", "validation", "test")}
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"]["num_workers"]), generator=generator, pin_memory=device.type == "cuda")
    val_loader = DataLoader(datasets["validation"], batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"]["num_workers"]), pin_memory=device.type == "cuda")
    model = build_model(config, datasets["train"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    start_epoch, global_step, best_val = 1, 0, float("inf")
    if args.resume:
        saved = load_checkpoint(Path(args.resume), model, optimizer, scaler, config, generator, device)
        start_epoch = int(saved["epoch"]) + 1; global_step = int(saved["global_step"]); best_val = float(saved["best_validation_metric"])
    write_manifest(output / "manifest.json", config, model, datasets, output, device, args.resume)
    epochs = int(args.max_epochs or config["training"]["epochs"]); history = []
    history_path = output / "training_history.json"
    if args.resume and history_path.exists(): history = json.loads(history_path.read_text(encoding="utf-8"))
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, updates = epoch_pass(model, train_loader, device, optimizer=optimizer, scaler=scaler, amp=amp, max_batches=args.max_train_batches)
        global_step += updates
        val_metrics, _ = epoch_pass(model, val_loader, device, amp=amp, seed=int(config["training"]["validation_seed"]), max_batches=args.max_val_batches)
        row = {"epoch": epoch, "global_step": global_step, "train": train_metrics, "validation": val_metrics}
        history.append(row); history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        improved = val_metrics["overall"] < best_val
        if improved: best_val = val_metrics["overall"]
        document = checkpoint_document(model, optimizer, scaler, epoch, global_step, best_val, config, seed, generator)
        atomic_save(document, output / "checkpoints" / "last.pt")
        if improved: atomic_save(document, output / "checkpoints" / "best.pt")
        print(json.dumps(row), flush=True)


if __name__ == "__main__": main()
