"""Formal trainer for Shandong91 Raw Body V2 faithful24."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, fit_faithful24_state_thresholds,
    threshold_document, threshold_sha256,
)
from src.models.shandong91_faithful24_diffusion import (
    Shandong91Faithful24Diffusion, Shandong91HeterogeneousRawBodyV2,
)

MODEL_ID = Shandong91HeterogeneousRawBodyV2.architecture
CHANNELS = ("Wind", "Solar", "Load")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available(): torch.backends.cudnn.benchmark = False


def move_batch(raw, device):
    batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
    for key in ("actual", "forecast", "residual", "time_mark", "recent_error", "node_state"):
        batch[key] = batch[key].float()
    return batch


def build_model(config, dataset, device):
    denoiser = Shandong91HeterogeneousRawBodyV2(
        config["model"], dataset.node_features.to(device),
        dataset.adjacency_with_self.float().to(device),
    ).to(device)
    return Shandong91Faithful24Diffusion(denoiser, config["diffusion"]).to(device)


def clone_state(model, *, cpu=True):
    return {
        name: (value.detach().cpu().clone() if cpu else value.detach().clone())
        for name, value in model.state_dict().items()
    }


def update_ema(ema, model, decay):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            source = value.detach()
            if source.is_floating_point():
                ema[name].mul_(decay).add_(source, alpha=1 - decay)
            else:
                ema[name].copy_(source)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_save(document, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(document, temporary); os.replace(temporary, path)


def evaluate(model, loader, device, seed, amp, max_batches=None):
    model.eval(); torch.manual_seed(seed)
    totals = {"overall": 0.0, **{name: 0.0 for name in CHANNELS}}
    counts = {name: 0 for name in totals}
    with torch.no_grad():
        for batch_index, raw in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = move_batch(raw, device); mask = batch["effective_mask"].bool()
            timestep = torch.randint(0, len(model.alpha_hat), (len(batch["residual"]),), device=device)
            noise = torch.randn_like(batch["residual"])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, error = model.prediction_and_error(batch, timestep, noise)
            for channel, name in enumerate(CHANNELS):
                current = mask[..., channel]; totals[name] += float((error[..., channel] * current).sum()); counts[name] += int(current.sum())
            totals["overall"] += float((error * mask).sum()); counts["overall"] += int(mask.sum())
    return {name: totals[name] / max(counts[name], 1) for name in totals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume"); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-dirty", action="store_true"); parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text("utf-8"))
    if config["model"]["architecture"] != MODEL_ID:
        raise ValueError("wrong V2 architecture identifier")
    if config["loss"]["formal_objective"] != "elementwise_effective_masked_mean":
        raise ValueError("formal objective must remain elementwise effective-mask mean")
    if config["training"]["optimizer"] != "AdamW" or config["training"]["scheduler"] != "none":
        raise ValueError("faithful24 training contract requires AdamW and no scheduler")
    if config["training"].get("amp_overflow_policy") != "fail_closed":
        raise ValueError("V2 AMP overflow must fail closed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    tracked = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if tracked and not args.allow_dirty:
        raise RuntimeError("formal V2 training requires clean tracked files")
    output = Path(args.output_dir)
    if output.exists() and not args.resume: raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    seed = int(config["training"]["seed"]); set_seed(seed)
    thresholds = fit_faithful24_state_thresholds(
        config["data"]["data_path"],
        low_quantile=float(config["model"]["state_low_quantile"]),
        high_quantile=float(config["model"]["state_high_quantile"]),
        ramp_quantile=float(config["model"]["state_ramp_quantile"]),
        ramp_lags=tuple(config["model"]["state_ramp_lags"]),
    )
    datasets = {split: Shandong91Faithful24Dataset(config["data"]["data_path"], split, thresholds)
                for split in ("train", "validation")}
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=int(config["training"]["batch_size"]),
                              shuffle=True, num_workers=int(config["training"]["num_workers"]),
                              generator=generator, pin_memory=device.type == "cuda")
    val_loader = DataLoader(datasets["validation"], batch_size=int(config["training"]["batch_size"]),
                            shuffle=False, num_workers=int(config["training"]["num_workers"]),
                            pin_memory=device.type == "cuda")
    model = build_model(config, datasets["train"], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = clone_state(model, cpu=False); start_epoch = 1; global_step = 0
    best_val = float("inf"); best_epoch = 0; history = []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("model_identifier") != MODEL_ID or saved["config_snapshot"]["model"] != config["model"]:
            raise ValueError("resume checkpoint is incompatible")
        if saved["state_threshold_sha256"] != threshold_sha256(thresholds):
            raise ValueError("state thresholds changed")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        ema = {name: value.to(device) for name, value in saved["ema_state_dict"].items()}
        optimizer.load_state_dict(saved["optimizer_state_dict"]); scaler.load_state_dict(saved["amp_scaler_state_dict"])
        restore_rng(saved["rng_state"]); generator.set_state(saved["train_loader_generator_state"].cpu())
        start_epoch = int(saved["epoch"]) + 1; global_step = int(saved["global_step"])
        best_val = float(saved["best_validation_metric"]); best_epoch = int(saved["best_epoch"])
        history_path = output / "training_history.json"
        if history_path.exists(): history = json.loads(history_path.read_text("utf-8"))
    manifest = {
        "model_identifier": MODEL_ID, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "device": str(device), "amp": amp, "config_snapshot": config,
        "state_threshold_sha256": threshold_sha256(thresholds),
        "condition_contract": datasets["train"].condition_manifest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "formal_training": "RUNNING", "created_unix": time.time(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    (output / "state_thresholds.json").write_text(json.dumps(threshold_document(thresholds), indent=2), "utf-8")
    epochs = int(args.max_epochs or config["training"]["epochs"]); accumulation = int(config["training"]["gradient_accumulation_steps"])
    clip = float(config["training"]["gradient_clip_norm"]); validation_every = int(config["training"]["validation_every"])
    patience = int(config["training"]["patience"]); min_delta = float(config["training"]["min_delta"])
    ema_decay = float(config["training"]["ema_decay"])
    for epoch in range(start_epoch, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals = {"overall": 0.0, **{name: 0.0 for name in CHANNELS}}; counts = {name: 0 for name in totals}
        used_batches = 0
        bounded_batches = len(train_loader)
        if args.max_train_batches is not None:
            bounded_batches = min(bounded_batches, args.max_train_batches)
        for batch_index, raw in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches: break
            batch = move_batch(raw, device); mask = batch["effective_mask"].bool()
            timestep = torch.randint(0, len(model.alpha_hat), (len(batch["residual"]),), device=device)
            noise = torch.randn_like(batch["residual"])
            group_start = (used_batches // accumulation) * accumulation
            accumulation_divisor = min(accumulation, bounded_batches - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, error = model.prediction_and_error(batch, timestep, noise)
                loss = model.masked_loss(error, mask) / accumulation_divisor
            if not torch.isfinite(loss): raise FloatingPointError("non-finite V2 loss")
            scaler.scale(loss).backward(); used_batches += 1
            should_step = used_batches % accumulation == 0 or used_batches == bounded_batches
            if should_step:
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("missing/non-finite V2 gradient; fail closed")
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                global_step += 1; update_ema(ema, model, ema_decay)
            totals["overall"] += float((error.detach() * mask).sum()); counts["overall"] += int(mask.sum())
            for channel, name in enumerate(CHANNELS):
                current = mask[..., channel]; totals[name] += float((error[..., channel].detach() * current).sum()); counts[name] += int(current.sum())
        row = {"epoch": epoch, "global_step": global_step,
               "train": {name: totals[name] / max(counts[name], 1) for name in totals}}
        if epoch == 1 or epoch % validation_every == 0 or epoch == epochs:
            raw_state = clone_state(model); model.load_state_dict(ema, strict=True)
            val = evaluate(
                model, val_loader, device,
                int(config["training"]["validation_seed"]), amp,
                max_batches=args.max_val_batches,
            )
            model.load_state_dict(raw_state, strict=True); row["validation_ema"] = val
            improved = val["overall"] < best_val - min_delta
            if improved: best_val = val["overall"]; best_epoch = epoch
            document = {
                "checkpoint_version": "shandong91_faithful24_v2_epoch_boundary",
                "model_identifier": MODEL_ID, "model_state_dict": clone_state(model),
                "ema_state_dict": {name: value.detach().cpu().clone() for name, value in ema.items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "amp_scaler_state_dict": scaler.state_dict(), "scheduler_state_dict": None,
                "epoch": epoch, "global_step": global_step,
                "best_validation_metric": best_val, "best_epoch": best_epoch,
                "config_snapshot": config, "state_threshold_sha256": threshold_sha256(thresholds),
                "state_thresholds": threshold_document(thresholds), "rng_state": rng_state(),
                "train_loader_generator_state": generator.get_state(),
                "resume_level": "exact deterministic at completed epoch boundary",
            }
            atomic_save(document, output / "checkpoints/last.pt")
            if improved: atomic_save(document, output / "checkpoints/best.pt")
            if best_epoch and epoch - best_epoch >= patience:
                history.append(row)
                (output / "training_history.json").write_text(json.dumps(history, indent=2), "utf-8")
                print(json.dumps(row), flush=True)
                break
        history.append(row)
        (output / "training_history.json").write_text(json.dumps(history, indent=2), "utf-8")
        print(json.dumps(row), flush=True)
    manifest["formal_training"] = "COMPLETE"; manifest["best_epoch"] = best_epoch
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__": main()
