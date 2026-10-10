"""Formal trainer for Shandong91 V3 low-rank common-factor Raw Body."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader
import yaml

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, fit_faithful24_state_thresholds,
    threshold_document, threshold_sha256,
)
from datasets.shandong91_low_rank import fit_train_only_pca, factor_sha256
from src.models.shandong91_low_rank_diffusion import (
    Shandong91HeterogeneousRawBodyV3, Shandong91LowRankDiffusion,
)
from train_shandong91_v2 import (
    atomic_save, clone_state, data_contract_record, loader_kwargs, move_batch,
    restore_rng, rng_state, set_seed, update_ema,
)

MODEL_ID = Shandong91HeterogeneousRawBodyV3.architecture
CHANNELS = ("Wind", "Solar", "Load")


def build_model(config, dataset, device, factor_document=None):
    factor_document = factor_document or fit_train_only_pca(
        config["data"]["data_path"], config["model"]["common_factor_counts"]
    )
    denoiser = Shandong91HeterogeneousRawBodyV3(
        config["model"], dataset.node_features.to(device),
        dataset.adjacency_with_self.float().to(device),
    ).to(device)
    return Shandong91LowRankDiffusion(
        denoiser, config["diffusion"], factor_document
    ).to(device)


def evaluate(model, loader, device, seed, amp, max_batches=None):
    model.eval(); torch.manual_seed(seed)
    totals = {"overall": 0.0, **{name: 0.0 for name in CHANNELS}}
    counts = {name: 0 for name in totals}
    with torch.no_grad():
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = move_batch(raw, device); mask = batch["effective_mask"].bool()
            timestep = torch.randint(0, len(model.alpha_hat), (len(mask),), device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, error = model.prediction_and_error(batch, timestep)
            totals["overall"] += float((error * mask).sum()); counts["overall"] += int(mask.sum())
            for channel, name in enumerate(CHANNELS):
                current = mask[..., channel]
                totals[name] += float((error[..., channel] * current).sum())
                counts[name] += int(current.sum())
    return {name: totals[name] / max(counts[name], 1) for name in totals}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume"); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-dirty", action="store_true"); parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int); parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text("utf-8"))
    if config["model"]["architecture"] != MODEL_ID:
        raise ValueError("wrong V3 architecture identifier")
    if config["loss"]["formal_objective"] != "elementwise_effective_masked_mean":
        raise ValueError("V3 retains the unweighted elementwise effective-mask objective")
    if config["training"]["optimizer"] != "AdamW" or config["training"]["scheduler"] != "none":
        raise ValueError("V3 requires AdamW and no scheduler")
    if config["training"].get("amp_overflow_policy") != "fail_closed":
        raise ValueError("V3 AMP overflow must fail closed")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if tracked and not args.allow_dirty:
        raise RuntimeError("formal V3 training requires clean tracked files")
    output = Path(args.output_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    seed = int(config["training"]["seed"]); set_seed(seed)
    factor_document = fit_train_only_pca(
        config["data"]["data_path"], config["model"]["common_factor_counts"]
    )
    thresholds = fit_faithful24_state_thresholds(
        config["data"]["data_path"],
        low_quantile=float(config["model"]["state_low_quantile"]),
        high_quantile=float(config["model"]["state_high_quantile"]),
        ramp_quantile=float(config["model"]["state_ramp_quantile"]),
        ramp_lags=tuple(config["model"]["state_ramp_lags"]),
    )
    datasets = {split: Shandong91Faithful24Dataset(config["data"]["data_path"], split, thresholds)
                for split in ("train", "validation")}
    training = config["training"]; microbatch = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    if microbatch * accumulation != int(training["effective_batch_size"]):
        raise ValueError("batch_size * gradient_accumulation_steps != effective_batch_size")
    generator = torch.Generator().manual_seed(seed); common = loader_kwargs(training, device)
    train_loader = DataLoader(datasets["train"], batch_size=microbatch, shuffle=True,
                              generator=generator, **common)
    val_loader = DataLoader(datasets["validation"], batch_size=microbatch, shuffle=False, **common)
    model = build_model(config, datasets["train"], device, factor_document)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ema = clone_state(model, cpu=False); start_epoch = 1; global_step = 0
    best_val = float("inf"); best_epoch = 0; history = []
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("model_identifier") != MODEL_ID or saved["config_snapshot"]["model"] != config["model"]:
            raise ValueError("resume checkpoint is incompatible")
        if saved.get("data_contract") != data_contract_record(config["data"]):
            raise ValueError("resume data contract changed")
        if factor_sha256(saved["factor_document"]) != factor_document["sha256"]:
            raise ValueError("train-only PCA contract changed")
        if saved["state_threshold_sha256"] != threshold_sha256(thresholds):
            raise ValueError("state thresholds changed")
        resume_keys = (
            "batch_size", "gradient_accumulation_steps", "effective_batch_size",
            "optimizer", "learning_rate", "weight_decay", "gradient_clip_norm",
            "ema_decay", "mixed_precision", "amp_overflow_policy",
        )
        saved_training = saved["config_snapshot"]["training"]
        if any(saved_training.get(key) != training.get(key) for key in resume_keys):
            raise ValueError("resume training semantics changed; start an isolated run")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        ema = {name: value.to(device) for name, value in saved["ema_state_dict"].items()}
        optimizer.load_state_dict(saved["optimizer_state_dict"]); scaler.load_state_dict(saved["amp_scaler_state_dict"])
        restore_rng(saved["rng_state"]); generator.set_state(saved["train_loader_generator_state"].cpu())
        start_epoch = int(saved["epoch"]) + 1; global_step = int(saved["global_step"])
        best_val = float(saved["best_validation_metric"]); best_epoch = int(saved["best_epoch"])
        history_path = output / "training_history.json"
        if history_path.exists():
            history = [row for row in json.loads(history_path.read_text("utf-8"))
                       if int(row["epoch"]) <= int(saved["epoch"])]
    manifest = {
        "model_identifier": MODEL_ID,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "device": str(device), "amp": amp, "config_snapshot": config,
        "data_contract": data_contract_record(config["data"]),
        "state_threshold_sha256": threshold_sha256(thresholds),
        "factor_document_sha256": factor_document["sha256"],
        "factor_counts": config["model"]["common_factor_counts"],
        "load_local_branch": "disabled_strict_rank1",
        "condition_contract": datasets["train"].condition_manifest(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "formal_training": "RUNNING", "created_unix": time.time(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    (output / "state_thresholds.json").write_text(json.dumps(threshold_document(thresholds), indent=2), "utf-8")
    (output / "train_only_pca_factors.json").write_text(json.dumps(factor_document, indent=2), "utf-8")
    epochs = int(args.max_epochs or training["epochs"]); clip = float(training["gradient_clip_norm"])
    validation_every = int(training["validation_every"]); patience = int(training["patience"])
    min_delta = float(training["min_delta"]); ema_decay = float(training["ema_decay"])
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        totals = {"overall": 0.0, **{name: 0.0 for name in CHANNELS}}; counts = {name: 0 for name in totals}
        used = 0; bounded = min(len(train_loader), args.max_train_batches or len(train_loader))
        for index, raw in enumerate(train_loader):
            if index >= bounded: break
            batch = move_batch(raw, device); mask = batch["effective_mask"].bool()
            timestep = torch.randint(0, len(model.alpha_hat), (len(mask),), device=device)
            divisor = min(accumulation, bounded - (used // accumulation) * accumulation)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                _, error = model.prediction_and_error(batch, timestep)
                loss = model.masked_loss(error, mask) / divisor
            if not torch.isfinite(loss): raise FloatingPointError("non-finite V3 loss")
            scaler.scale(loss).backward(); used += 1
            if used % accumulation == 0 or used == bounded:
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("missing/non-finite V3 gradient; fail closed")
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                global_step += 1; update_ema(ema, model, ema_decay)
            totals["overall"] += float((error.detach() * mask).sum()); counts["overall"] += int(mask.sum())
            for channel, name in enumerate(CHANNELS):
                current = mask[..., channel]
                totals[name] += float((error[..., channel].detach() * current).sum())
                counts[name] += int(current.sum())
        row = {"epoch": epoch, "global_step": global_step,
               "train": {name: totals[name] / max(counts[name], 1) for name in totals},
               "epoch_seconds": time.perf_counter() - started}
        improved = False; validated = epoch == 1 or epoch % validation_every == 0 or epoch == epochs
        if validated:
            raw_state = clone_state(model); model.load_state_dict(ema, strict=True)
            row["validation_ema"] = evaluate(model, val_loader, device,
                                               int(training["validation_seed"]), amp,
                                               args.max_val_batches)
            model.load_state_dict(raw_state, strict=True)
            improved = row["validation_ema"]["overall"] < best_val - min_delta
            if improved: best_val = row["validation_ema"]["overall"]; best_epoch = epoch
        document = {
            "checkpoint_version": "shandong91_v3_low_rank_epoch_boundary",
            "model_identifier": MODEL_ID, "model_state_dict": clone_state(model),
            "ema_state_dict": {name: value.detach().cpu().clone() for name, value in ema.items()},
            "optimizer_state_dict": optimizer.state_dict(), "amp_scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": None, "epoch": epoch, "global_step": global_step,
            "best_validation_metric": best_val, "best_epoch": best_epoch,
            "config_snapshot": config, "state_threshold_sha256": threshold_sha256(thresholds),
            "state_thresholds": threshold_document(thresholds), "factor_document": factor_document,
            "data_contract": data_contract_record(config["data"]), "rng_state": rng_state(),
            "train_loader_generator_state": generator.get_state(),
            "resume_level": "exact deterministic at completed epoch boundary",
        }
        atomic_save(document, output / "checkpoints/last.pt")
        if improved: atomic_save(document, output / "checkpoints/best.pt")
        history.append(row); (output / "training_history.json").write_text(json.dumps(history, indent=2), "utf-8")
        print(json.dumps(row), flush=True)
        if validated and best_epoch and epoch - best_epoch >= patience: break
    manifest["formal_training"] = "COMPLETE"; manifest["best_epoch"] = best_epoch
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    main()
