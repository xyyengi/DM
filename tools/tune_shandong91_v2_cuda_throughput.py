"""Bounded CUDA/AMP throughput tuner for Shandong91 Raw Body V2.

This is an engineering probe, not training.  It keeps the formal effective
batch size fixed while comparing microbatch/accumulation pairs, benchmarks
DataLoader worker counts, and measures one denoiser call for generation chunk
sizes.  The selected values are written to an isolated resolved config.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader, default_collate
import yaml

from datasets.shandong91_faithful24 import (
    Shandong91Faithful24Dataset, fit_faithful24_state_thresholds,
)
from train_shandong91_v2 import build_model, move_batch, set_seed


def parse_candidates(text: str) -> list[int]:
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("candidates must be positive integers")
    return values


def expanded_sample(dataset, count: int, device: torch.device):
    raw = default_collate([dataset[index % len(dataset)] for index in range(count)])
    return move_batch(raw, device)


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def training_probe(config, dataset, device, microbatch, effective_batch, repeats):
    cleanup()
    if effective_batch % microbatch:
        return {"status": "SKIP", "reason": "does not divide effective batch"}
    accumulation = effective_batch // microbatch
    try:
        set_seed(int(config["training"]["seed"]))
        model = build_model(config, dataset, device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        batch = expanded_sample(dataset, microbatch, device)

        def microstep():
            timestep = torch.randint(0, len(model.alpha_hat), (microbatch,), device=device)
            noise = torch.randn_like(batch["residual"])
            with torch.autocast("cuda", dtype=torch.float16):
                _, error = model.prediction_and_error(batch, timestep, noise)
                loss = model.masked_loss(error, batch["effective_mask"].bool()) / accumulation
            scaler.scale(loss).backward()

        def update():
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation): microstep()
            scaler.unscale_(optimizer)
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("non-finite gradient during throughput probe")
            scaler.step(optimizer); scaler.update()

        optimizer.zero_grad(set_to_none=True)
        microstep(); optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(repeats): update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            "status": "PASS", "microbatch": microbatch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
            "updates_timed": repeats, "seconds": elapsed,
            "samples_per_second": repeats * effective_batch / elapsed,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del model, optimizer, scaler, batch
        cleanup()
        return result
    except torch.OutOfMemoryError as exc:
        cleanup()
        return {"status": "OOM", "microbatch": microbatch, "error": str(exc)}


def generation_probe(config, dataset, device, chunk, repeats):
    cleanup()
    try:
        set_seed(int(config["training"]["seed"]))
        model = build_model(config, dataset, device).eval()
        batch = expanded_sample(dataset, chunk, device)
        timestep = torch.full((chunk,), len(model.alpha_hat) - 1, device=device, dtype=torch.long)
        noisy = torch.randn_like(batch["forecast"])

        def denoise():
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                return model.denoiser(
                    noisy, timestep, batch["forecast"], batch["forecast_valid_mask"],
                    batch["time_mark"], batch["recent_error"],
                    batch["recent_error_valid_mask"], batch["node_state"],
                )

        output = denoise(); torch.cuda.synchronize()
        if not torch.isfinite(output).all():
            raise FloatingPointError("non-finite generation probe output")
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(repeats): output = denoise()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            "status": "PASS", "member_chunk": chunk,
            "denoiser_calls_timed": repeats, "seconds": elapsed,
            "members_per_second_per_diffusion_step": repeats * chunk / elapsed,
            "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        del model, batch, output
        cleanup()
        return result
    except torch.OutOfMemoryError as exc:
        cleanup()
        return {"status": "OOM", "member_chunk": chunk, "error": str(exc)}


def loader_probe(dataset, batch_size, workers, batches):
    kwargs = {"batch_size": batch_size, "shuffle": False, "num_workers": workers}
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **kwargs)
    iterator = iter(loader)
    warmup = min(2, len(loader))
    for _ in range(warmup): next(iterator)
    count = min(batches, len(loader) - warmup)
    started = time.perf_counter(); samples = 0
    for _ in range(count):
        batch = next(iterator); samples += len(batch["residual"])
    elapsed = time.perf_counter() - started
    del iterator, loader
    return {"workers": workers, "batches_timed": count, "seconds": elapsed,
            "samples_per_second": samples / max(elapsed, 1e-12)}


def select_fastest(rows, metric, total_gb, max_fraction):
    eligible = [row for row in rows if row["status"] == "PASS"
                and row["peak_reserved_gb"] <= total_gb * max_fraction]
    if not eligible:
        raise RuntimeError(f"no candidate passed the {max_fraction:.0%} memory-headroom gate")
    return max(eligible, key=lambda row: row[metric])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-candidates", default="2,4,8,16")
    parser.add_argument("--generation-candidates", default="2,4,8,16,32,64")
    parser.add_argument("--worker-candidates", default="2,4,8")
    parser.add_argument("--train-repeats", type=int, default=1)
    parser.add_argument("--generation-repeats", type=int, default=3)
    parser.add_argument("--loader-batches", type=int, default=5)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("throughput tuning requires CUDA")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing existing tuning output: {output}")
    output.mkdir(parents=True)

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text("utf-8"))
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_gb = properties.total_memory / 2**30
    effective_batch = int(config["training"]["effective_batch_size"])
    thresholds = fit_faithful24_state_thresholds(
        config["data"]["data_path"],
        low_quantile=float(config["model"]["state_low_quantile"]),
        high_quantile=float(config["model"]["state_high_quantile"]),
        ramp_quantile=float(config["model"]["state_ramp_quantile"]),
        ramp_lags=tuple(config["model"]["state_ramp_lags"]),
    )
    dataset = Shandong91Faithful24Dataset(config["data"]["data_path"], "train", thresholds)

    train_rows = [training_probe(config, dataset, device, value, effective_batch, args.train_repeats)
                  for value in parse_candidates(args.train_candidates)]
    selected_train = select_fastest(train_rows, "samples_per_second", total_gb, args.max_memory_fraction)
    loader_rows = [loader_probe(dataset, selected_train["microbatch"], value, args.loader_batches)
                   for value in parse_candidates(args.worker_candidates)]
    selected_loader = max(loader_rows, key=lambda row: row["samples_per_second"])
    generation_rows = [generation_probe(config, dataset, device, value, args.generation_repeats)
                       for value in parse_candidates(args.generation_candidates)]
    selected_generation = select_fastest(
        generation_rows, "members_per_second_per_diffusion_step",
        total_gb, args.max_memory_fraction,
    )

    resolved = copy.deepcopy(config)
    resolved["experiment"]["name"] += "_autotuned_cuda"
    resolved["training"]["batch_size"] = selected_train["microbatch"]
    resolved["training"]["gradient_accumulation_steps"] = selected_train["gradient_accumulation_steps"]
    resolved["training"]["num_workers"] = selected_loader["workers"]
    resolved["training"]["persistent_workers"] = True
    resolved["training"]["prefetch_factor"] = 2
    resolved["generation"]["member_chunk"] = selected_generation["member_chunk"]
    resolved["throughput_tuning"] = {
        "source_config": str(config_path),
        "selection_rule": "highest measured throughput with peak reserved memory <= 90% of device memory",
        "scientific_contract_changes": False,
    }
    resolved_path = output / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), "utf-8")
    report = {
        "status": "PASS", "scope": "bounded CUDA/AMP engineering throughput probe; no formal training",
        "device": properties.name, "total_memory_gb": total_gb,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "effective_batch_size_held_fixed": effective_batch,
        "max_memory_fraction": args.max_memory_fraction,
        "training_candidates": train_rows, "selected_training": selected_train,
        "loader_candidates": loader_rows, "selected_loader": selected_loader,
        "generation_candidates": generation_rows, "selected_generation": selected_generation,
        "resolved_config": str(resolved_path), "formal_training": "NOT RUN",
    }
    (output / "throughput_report.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
