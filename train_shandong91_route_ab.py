"""Bounded Route-A/Route-B trainer. It cannot overwrite V2 and is not a formal launcher."""
from __future__ import annotations

import argparse, copy, json, os, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from datasets.shandong91_faithful24 import Shandong91Faithful24Dataset, fit_faithful24_state_thresholds, threshold_document
from datasets.shandong91_system_load import fit_system_load_contract, SystemLoadTransform
from src.models.shandong91_faithful24_diffusion import Shandong91Faithful24Diffusion, Shandong91HeterogeneousRawBodyV2
from src.models.shandong91_route_variants import Shandong91V2MultiscaleGraph, SystemLoadDiffusionAdapter
from train_shandong91_v2 import move_batch, set_seed, clone_state, update_ema, atomic_save


def build(config, dataset, device):
    architecture = config["model"]["architecture"]
    model_config = copy.deepcopy(config["model"])
    model_config["architecture"] = model_config.get("denoiser_architecture", architecture)
    if architecture == Shandong91V2MultiscaleGraph.architecture:
        denoiser = Shandong91V2MultiscaleGraph(model_config, dataset.node_features.to(device), dataset.adjacency_with_self.float().to(device))
        return Shandong91Faithful24Diffusion(denoiser, config["diffusion"]).to(device), None
    if architecture == SystemLoadDiffusionAdapter.architecture:
        denoiser = Shandong91HeterogeneousRawBodyV2(model_config, dataset.node_features.to(device), dataset.adjacency_with_self.float().to(device))
        diffusion = Shandong91Faithful24Diffusion(denoiser, config["diffusion"]).to(device)
        contract = fit_system_load_contract(config["data"]["data_path"])
        return SystemLoadDiffusionAdapter(diffusion, SystemLoadTransform(contract, dataset.scales)).to(device), contract
    raise ValueError(f"unsupported isolated route architecture: {architecture}")


def batch_loss(model, batch, route):
    timestep = torch.randint(0, 500, (len(batch["residual"]),), device=batch["residual"].device)
    node_noise = torch.randn_like(batch["residual"])
    if route == "A":
        system_noise = torch.randn(len(batch["residual"]), 168, device=batch["residual"].device)
        _, _, errors = model.prediction_and_error(batch, timestep, node_noise, system_noise)
        return model.masked_loss(errors, batch["effective_mask"])
    _, error = model.prediction_and_error(batch, timestep, node_noise)
    return model.masked_loss(error, batch["effective_mask"])


def main():
    p = argparse.ArgumentParser(); p.add_argument("--route", choices=("A","B"), required=True)
    p.add_argument("--config", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=("cpu","cuda"), default="cuda"); p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--max-train-batches", type=int); p.add_argument("--resume")
    a = p.parse_args(); config = yaml.safe_load(Path(a.config).read_text("utf-8"))
    if os.environ.get("SHANDONG91_DATA"):
        config["data"]["data_path"] = os.environ["SHANDONG91_DATA"]
    expected = SystemLoadDiffusionAdapter.architecture if a.route == "A" else Shandong91V2MultiscaleGraph.architecture
    if config["model"]["architecture"] != expected: raise ValueError("route/config mismatch")
    if a.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; Gate 1 not run")
    output = Path(a.output_dir)
    if output.exists() and not a.resume: raise FileExistsError(f"refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=bool(a.resume)); (output / "checkpoints").mkdir(exist_ok=True)
    device = torch.device("cuda:0" if a.device == "cuda" else "cpu"); set_seed(int(config["training"]["seed"]))
    thresholds = fit_faithful24_state_thresholds(config["data"]["data_path"])
    dataset = Shandong91Faithful24Dataset(config["data"]["data_path"], "train", thresholds)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=0)
    model, load_contract = build(config, dataset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    ema = clone_state(model, cpu=False); start = 1; history = []
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=False)
        if saved["architecture"] != expected: raise ValueError("checkpoint route mismatch")
        model.load_state_dict(saved["model"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        ema = {k:v.to(device) for k,v in saved["ema"].items()}; start = int(saved["epoch"]) + 1
    manifest = {"scope":"bounded short experiment; not formal training", "route":a.route,
        "architecture":expected, "device":str(device), "amp":device.type=="cuda",
        "load_contract":load_contract, "state_thresholds":threshold_document(thresholds)}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),"utf-8")
    amp = device.type == "cuda"; scaler = torch.amp.GradScaler(device.type, enabled=amp)
    for epoch in range(start, a.max_epochs + 1):
        begun=time.perf_counter(); model.train(); total=0.; count=0
        if amp: torch.cuda.reset_peak_memory_stats(device)
        for i, raw in enumerate(loader):
            if a.max_train_batches is not None and i >= a.max_train_batches: break
            batch=move_batch(raw,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): loss=batch_loss(model,batch,a.route)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite route loss")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError("missing/non-finite route gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["training"]["gradient_clip_norm"]))
            scaler.step(optimizer); scaler.update(); update_ema(ema,model,float(config["training"]["ema_decay"]))
            total += float(loss.detach()); count += 1
        row={"epoch":epoch,"loss":total/max(count,1),"batches":count,"epoch_seconds":time.perf_counter()-begun}
        if amp: row["cuda_peak_allocated_gb"]=torch.cuda.max_memory_allocated(device)/2**30
        history.append(row); print(json.dumps(row),flush=True)
        doc={"architecture":expected,"route":a.route,"epoch":epoch,"model":clone_state(model),"ema":{k:v.cpu() for k,v in ema.items()},"optimizer":optimizer.state_dict(),"config":config,"load_contract":load_contract}
        atomic_save(doc,output/"checkpoints"/"last.pt"); atomic_save(doc,output/"checkpoints"/"best.pt")
        (output/"training_history.json").write_text(json.dumps(history,indent=2),"utf-8")

if __name__ == "__main__": main()
