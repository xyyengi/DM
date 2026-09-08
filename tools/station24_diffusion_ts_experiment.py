"""Isolated train/generate/preflight entry point; invoke with python -m tools..."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import copy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.station_diffusion_ts import StationDiffusionTS
from src.models.station_diffusion_ts_inherited import StationDiffusionTSInherited
from station_dataset import (fit_station_residual_scale, get_station_dataloader,
                             build_station_daylight_mask, load_station_static_data)
from station_evaluation import evaluate_station_scenarios, save_evaluation
from station_jstd_targets import fit_station_jstd_event_thresholds


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def precision_policy(config, device):
    """Explicit precision; absent dtype preserves legacy CUDA FP16 behavior."""
    name=config["train"].get("amp_dtype","float16")
    if name not in ("float16","bfloat16"):
        raise ValueError(f"unsupported amp_dtype: {name}")
    enabled=device=="cuda" and bool(config["train"].get("amp",True))
    if enabled and name=="bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 is required by this config; device does not support it")
    return enabled,getattr(torch,name),enabled and name=="float16"


@lru_cache(maxsize=4)
def inherited_assets(run, data):
    root=Path(run); manifest=json.loads((root/"graphs/graph_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fit_split") != "train" or manifest.get("validation_actual_used") or manifest.get("test_actual_used"):
        raise ValueError("inherited graph provenance is not train-only")
    primary=root/"graphs/primary_adjacency.npy";secondary=root/"graphs/secondary_adjacency.npy"
    if digest(primary)!=manifest["primary_sha256"] or digest(secondary)!=manifest["secondary_sha256"]:
        raise ValueError("inherited graph hash mismatch")
    state=json.loads((root/"state_thresholds.json").read_text(encoding="utf-8"))
    if state.get("fit_split")!="train":raise ValueError("state thresholds must be train-only")
    static=load_station_static_data(data)
    if not np.allclose(np.load(primary),static["station_adjacency"].numpy()):raise ValueError("geographic graph differs from baseline")
    return {"geographic":np.load(primary),"historical":np.load(secondary),
            "station_features":static["station_features"],"capacities":static["station_capacities"]},state


def build_model(config,data):
    if config["model"]["version"]==StationDiffusionTSInherited.VERSION:
        assets,_=inherited_assets(config["inherit_run"],data)
        return StationDiffusionTSInherited(config["model"],assets)
    return StationDiffusionTS(config["model"])


def condition_keys(model):
    return getattr(model,"CONDITION_KEYS",("forecast","recent_error","recent_error_mask"))


def loader(data, split, scale, size, seed, config=None):
    conditions={"use_recent_error":True,"recent_error_hours":24}
    state=None
    if config and "inherit_run" in config:
        _,state=inherited_assets(config["inherit_run"],data)
        conditions.update(use_state_encoder=True,state_ramp_lags=[3,6])
    return get_station_dataloader(data, split, scale, size, seed, num_workers=0,
        condition_config=conditions,state_thresholds=state)[0]


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def evidence_fingerprint(args):
    paths = [Path(args.config), Path(__file__), Path("src/models/station_diffusion_ts.py"),
             *Path("src/models/diffusion_ts_vendor").glob("*.py"), Path("station_dataset.py")]
    paths += [Path(args.data)/f"{split}_{name}" for split in ("train", "val")
              for name in ("actual.npy", "forecast.npy", "residual.npy", "fill_mask.npy", "issue_dates.csv")]
    paths += [Path(args.data)/name for name in ("station_order.csv","station_features.npy","station_adjacency.npy")]
    paths += [Path(args.data)/f"{split}_{name}.npy" for split in ("train","val") for name in ("time_mark","lead_mark")]
    paths += [Path("src/models/station_diffusion_ts_inherited.py")]
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if "inherit_run" in config:
        paths += [Path(config["inherit_run"])/name for name in ("graphs/graph_manifest.json","graphs/primary_adjacency.npy","graphs/secondary_adjacency.npy","state_thresholds.json")]
    return {str(p):digest(p) for p in paths}


def preflight(args, config, model, scale):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    report = {"checks": checks, "launch_eligible": False, "device": args.device,
              "version": model.VERSION, "config_sha256": digest(args.config),
              "fingerprint": evidence_fingerprint(args), "torch_version": torch.__version__}
    try:
        amp_enabled,amp_dtype,scale_enabled=precision_policy(config,args.device)
        report["precision"]={"autocast_enabled":amp_enabled,"dtype":str(amp_dtype),"grad_scaler_enabled":scale_enabled}
        ds = loader(args.data, "train", scale, 2, 2027, config).dataset
        for split in ("train", "val"):
            fill = np.load(Path(args.data)/f"{split}_fill_mask.npy")
            report[f"{split}_missing_fraction"] = float(fill.mean())
            report[f"{split}_fft_excluded_station_sequences"] = int(np.any(fill,axis=1).sum())
        checks["masked_time_loss_complete_station_fft"] = "PASS"
        # Same observed absolute hours must not appear as training and validation targets.
        def hours(split):
            frame = pd.read_csv(Path(args.data)/f"{split}_issue_dates.csv")
            return {pd.Timestamp(s)+pd.Timedelta(hours=i) for s in frame.target_start for i in range(168)}
        overlap = len(hours("train") & hours("val"))
        report["train_val_shared_target_hours"] = overlap
        if overlap:
            raise ValueError("train/val future target overlap; split must be audited before training")
        checks["train_val_target_disjoint"] = "PASS"
        batch = move(next(iter(loader(args.data, "train", scale, 2, 2027, config))), args.device)
        model.eval()
        noise = torch.randn(2, 168, 24, device=args.device)
        t = torch.tensor([50, 250], device=args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
        scaler = torch.amp.GradScaler(args.device, enabled=scale_enabled)
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        before_buffers={k:p.detach().clone() for k,p in model.named_buffers()}
        start = time.perf_counter()
        losses = []
        gradient_seen = set()
        for step in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(args.device, enabled=amp_enabled, dtype=amp_dtype):
                loss, _ = model(batch, t, noise)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        report["nonfinite_gradient"]={"parameter":name,"step":step,"loss":float(loss.detach()),"scale":scaler.get_scale()}
                        raise ValueError(f"nonfinite gradient: {name}")
                    if p.grad.abs().max() > 0:
                        gradient_seen.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        missing_grad = sorted(set(before)-gradient_seen)
        unchanged = [k for k,p in model.named_parameters() if torch.equal(before[k], p)]
        report.update(losses=losses, no_gradient=missing_grad, unchanged=unchanged,
                      probe_seconds=time.perf_counter()-start,
                      parameter_count=sum(p.numel() for p in model.parameters()))
        if missing_grad or unchanged:
            raise ValueError("parameters failed gradient/update audit; see report")
        if losses[-1] >= losses[0]:
            raise ValueError("fixed batch loss did not improve")
        checks["all_parameters_gradient_and_update"] = "PASS"
        checks["fixed_batch_learning"] = "PASS"
        if any(not torch.equal(before_buffers[k],p) for k,p in model.named_buffers()):
            raise ValueError("fixed prior/diffusion buffers changed during optimization")
        checks["fixed_buffers_unchanged"]="PASS"
        if isinstance(model,StationDiffusionTSInherited):
            clean=batch["actual"].transpose(1,2)*2-1
            valid=batch["valid_mask"].transpose(1,2)
            aa=model.alpha_bar[t,None,None]
            prediction=model.predict(aa.sqrt()*clean+(1-aa).sqrt()*noise,t,batch)
            structural=model.structure_losses(prediction,clean,valid,batch["forecast"].transpose(1,2)*2-1)
            report["structural_loss_gradient_audit"]={}
            for key,value in structural.items():
                gradient=torch.autograd.grad(value.mean(),prediction,retain_graph=True)[0]
                norm=float(gradient.norm())
                report["structural_loss_gradient_audit"][key]={"loss":float(value.detach().mean()),"x0_gradient_norm":norm,
                    "weighted_x0_gradient_norm":norm*config["model"]["structure_weights"][key]}
                if not np.isfinite(norm) or norm<=0:raise ValueError(f"structural loss has no finite gradient: {key}")
            checks["each_structure_loss_gradient"]="PASS"
        original = model.sample(batch, noise, steps=4)
        changed = dict(batch)
        changed["actual"] = torch.randn_like(batch["actual"])
        changed["residual_target"] = torch.randn_like(batch["residual_target"])
        if not torch.equal(original, model.sample(changed, noise, steps=4)):
            raise ValueError("future target leaked into sampling")
        checks["future_actual_residual_independence"] = "PASS"
        altered_forecast = dict(batch, forecast=torch.zeros_like(batch["forecast"]))
        if torch.allclose(original, model.sample(altered_forecast, noise, steps=4)):
            raise ValueError("forecast condition has no effect")
        checks["forecast_condition_effect"] = "PASS"
        changed_history = dict(batch, recent_error=torch.ones_like(batch["recent_error"]),
                               recent_error_mask=torch.ones_like(batch["recent_error_mask"]))
        if torch.allclose(original, model.sample(changed_history, noise, steps=4)):
            raise ValueError("recent history has no effect")
        checks["recent_history_effect"] = "PASS"
        torch.save(model.state_dict(), out/"smoke_state.pt")
        restored = build_model(config,args.data).to(args.device).eval()
        restored.load_state_dict(torch.load(out/"smoke_state.pt", weights_only=True, map_location=args.device))
        if not torch.equal(original, restored.sample(batch, noise, steps=4)):
            raise ValueError("save reload differs")
        checks["save_reload"] = "PASS"
        if isinstance(model,StationDiffusionTSInherited):
            report["inherited_condition_effect_rms"]={}
            for key in ("calendar","lead","node_state"):
                changed=dict(batch);changed[key]=torch.zeros_like(batch[key])
                delta=float((model.sample(changed,noise,steps=4)-original).square().mean().sqrt())
                report["inherited_condition_effect_rms"][key]=delta
                if not np.isfinite(delta) or delta<=0:raise ValueError(f"condition unused: {key}")
            graph_variant=copy.deepcopy(model)
            graph_variant.geographic.copy_(torch.eye(24,device=args.device))
            graph_variant.historical.copy_(torch.eye(24,device=args.device))
            delta=float((graph_variant.sample(batch,noise,steps=4)-original).square().mean().sqrt())
            report["graph_removal_rms"]=delta
            if not np.isfinite(delta) or delta<=0:raise ValueError("graphs unused")
            checks["inherited_conditions_and_graph_effect"]="PASS"
        parts = model.components(noise, t, batch)
        report["component_rms"] = [float(p.detach().square().mean().sqrt()) for p in parts]
        report["component_correlation"] = np.corrcoef([p.detach().cpu().flatten().numpy() for p in parts]).tolist()
        report["component_sum_energy_ratio"]=float(sum(parts).detach().square().mean()/sum(p.detach().square().mean() for p in parts).clamp(min=1e-12))
        report["sample_component_removal_rms"] = {}
        for i, name in enumerate(("trend", "seasonal", "residual")):
            weights = [1., 1., 1.]; weights[i] = 0.
            removed = model.sample(batch, noise, steps=4, component_weights=weights)
            delta = float((removed-original).square().mean().sqrt())
            report["sample_component_removal_rms"][name] = delta
            if not np.isfinite(delta) or delta <= 0:
                raise ValueError(f"component does not affect sampling: {name}")
        checks["paired_short_sampler_components"] = "PASS"
        # Measure short/long structures separately without calling trend==slow or season==fast.
        clean = batch["actual"].transpose(1,2)
        report["sample_structure_errors"] = {}
        for lag in (1,3,6):
            report["sample_structure_errors"][f"ramp_{lag}h_mae"] = float(
                ((original[:,lag:]-original[:,:-lag])-(clean[:,lag:]-clean[:,:-lag])).abs().mean())
        for width in (12,24):
            report["sample_structure_errors"][f"mean_{width}h_mae"] = float(
                (original.unfold(1,width,1).mean(-1)-clean.unfold(1,width,1).mean(-1)).abs().mean())
        if args.device == "cuda":
            # Probe the actual paid batch/chunk, not just the two-case audit batch.
            configured = move(next(iter(loader(args.data, "train", scale, config["train"]["batch_size"], 2027, config))), args.device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                configured_loss, _ = model(configured)
            scaler.scale(configured_loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update(); torch.cuda.synchronize()
            report["configured_training_probe"] = {"batch_size": len(configured["actual"]),
                "seconds": time.perf_counter()-started,
                "peak_allocated_gb": torch.cuda.max_memory_allocated()/1024**3}
            model.eval()
            chunk = config["generation"]["chunk"]
            chunk_batch = {name: batch[name][:1].repeat(chunk,*([1]*(batch[name].ndim-1))) for name in condition_keys(model)}
            chunk_noise = torch.randn(chunk,168,24,device=args.device)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            probe = model.sample(chunk_batch,chunk_noise,steps=8)
            torch.cuda.synchronize()
            if not torch.isfinite(probe).all(): raise ValueError("configured chunk nonfinite")
            report["cuda_probe"]={"seconds_8_steps":time.perf_counter()-started,"member_chunk":chunk,
                "peak_allocated_gb":torch.cuda.max_memory_allocated()/1024**3}
            checks["configured_training_batch_and_generation_chunk"] = "PASS"
        else:
            checks["configured_training_batch_and_generation_chunk"] = "NOT RUN"
        checks["cuda_amp"] = "PASS" if args.device == "cuda" else "NOT RUN"
        checks["formal_500_member_component_ablation"] = "NOT RUN"
        report["launch_eligible"] = args.device == "cuda"
    except Exception as error:
        report["error"] = str(error)
        write_json(out/"preflight.json", report)
        raise
    write_json(out/"preflight.json", report)
    print(json.dumps(report), flush=True)


def train(args, config, model, scale):
    gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    eligible = gate.get("launch_eligible") or (args.local_smoke and args.device == "cpu" and "error" not in gate)
    if not eligible or gate.get("fingerprint") != evidence_fingerprint(args):
        raise ValueError("matching target CUDA preflight required")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    write_json(out/"launch_manifest.json", gate)
    write_json(out/"residual_scale.json", scale)
    (out/"config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(out/"jstd_event_targets.json", {"thresholds": fit_station_jstd_event_thresholds(args.data, {})})
    tc = config["train"]
    tr = loader(args.data, "train", scale, tc["batch_size"], tc["seed"], config)
    va = loader(args.data, "val", scale, tc["batch_size"], tc["seed"], config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    amp_enabled,amp_dtype,scale_enabled=precision_policy(config,args.device)
    scaler = torch.amp.GradScaler(args.device, enabled=scale_enabled)
    history, best, best_epoch = [], float("inf"), 0
    for epoch in range(1, tc["epochs"]+1):
        started = time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
        sums = np.zeros(3); size = 0; extra_sums={}
        for j, batch in enumerate(tr):
            batch = move(batch, args.device)
            group_start = (j//tc["accumulation"])*tc["accumulation"]
            group_samples = min(tc["accumulation"]*tc["batch_size"], len(tr.dataset)-group_start*tc["batch_size"])
            with torch.amp.autocast(args.device, enabled=amp_enabled, dtype=amp_dtype):
                loss, parts = model(batch)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            n = len(batch["actual"])
            scaler.scale(loss*n/group_samples).backward()
            sums += n*np.array([float(loss.detach()), float(parts["reconstruction"]), float(parts["fourier"])])
            size += n
            for key,value in parts.items():extra_sums[key]=extra_sums.get(key,0.)+n*float(value)
            if (j+1)%tc["accumulation"] == 0 or j+1 == len(tr):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_loss": float(sums[0]/size),
               "train_reconstruction": float(sums[1]/size), "train_fourier": float(sums[2]/size),
               "seconds": time.perf_counter()-started}
        row.update({f"train_{key}":value/size for key,value in extra_sums.items()})
        if epoch == 1 or epoch%tc["validation_every"] == 0:
            model.eval(); total = np.zeros(3); seen = 0; val_parts={}
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if args.device == "cuda" else []):
                torch.manual_seed(tc["validation_seed"])
                with torch.no_grad():
                    for batch in va:
                        batch = move(batch, args.device)
                        loss, parts = model(batch)
                        n = len(batch["actual"]); seen += n
                        total += n*np.array([float(loss), float(parts["reconstruction"]), float(parts["fourier"])])
                        for key,value in parts.items():val_parts[key]=val_parts.get(key,0.)+n*float(value)
            row.update(val_loss=float(total[0]/seen), val_reconstruction=float(total[1]/seen), val_fourier=float(total[2]/seen))
            row.update({f"val_{key}":value/seen for key,value in val_parts.items()})
            if row["val_loss"] < best-1e-5:
                best, best_epoch = row["val_loss"], epoch
                torch.save({"model_state_dict": model.state_dict(), "config": config,
                    "epoch": epoch, "validation_objective": best, "version": model.VERSION}, out/"model_best.pt")
        history.append(row); write_json(out/"training_history.json", history)
        print(json.dumps(row), flush=True)
        if epoch-best_epoch >= tc["patience_epochs"]:
            break
    write_json(out/"training_summary.json", {"completed": True, "best_epoch": best_epoch,
        "best_validation_objective": best, "stopped_epoch": epoch,
        "local_smoke_only": args.local_smoke,
        "target": "joint_actual_power", "frozen_parameters": [], "raw_baseline_unchanged": True})


def generate(args, config, model, scale):
    run = Path(args.run)
    saved = torch.load(run/"model_best.pt", map_location=args.device, weights_only=False)
    if saved["version"] != model.VERSION or saved["config"] != config:
        raise ValueError("checkpoint/config mismatch")
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    out=Path(args.output); out.mkdir(parents=True, exist_ok=False)
    gc=config["generation"]; k=int(gc["members"]); chunk=int(gc["chunk"])
    val=loader(args.data, "val", scale, 1, 2027, config)
    raw=np.lib.format.open_memmap(out/"actual_scenarios_raw_normalized.npy", mode="w+", dtype="float32", shape=(len(val.dataset), k, 168, 24))
    # Pre-draw each issue's initial noise so changing chunk leaves noise unchanged.
    generator=torch.Generator(device="cpu").manual_seed(gc["seed"])
    begin=time.perf_counter()
    for i,batch in enumerate(val):
        noise=torch.randn(k,168,24,generator=generator)
        for start in range(0,k,chunk):
            n=min(chunk,k-start)
            condition={name:batch[name].repeat(n,*([1]*(batch[name].ndim-1))).to(args.device) for name in condition_keys(model)}
            raw[i,start:start+n]=model.sample(condition,noise[start:start+n].to(args.device),gc["steps"]).cpu().numpy()
        raw.flush(); print(f"generated issue {i+1}/{len(val)}",flush=True)
    daylight,day_audit=build_station_daylight_mask(args.data,"val")
    projected=np.lib.format.open_memmap(out/"actual_scenarios_normalized.npy",mode="w+",dtype="float32",shape=raw.shape)
    for i in range(len(raw)):
        projected[i]=np.where(daylight[i][None],np.clip(raw[i],0,1),0)
    projected.flush()
    actual=np.asarray(val.dataset.actual); forecast=np.asarray(val.dataset.forecast)
    np.save(out/"actual_data_normalized.npy",actual); np.save(out/"forecast_data_normalized.npy",forecast)
    np.save(out/"station_daylight_mask.npy",daylight)
    # Compatibility with event evaluator only: no tail routing in this model.
    np.save(out/"tail_expert_route.npy",np.zeros((len(raw),k),dtype=bool))
    stations=pd.read_csv(Path(args.data)/"station_order.csv").sort_values("channel_index").reset_index(drop=True)
    static=load_station_static_data(args.data)
    metrics,station_frame,lead_frame=evaluate_station_scenarios(projected,raw,actual,forecast,stations,
        static["station_adjacency"].numpy(),daylight_mask=daylight,energy_score_member_limit=gc["energy_members"])
    metadata={"condition_variant":config["experiment"],"architecture":model.VERSION,"checkpoint_epoch":saved["epoch"],
        "checkpoint_state_source":"raw","checkpoint_validation_objective":saved["validation_objective"],
        "n_samples":k,"evaluation_member_count":k,"split":"val","generation_seed":gc["seed"],
        "generation_seconds":time.perf_counter()-begin,"sampling_steps":gc["steps"],"sampler":"DDIM_eta0",
        "tail_route_semantics":"compatibility_zeros_no_expert_groups","use_body_tail_experts":False,
        "future_actual_used_as_generation_condition":False,"daylight_audit":day_audit,
        "physical_projection":"clip_0_1_and_station_astronomical_solar_night",
        "local_smoke_only":args.local_smoke,"condition_fields":list(condition_keys(model)),
        "parameter_count":sum(p.numel() for p in model.parameters()),"joint_station_count":24}
    metrics["run"]=metadata
    save_evaluation(out,metrics,station_frame,lead_frame);write_json(out/"generation_metadata.json",metadata)
    print(f"GENERATION_COMPLETE result={out}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode", choices=["preflight","train","generate"])
    p.add_argument("--config",default="configs/station24_diffusion_ts_joint_v1.yaml")
    p.add_argument("--data",default="diffusion_input_station")
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--gate");p.add_argument("--run")
    p.add_argument("--local-smoke",action="store_true",help="CPU-only two-epoch/3-member/4-step integration test; never formal evidence")
    args=p.parse_args()
    torch.set_num_threads(4)
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.local_smoke:
        if args.device != "cpu":raise ValueError("local smoke must use CPU")
        config["train"]["epochs"]=2
        config["generation"].update(members=3,steps=4,chunk=3)
        config["local_smoke_only"]=True
    torch.manual_seed(config["train"]["seed"]);np.random.seed(config["train"]["seed"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required on server")
    model=build_model(config,args.data).to(args.device)
    scale=fit_station_residual_scale(args.data)
    {"preflight":preflight,"train":train,"generate":generate}[args.mode](args,config,model,scale)


if __name__ == "__main__":
    main()
