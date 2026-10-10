"""Bounded V3 engineering gate; formal CUDA/AMP mode is mandatory on server."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from datasets.shandong91_faithful24 import Shandong91Faithful24Dataset, fit_faithful24_state_thresholds
from datasets.shandong91_low_rank import FixedPCAFactorTransform, fit_train_only_pca
from src.models.shandong91_low_rank_diffusion import Shandong91HeterogeneousRawBodyV3, Shandong91LowRankDiffusion
from train_shandong91_v2 import move_batch

CHANNELS=("Wind","Solar","Load")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/shandong91/raw_body_v3_low_rank_common_factor.yaml")
    parser.add_argument("--output",required=True); parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--amp",action="store_true"); parser.add_argument("--full-model",action="store_true")
    args=parser.parse_args(); output=Path(args.output)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if args.amp and args.device!="cuda": raise ValueError("AMP requires CUDA")
    output.mkdir(parents=True); device=torch.device("cuda:0" if args.device=="cuda" else "cpu")
    config=yaml.safe_load(Path(args.config).read_text("utf-8")); torch.manual_seed(20271010)
    factors=fit_train_only_pca(config["data"]["data_path"],config["model"]["common_factor_counts"])
    thresholds=fit_faithful24_state_thresholds(config["data"]["data_path"])
    dataset=Shandong91Faithful24Dataset(config["data"]["data_path"],"train",thresholds)
    batch=move_batch({k:v.unsqueeze(0) for k,v in dataset[0].items()},device); mask=batch["effective_mask"].bool()
    small=copy.deepcopy(config["model"])
    if not args.full_model: small.update(base_channels=8,channel_multipliers=[1,2,2],group_norm_groups=4,state_channels=[4,8,8],dropout=0.0)
    denoiser=Shandong91HeterogeneousRawBodyV3(small,dataset.node_features.to(device),dataset.adjacency_with_self.float().to(device))
    model=Shandong91LowRankDiffusion(denoiser,config["diffusion"],factors).to(device)
    checks={}
    def record(name,passed,detail): checks[name]={"status":"PASS" if passed else "FAIL","detail":detail}
    fit_detail={n:{k:factors["resources"][n][k] for k in ("factor_count","complete_train_rows","train_start","train_end","explained_variance_ratio_cumulative","train_reconstruction_rmse_mw")} for n in CHANNELS}
    record("train_only_pca_fit",all(v["train_end"]=="2025-10-31T23:00:00+08:00" for v in fit_detail.values()),fit_detail)
    record("load_rank1_data_fact",fit_detail["Load"]["explained_variance_ratio_cumulative"]>.999999,{"fact":"77 active Load nodes are strict rank-1 to numerical tolerance","inference":"fixed-proportion allocation is highly credible; upstream source unconfirmed","independent_measurements_claim":False,**fit_detail["Load"]})
    root=Path(config["data"]["data_path"]); hourly=np.load(root/"hourly/residual_mw.npy",mmap_mode="r")
    actual=np.load(root/"hourly/actual_mw.npy",mmap_mode="r"); forecast=np.load(root/"hourly/forecast_mw.npy",mmap_mode="r")
    solar_duplicate=max(float(np.max(np.abs(array[:,79,1]-array[:,82,1]))) for array in (hourly,actual,forecast))
    record("solar_80_83_retained_risk",solar_duplicate==0.0,{"max_abs_delta_mw":solar_duplicate,"policy":"retain both; unresolved mapping risk"})
    factor,local=model.transform.decompose(batch["residual"],mask); restored=model.transform.reconstruct(factor,local,mask)
    reconstruction=float((restored-batch["residual"]).abs()[mask].max())
    orthogonality=float(model.transform.project_factor(local,mask).abs().max())
    record("decomposition_reconstruction",reconstruction<1e-5,{"max_abs_normalized":reconstruction})
    record("local_common_no_overlap",orthogonality<2e-5,{"projected_factor_max_abs":orthogonality})
    record("load_local_disabled",float(local[...,2].abs().max())==0.0,{"max_abs":float(local[...,2].abs().max())})
    contract=dataset.condition_manifest(); record("causal_conditions",contract["recent_error_future_actual_used"] is False and contract["lead"]=="disabled",{"recent_error":contract["recent_error"],"recent_error_future_actual_used":contract["recent_error_future_actual_used"],"lead":contract["lead"],"state_threshold_sha256":contract["state_threshold_sha256"]})
    timestep=torch.tensor([137],device=device); source_noise=torch.randn_like(batch["residual"])
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp):
        prediction,error,latents=model.prediction_and_error(batch,timestep,noise=source_noise,return_latents=True)
        loss=model.masked_loss(error,mask)
    latents["predicted_factor_noise"].retain_grad(); model.zero_grad(set_to_none=True); loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    factor_grad=latents["predicted_factor_noise"].grad
    record("forward_backward_finite",prediction.shape==(1,168,91,3) and torch.isfinite(loss) and gradients and all(torch.isfinite(g).all() for g in gradients),{"loss":float(loss.detach()),"prediction_shape":list(prediction.shape)})
    factor_slices={"Wind":(0,5),"Solar":(5,6),"Load":(6,7)}
    for name,(start,stop) in factor_slices.items():
        norm=float(factor_grad[...,start:stop].double().norm())
        record(f"{name.lower()}_factor_gradient",np.isfinite(norm) and norm>0,{"l2":norm})
    core=denoiser.encoder_blocks[0].conv1.weight; before=core.detach().clone(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    optimizer.step(); delta=float((core.detach()-before).abs().max())
    record("optimizer_core_update",np.isfinite(delta) and delta>0,{"max_abs_delta":delta})
    changed=prediction.detach().clone(); changed[~mask]+=999
    invariant=model.masked_loss((changed-model.transform.reconstruct(latents["factor_noise"],latents["local_noise"],mask,include_mean=False)).square(),mask)
    record("mask_invariance",abs(float(invariant)-float(loss.detach()))<1e-6,{"absolute_delta":abs(float(invariant)-float(loss.detach()))})
    checkpoint=output/"reload_probe.pt"; torch.save(model.state_dict(),checkpoint)
    clone=Shandong91LowRankDiffusion(Shandong91HeterogeneousRawBodyV3(small,dataset.node_features.to(device),dataset.adjacency_with_self.float().to(device)),config["diffusion"],factors).to(device)
    clone.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True); clone.eval(); model.eval()
    initial_factor=torch.randn(1,168,7,device=device); initial_local=torch.randn_like(batch["residual"])
    with torch.no_grad():
        left=model.sample(batch,mask,method="ddim",inference_steps=2,initial_factor_noise=initial_factor,initial_local_noise=initial_local)
        right=clone.sample(batch,mask,method="ddim",inference_steps=2,initial_factor_noise=initial_factor,initial_local_noise=initial_local)
    reload_delta=float((left-right).abs().max()); record("checkpoint_reload_and_sampler",reload_delta<=1e-6 and torch.isfinite(left).all(),{"max_abs_delta":reload_delta,"shape":list(left.shape)})
    load_basis=model.transform.basis[6,:,2].double(); load_mean=model.transform.mean[:,2].double()
    pivot=int(load_basis.abs().argmax()); centered=left[...,2].double()-load_mean
    coefficient=centered[...,pivot]/load_basis[pivot]
    expected=coefficient.unsqueeze(-1)*load_basis
    load_active=mask[...,2]; load_rank1_error=float((expected-centered).abs()[load_active].max())
    load_scale=float(centered.abs()[load_active].max()); load_rank1_relative=load_rank1_error/max(load_scale,1.0)
    record("sample_load_rank1",load_rank1_error<2e-4 and load_rank1_relative<1e-6,{"reconstruction_max_abs":load_rank1_error,"relative_to_max_abs":load_rank1_relative,"contract":"centered Load lies in the single fixed PCA loading span"})
    if args.full_model and device.type=="cuda":
        formal=int(config["training"]["batch_size"]); expanded={k:v.expand(formal,*v.shape[1:]).contiguous() for k,v in batch.items()}
        model.train(); model.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=args.amp):
            _,formal_error=model.prediction_and_error(expanded,torch.full((formal,),137,device=device,dtype=torch.long)); formal_loss=model.masked_loss(formal_error,expanded["effective_mask"])
        formal_loss.backward(); finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        record("formal_microbatch_cuda_amp",bool(torch.isfinite(formal_loss) and finite),{"batch_size":formal,"loss":float(formal_loss.detach()),"peak_allocated_gb":torch.cuda.max_memory_allocated(device)/2**30})
    parameter_count=sum(p.numel() for p in model.parameters()); status="PASS" if all(v["status"]=="PASS" for v in checks.values()) else "FAIL"
    report={"status":status,"scope":"bounded V3 engineering preflight; no formal training/generation","device":str(device),"amp":args.amp,"full_model":args.full_model,"parameter_count":parameter_count,"factor_document_sha256":factors["sha256"],"checks":checks,"formal_training":"NOT RUN","formal_generation":"NOT RUN","cuda_amp_safety":"PASS" if args.full_model and device.type=="cuda" and status=="PASS" else "NOT RUN"}
    (output/"report.json").write_text(json.dumps(report,indent=2),"utf-8"); (output/"report.md").write_text("\n".join(["# Shandong91 V3 preflight","",f"- Overall: **{status}**",f"- Parameters: **{parameter_count:,}**","- Formal training: **NOT RUN**","- Formal generation: **NOT RUN**",""]+[f"- {k}: **{v['status']}**" for k,v in checks.items()]),"utf-8")
    print(json.dumps(report,indent=2));
    if status!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
