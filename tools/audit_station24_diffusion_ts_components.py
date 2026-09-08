"""Read-only best-checkpoint decomposition audit; no optimizer or training."""
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from tools.station24_diffusion_ts_experiment import build_model,loader,move,write_json,condition_keys,digest
from station_dataset import fit_station_residual_scale


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True)
    p.add_argument("--data",default="diffusion_input_station");p.add_argument("--device",default="cuda",choices=["cuda","cpu"])
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(8317)
    run=Path(args.run);config=yaml.safe_load((run/"config_used.yaml").read_text(encoding="utf-8"))
    checkpoint=run/"model_best.pt";before=digest(checkpoint)
    saved=torch.load(checkpoint,map_location=args.device,weights_only=False)
    model=build_model(config,args.data).to(args.device).eval()
    model.load_state_dict(saved["model_state_dict"])
    dl=loader(args.data,"val",fit_station_residual_scale(args.data),1,2027,config)
    records=[]
    for index,batch in enumerate(dl):
        if index==3:break
        batch=move(batch,args.device);condition={k:batch[k] for k in condition_keys(model)}
        clean=batch["actual"].transpose(1,2)*2-1
        noise=torch.randn_like(clean);t=torch.tensor([250],device=args.device);a=model.alpha_bar[t,None,None]
        parts=model.components(a.sqrt()*clean+(1-a).sqrt()*noise,t,condition)
        values=[x.detach() for x in parts]
        names=("trend","seasonal","residual")
        row={"issue_index":index,"teacher_forced_timestep":250,"components":{}}
        for name,x in zip(names,values):
            row["components"][name]={"rms":float(x.square().mean().sqrt()),
                **{f"mean_{w}h_energy_fraction":float(x.unfold(1,w,1).mean(-1).square().mean()/x.square().mean().clamp(min=1e-12)) for w in (12,24)}}
        row["pair_correlation"]=np.corrcoef([x.cpu().flatten().numpy() for x in values]).tolist()
        row["sum_energy_over_component_energies"]=float(sum(values).square().mean()/sum(x.square().mean() for x in values).clamp(min=1e-12))
        full=model.sample(condition,noise,8)
        row["paired_8_step_component_removal_rms"]={}
        for i,name in enumerate(names):
            weights=[1.,1.,1.];weights[i]=0.
            removed=model.sample(condition,noise,8,weights)
            delta=float((removed-full).square().mean().sqrt())
            if not np.isfinite(delta):raise ValueError("nonfinite component sampling audit")
            row["paired_8_step_component_removal_rms"][name]=delta
        row["full_sample_boundary_fraction"]=float(((full<=1e-6)|(full>=1-1e-6)).float().mean())
        records.append(row)
    if digest(checkpoint)!=before:raise ValueError("checkpoint was changed during audit")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/"component_audit.json",{"checkpoint_sha256":before,"checkpoint_epoch":saved["epoch"],
        "no_training":True,"scope":"first_three_validation_windows_mechanism_only",
        "teacher_forced_uses_actual":True,"sampling_uses_actual":False,
        "formal_500_member_ablation":"NOT RUN","orthogonal_decomposition_claim":False,
        "local_smoke_only":config.get("local_smoke_only",False),"records":records})
    print(f"TS_COMPONENT_AUDIT_COMPLETE output={out}",flush=True)


if __name__=="__main__":main()
