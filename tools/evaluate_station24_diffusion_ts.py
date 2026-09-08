"""Joint wind/solar/total and lead-day comparison for the Diffusion-TS experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def crps(samples, truth):
    ordered=np.sort(samples,axis=1)
    k=ordered.shape[1]
    coefficient=(2*np.arange(1,k+1)-k-1)[None,:,None]
    return np.abs(samples-truth[:,None]).mean(1)-(ordered*coefficient).sum(1)/(k*k)


def correlation(x,y):
    x=x-x.mean(-1,keepdims=True);y=y-y.mean(-1,keepdims=True)
    denominator=np.sqrt((x*x).sum(-1)*(y*y).sum(-1))
    return np.divide((x*y).sum(-1),denominator,out=np.full_like(denominator,np.nan),where=denominator>1e-10)


def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline",required=True);p.add_argument("--candidate",required=True)
    p.add_argument("--data",default="diffusion_input_station");p.add_argument("--output",required=True)
    p.add_argument("--candidate-label",default="Diffusion-TS joint V1")
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    stations=pd.read_csv(Path(args.data)/"station_order.csv").sort_values("channel_index").reset_index(drop=True)
    weights=stations.capacity_mw.to_numpy()
    daily=[];joint=[];reference=None;ordinary=[];multiscale=[]
    for label,folder in [("Raw body-tail",args.baseline),(args.candidate_label,args.candidate)]:
        folder=Path(folder)
        actual=np.load(folder/"actual_data_normalized.npy");forecast=np.load(folder/"forecast_data_normalized.npy")
        if reference is None:reference=(actual,forecast)
        elif not all(np.array_equal(a,b) for a,b in zip(reference,(actual,forecast))):
            raise ValueError("different validation targets/forecast/order")
        scenarios=np.load(folder/"actual_scenarios_normalized.npy",mmap_mode="r")
        if scenarios.shape != (23,500,168,24):raise ValueError("formal 23x500x168x24 required")
        aggregated={}
        for kind in ("wind","solar","renewable"):
            index=np.arange(24) if kind=="renewable" else stations.index[stations.data_type.eq(kind)].to_numpy()
            sample=np.einsum("nkts,s->nkt",scenarios[...,index],weights[index])
            truth=np.einsum("nts,s->nt",actual[...,index],weights[index])
            predicted=np.einsum("nts,s->nt",forecast[...,index],weights[index])
            aggregated[kind]=(sample,truth,predicted)
            for scale in (1,3,6,12,24):
                if scale<=6:
                    sx=sample[:,:,scale:]-sample[:,:,:-scale];sy=truth[:,scale:]-truth[:,:-scale]
                    feature="ramp"
                else:
                    sx=np.lib.stride_tricks.sliding_window_view(sample,scale,axis=-1).mean(-1)
                    sy=np.lib.stride_tricks.sliding_window_view(truth,scale,axis=-1).mean(-1)
                    feature="moving_mean"
                bounds=np.quantile(sx,[.05,.95],axis=1)
                multiscale.append({"variant":label,"source":kind,"feature":feature,"hours":scale,
                    "crps_mw":float(crps(sx,sy).mean()),
                    "coverage90":float(((sy>=bounds[0])&(sy<=bounds[1])).mean()),
                    "width90_mw":float((bounds[1]-bounds[0]).mean())})
            score=crps(sample,truth);q=np.quantile(sample,[.05,.95],axis=1)
            for day in range(7):
                sl=slice(day*24,(day+1)*24)
                daily.append({"variant":label,"source":kind,"lead_day":day+1,
                    "crps_mw":score[:,sl].mean(),"coverage90":((truth[:,sl]>=q[0,:,sl])&(truth[:,sl]<=q[1,:,sl])).mean(),
                    "width90_mw":(q[1,:,sl]-q[0,:,sl]).mean(),"forecast_mae_mw":np.abs(predicted[:,sl]-truth[:,sl]).mean()})
        ws,wa,wf=aggregated["wind"];ss,sa,sf=aggregated["solar"]
        for residual in (False,True):
            member_corr=correlation(ws-wf[:,None] if residual else ws,ss-sf[:,None] if residual else ss)
            truth_corr=correlation(wa-wf if residual else wa,sa-sf if residual else sa)
            estimated=np.nanmean(member_corr,axis=1)
            joint.append({"variant":label,"series":"residual" if residual else "power",
                "wind_solar_correlation_rmse":float(np.sqrt(np.nanmean((estimated-truth_corr)**2))),
                "valid_issue_count":int(np.isfinite(truth_corr).sum())})
        metrics=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        ordinary.append({"variant":label,"wind_crps":metrics["station_average"]["wind"]["crps"],
            "solar_crps":metrics["station_average"]["solar"]["crps"],
            "renewable_crps_mw":metrics["aggregate_mw"]["renewable"]["crps"],
            "energy_score":metrics["joint"]["energy_score_pu"],"spatial_corr_rmse":metrics["joint"]["spatial_corr_rmse_all_pairs"]})
    pd.DataFrame(daily).to_csv(out/"lead_day_wind_solar_total.csv",index=False)
    pd.DataFrame(joint).to_csv(out/"same_member_wind_solar_correlation.csv",index=False)
    pd.DataFrame(ordinary).to_csv(out/"ordinary_comparison.csv",index=False)
    pd.DataFrame(multiscale).to_csv(out/"wind_solar_total_multiscale.csv",index=False)
    print(f"DIFFUSION_TS_JOINT_EVALUATION_COMPLETE output={out}",flush=True)


if __name__=="__main__":main()
