"""Strict-mask evaluation for Shandong91 V3, including aggregate covariance diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_shandong91_v2 import (
    crps, energy_score, paired_comparison_metrics, spatial_correlation_rmse,
    trajectory_acf,
)

CHANNELS = ("Wind", "Solar", "Load")


def covariance_amplification(samples, mask):
    masked = np.where(mask[:, None], samples, 0.0)
    node_variance = np.var(masked, axis=1, ddof=0)
    system_variance = np.var(masked.sum(axis=3), axis=1, ddof=0)
    denominator = node_variance.sum(axis=2)
    valid = denominator > 1e-12
    ratio = system_variance[valid] / denominator[valid]
    return {
        "definition": "member-conditional Var(sum active nodes) / sum Var(active node), per window/time",
        "count": int(ratio.size), "mean": float(ratio.mean()),
        "median": float(np.median(ratio)), "q05": float(np.quantile(ratio,.05)),
        "q95": float(np.quantile(ratio,.95)),
    }


def system_resource_correlations(system_scenarios, system_truth, active):
    truth_rows = np.stack([
        np.where(active[..., c], system_truth[..., c], np.nan) for c in range(3)
    ], axis=-1).reshape(-1,3)
    truth_rows = truth_rows[np.isfinite(truth_rows).all(axis=1)]
    conditional = []
    for issue in range(system_scenarios.shape[0]):
        for hour in range(system_scenarios.shape[2]):
            if active[issue,hour].all():
                conditional.append(np.corrcoef(system_scenarios[issue,:,hour,:], rowvar=False))
    return {
        "truth_cross_time_overlapping_windows": np.corrcoef(truth_rows, rowvar=False).tolist(),
        "generated_member_conditional_mean": np.nanmean(np.stack(conditional),axis=0).tolist(),
        "warning": "truth cross-time and generated member-conditional correlations are distinct estimands",
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--result",required=True); parser.add_argument("--output-dir",required=True)
    parser.add_argument("--baseline-result")
    args=parser.parse_args()
    source=Path(args.result); output=Path(args.output_dir)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    scenarios=np.load(source/"actual_scenarios_mw.npy",mmap_mode="r")
    truth=np.load(source/"actual_mw.npy",mmap_mode="r")
    forecast=np.load(source/"forecast_mw.npy",mmap_mode="r")
    mask=np.load(source/"effective_mask.npy",mmap_mode="r").astype(bool)
    if scenarios.shape[0]!=truth.shape[0] or scenarios.shape[2:]!=truth.shape[1:]: raise ValueError("shape mismatch")
    metrics={"evaluation_contract":"strict elementwise effective_mask","shape":list(scenarios.shape),
             "acf_contract":"truth trajectory ACF compared with individual generated-member trajectory ACF",
             "finite_ratio":float(np.isfinite(scenarios).mean()),"channels":{},"system_aggregate":{},"joint":{},"physical":{}}
    systems=[]; truths=[]; active_resources=[]
    for channel,name in enumerate(CHANNELS):
        samples=np.asarray(scenarios[...,channel]); target=np.asarray(truth[...,channel]); current=np.asarray(mask[...,channel])
        prediction=np.asarray(forecast[...,channel]); lower=np.quantile(samples,.05,axis=1); upper=np.quantile(samples,.95,axis=1)
        interval=upper-lower+20*(np.maximum(lower-target,0)+np.maximum(target-upper,0))
        residual_truth=target-prediction; residual_samples=samples-prediction[:,None]
        node_coverage=[]
        for node in range(target.shape[2]):
            node_mask=current[...,node]
            if node_mask.any():
                node_coverage.append(float(((target[...,node]>=lower[...,node]) &
                                            (target[...,node]<=upper[...,node]))[node_mask].mean()))
        valid_samples=residual_samples[np.broadcast_to(current[:,None],residual_samples.shape)]
        valid_truth=residual_truth[current]
        metrics["channels"][name]={
            "effective_elements":int(current.sum()),
            "crps_mw":float(crps(samples,target)[current].mean()),
            "coverage90":float(((target>=lower)&(target<=upper))[current].mean()),
            "width90_mw":float((upper-lower)[current].mean()),
            "interval_score90_mw":float(interval[current].mean()),
            "ensemble_conditional_std_mw":float(np.std(samples,axis=1)[current].mean()),
            "residual_quantiles_mw":{
                key:{"truth":float(np.quantile(valid_truth,q)),"ensemble":float(np.quantile(valid_samples,q))}
                for key,q in (("q01",.01),("q05",.05),("q50",.50),("q95",.95),("q99",.99))
            },
            "node_coverage90":{"count":len(node_coverage),"min":float(np.min(node_coverage)),
                               "median":float(np.median(node_coverage)),"max":float(np.max(node_coverage)),
                               "values":node_coverage},
            "temporal_acf":trajectory_acf(target,samples,current),
            "spatial_residual_correlation":spatial_correlation_rmse(target-prediction,samples-prediction[:,None],current),
        }
        ramp={}
        for lag in (1,3,6):
            ramp_mask=current[:,lag:] & current[:,:-lag]
            ramp[str(lag)]={"crps_mw":float(crps(samples[:,:,lag:]-samples[:,:,:-lag],
                                                  target[:,lag:]-target[:,:-lag])[ramp_mask].mean())}
        metrics["channels"][name]["ramp"]=ramp
        system=np.sum(np.where(current[:,None],samples,0),axis=3); system_truth=np.sum(np.where(current,target,0),axis=2)
        active=current.any(axis=2); lo=np.quantile(system,.05,axis=1); hi=np.quantile(system,.95,axis=1)
        sys_interval=hi-lo+20*(np.maximum(lo-system_truth,0)+np.maximum(system_truth-hi,0))
        metrics["system_aggregate"][name]={
            "coverage90":float(((system_truth>=lo)&(system_truth<=hi))[active].mean()),
            "crps_mw":float(crps(system,system_truth)[active].mean()),
            "width90_mw":float((hi-lo)[active].mean()),
            "interval_score90_mw":float(sys_interval[active].mean()),
            "covariance_amplification":covariance_amplification(samples,current),
        }
        systems.append(system); truths.append(system_truth); active_resources.append(active)
        expanded_mask=np.broadcast_to(current[:,None],samples.shape)
        metrics["physical"][name]={
            "generated_negative_rate":float((samples[expanded_mask]<0).mean()),
            "truth_negative_rate":float((target[current]<0).mean()),
            "note":"read-only diagnostic; no clipping or label changes",
        }
    joint_s=np.stack(systems,axis=-1); joint_y=np.stack(truths,axis=-1); active=np.stack(active_resources,axis=-1)
    score,used=energy_score(joint_s,joint_y)
    metrics["joint"]={"system_resource_energy_score_mw":score,"members_used":used,
                      "system_residual_correlations":system_resource_correlations(
                          joint_s-np.stack([np.sum(np.where(mask[...,c],forecast[...,c],0),axis=2) for c in range(3)],axis=-1)[:,None],
                          joint_y-np.stack([np.sum(np.where(mask[...,c],forecast[...,c],0),axis=2) for c in range(3)],axis=-1),active)}
    if args.baseline_result:
        baseline_path=Path(args.baseline_result)
        baseline=np.load(baseline_path/"actual_scenarios_mw.npy",mmap_mode="r")
        if baseline.shape[0]!=truth.shape[0] or baseline.shape[2:]!=truth.shape[1:]:
            raise ValueError("baseline is not paired to the same validation layout")
        baseline_metrics=paired_comparison_metrics(baseline,truth,mask)
        metrics["baseline_comparison"]={
            "baseline_result":str(baseline_path),
            "contract":"paired same validation truth and candidate effective_mask",
            "baseline":baseline_metrics,
            "candidate_minus_baseline":{
                name:{key:metrics["channels"][name][key]-baseline_metrics["channels"][name][key]
                      for key in ("crps_mw","coverage90","width90_mw","interval_score90_mw","ensemble_conditional_std_mw")}
                for name in CHANNELS
            },
        }
    metadata_path=source/"metadata.json"
    metadata=json.loads(metadata_path.read_text("utf-8")) if metadata_path.exists() else {}
    metrics["model_identifier"]=metadata.get("model_identifier","unknown")
    output.mkdir(parents=True); (output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=True),"utf-8")
    lines=["# Shandong91 low-rank Raw Body evaluation","",
           f"- Model: `{metrics['model_identifier']}`",
           f"- Shape: `{metrics['shape']}`",f"- Finite ratio: `{metrics['finite_ratio']}`",
           "- Mask: strict elementwise `effective_mask`",
           "- Covariance amplification: conditional across members; truth cross-time statistics are not conflated.","",
           "| Resource | CRPS | Coverage90 | Width90 | Interval score | Cond. std | System CRPS | System coverage | Cov amp |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in CHANNELS:
        c=metrics["channels"][name]; s=metrics["system_aggregate"][name]
        lines.append(f"| {name} | {c['crps_mw']:.6f} | {c['coverage90']:.4f} | {c['width90_mw']:.6f} | {c['interval_score90_mw']:.6f} | {c['ensemble_conditional_std_mw']:.6f} | {s['crps_mw']:.6f} | {s['coverage90']:.4f} | {s['covariance_amplification']['mean']:.4f} |")
    lines += ["",f"Joint system-resource Energy Score: `{score:.6f}` MW.",
              "", "Physical negative rates are read-only diagnostics; no clipping is applied."]
    (output/"RESULT_SUMMARY.md").write_text("\n".join(lines),"utf-8")


if __name__=="__main__": main()
