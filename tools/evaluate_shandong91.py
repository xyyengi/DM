"""Ordinary, ramp and joint diagnostics for Shandong91 scenarios."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
CHANNELS=('Wind','Solar','Load')

def crps(samples,y):
    x=np.sort(samples,axis=1); m=x.shape[1]; weights=(2*np.arange(1,m+1)-m-1).reshape(1,m,*([1]*(x.ndim-2)))
    return np.mean(np.abs(x-y[:,None]),axis=1)-np.sum(weights*x,axis=1)/(m*m)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--result',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
    src=Path(a.result); out=Path(a.output_dir)
    if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
    s=np.load(src/'actual_scenarios_mw.npy',mmap_mode='r'); y=np.load(src/'actual_mw.npy',mmap_mode='r'); f=np.load(src/'forecast_mw.npy',mmap_mode='r'); mask=np.load(src/'valid_mask.npy',mmap_mode='r').astype(bool)
    metrics={'shape':list(s.shape),'finite_ratio':float(np.isfinite(s).mean()),'channels':{},'ramp':{},'joint':{}}
    for c,name in enumerate(CHANNELS):
        channel=np.asarray(s[...,c]); truth=np.asarray(y[...,c]); m=mask[...,c]
        med=np.median(channel,axis=1); lo=np.quantile(channel,.05,axis=1); hi=np.quantile(channel,.95,axis=1); score=crps(channel,truth)
        metrics['channels'][name]={'n':int(m.sum()),'crps_mw':float(score[m].mean()),'median_mae_mw':float(np.abs(med-truth)[m].mean()),'forecast_mae_mw':float(np.abs(f[...,c]-truth)[m].mean()),'coverage90':float(((truth>=lo)&(truth<=hi))[m].mean()),'width90_mw':float((hi-lo)[m].mean())}
        metrics['ramp'][name]={}
        for lag in (1,3,6):
            rm=m[:,lag:]&m[:,:-lag]; rs=s[:,:,lag:,:,c]-s[:,:,:-lag,:,c]; ry=y[:,lag:,:,c]-y[:,:-lag,:,c]
            metrics['ramp'][name][str(lag)]={'crps_mw':float(crps(rs,ry)[rm].mean()),'median_mae_mw':float(np.abs(np.median(rs,axis=1)-ry)[rm].mean())}
    # Joint score on per-resource system aggregates, bounded member subset.
    k=min(s.shape[1],80); agg=np.asarray(s[:,:k].sum(axis=3)); truth=np.asarray(y.sum(axis=2)); scores=[]
    for issue in range(agg.shape[0]):
        x=agg[issue].reshape(k,-1).astype(np.float64); target=truth[issue].reshape(-1).astype(np.float64)
        attraction=np.linalg.norm(x-target,axis=1).mean(); square=np.sum(x*x,axis=1); distance=np.sqrt(np.maximum(square[:,None]+square[None,:]-2*x@x.T,0.0)).mean()
        scores.append(attraction-0.5*distance)
    metrics['joint']['aggregate_energy_score_mw']=float(np.mean(scores))
    out.mkdir(parents=True); (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(12,4)); names=list(CHANNELS)
    axes[0].bar(names,[metrics['channels'][n]['crps_mw'] for n in names]); axes[0].set_title('CRPS (MW)')
    axes[1].bar(names,[100*metrics['channels'][n]['coverage90'] for n in names]); axes[1].axhline(90,color='k',ls='--'); axes[1].set_title('90% coverage')
    axes[2].bar(names,[metrics['channels'][n]['median_mae_mw'] for n in names]); axes[2].set_title('Median MAE (MW)')
    fig.tight_layout(); fig.savefig(out/'ordinary_metrics.png',dpi=160); plt.close(fig)
    lines=['# Shandong91 Raw Body result summary','',f"- Scenario shape: `{metrics['shape']}`",f"- Finite ratio: `{metrics['finite_ratio']}`",f"- Aggregate Energy Score: `{metrics['joint']['aggregate_energy_score_mw']:.6f} MW`",'','| Resource | CRPS MW | Median MAE MW | Forecast MAE MW | 90% coverage | 90% width MW |','|---|---:|---:|---:|---:|---:|']
    for n in names:
        d=metrics['channels'][n]; lines.append(f"| {n} | {d['crps_mw']:.6f} | {d['median_mae_mw']:.6f} | {d['forecast_mae_mw']:.6f} | {100*d['coverage90']:.2f}% | {d['width90_mw']:.6f} |")
    lines += ['', 'Event/extreme-tail evaluation: NOT APPLICABLE in v1; no 91-node event definition was introduced.']
    (out/'RESULT_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__': main()
