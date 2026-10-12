"""Create a deterministic prefix subset of an existing Station-24 ensemble."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from station_evaluation import evaluate_station_scenarios,save_evaluation
from tools.merge_station24_independent_tail_members import MEMBER_ARRAYS,STATIC_ARRAYS

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--members',type=int,required=True);p.add_argument('--data-path',default='diffusion_input_station')
    a=p.parse_args();src=Path(a.source);out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    meta=json.loads((src/'generation_metadata.json').read_text(encoding='utf-8'))
    if meta.get('split')!='val' or meta.get('test_used',False):raise ValueError('validation-only subset required')
    out.mkdir(parents=True)
    for name in MEMBER_ARRAYS:
        value=np.load(src/name,mmap_mode='r')
        if not 1<a.members<=value.shape[1]:raise ValueError('invalid member prefix')
        np.save(out/name,value[:,:a.members])
    for name in STATIC_ARRAYS:np.save(out/name,np.load(src/name,mmap_mode='r'))
    data=Path(a.data_path);stations=pd.read_csv(data/'station_order.csv').sort_values('channel_index').reset_index(drop=True)
    summary,station,lead=evaluate_station_scenarios(np.load(out/'actual_scenarios_normalized.npy',mmap_mode='r'),
        np.load(out/'actual_scenarios_raw_normalized.npy',mmap_mode='r'),np.load(out/'actual_data_normalized.npy',mmap_mode='r'),
        np.load(out/'forecast_data_normalized.npy',mmap_mode='r'),stations,np.load(data/'station_adjacency.npy'),
        daylight_mask=np.load(out/'station_daylight_mask.npy',mmap_mode='r'),interval_levels=(.8,.9,.95,.99),
        energy_score_member_limit=min(80,a.members))
    run=dict(meta);run.update(n_samples=a.members,evaluation_member_count=a.members,
        condition_variant=f"{meta['condition_variant']}_prefix_n{a.members}",source_result=str(src),
        member_selection=f'prefix_0_{a.members-1}',test_used=False)
    summary['run']=run;save_evaluation(out,summary,station,lead)
    (out/'generation_metadata.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    print(f'STATION24_MEMBER_SUBSET_COMPLETE output={out} members={a.members}')
if __name__=='__main__':main()
