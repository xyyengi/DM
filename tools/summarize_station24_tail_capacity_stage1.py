"""Collect Stage-1 numeric evidence without making an automatic promotion."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import pandas as pd

TAGS=('c16_d2_h3','c24_d2_h3','c32_d2_h3','c48_d2_h3','c24_d3_h3','c24_d2_h2')
def flatten(value,prefix=''):
    rows=[]
    if isinstance(value,dict):
        for k,v in value.items():rows.extend(flatten(v,f'{prefix}.{k}' if prefix else str(k)))
    elif isinstance(value,(int,float)) and not isinstance(value,bool):rows.append((prefix,float(value)))
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);records=[];status=[]
    for tag in TAGS:
        folder=root/tag;state='missing'
        if (folder/'status.env').is_file():
            state=dict(line.split('=',1) for line in (folder/'status.env').read_text().splitlines() if '=' in line).get('state','unknown')
        status.append({'candidate':tag,'state':state})
        metrics=folder/'mixture_raw120_tail30_n150/metrics.json'
        if metrics.is_file():
            for key,value in flatten(json.loads(metrics.read_text(encoding='utf-8'))):
                records.append({'candidate':tag,'source':'mixture_metrics','metric':key,'value':value})
        post=folder/'stage1_numeric_evaluation'
        for path in post.rglob('*.csv') if post.is_dir() else ():
            try: frame=pd.read_csv(path)
            except Exception: continue
            for index,row in frame.iterrows():
                identity=';'.join(f'{k}={row[k]}' for k in frame.columns if not pd.api.types.is_number(row[k]) and pd.notna(row[k]))
                for key,value in row.items():
                    if isinstance(value,(int,float)) and pd.notna(value):
                        records.append({'candidate':tag,'source':str(path.relative_to(folder)),'metric':f'row={index};{identity};{key}','value':float(value)})
    pd.DataFrame(status).to_csv(root/'stage1_status.csv',index=False)
    pd.DataFrame(records,columns=['candidate','source','metric','value']).to_csv(root/'stage1_numeric_metrics_long.csv',index=False)
    complete=[r['candidate'] for r in status if r['state']=='complete']
    lines=['# Stage 1 review gate','',f'Completed candidates: {len(complete)}/6: '+(', '.join(complete) or 'none'),'',
           'Automatic promotion is intentionally disabled. Review ordinary quality, persistent events, ramp/recovery and parameter count under SEARCH_PROTOCOL.md before writing a promotion decision.',
           '', 'Still uncertain until review: capacity limitation, depth effect, Haar effect, and whether any candidate merits full-budget continuation.',
           '', 'Next stage, only after a written decision: resume at most two non-baseline candidates plus c24_d2_h3, or extend at most three ambiguous non-baselines to 3,040 updates.']
    (root/'STAGE1_REVIEW_REQUIRED.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'STAGE1_SUMMARY_COMPLETE root={root} complete={len(complete)}')
if __name__=='__main__':main()
