"""Freeze the 24 candidate configs and exact instantiated parameter manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.station_joint_multiresidual_tail import JointMultiresolutionResidualTail
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_lightweight_tail import CAPACITY_SEARCH_VERSION, expected_config

CHANNELS=(16,24,32,48)
DEPTHS=(1,2,3)
HAAR=(2,3)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    out=root/'configs/station24_lightweight_tail_capacity_search'
    out.mkdir(parents=True,exist_ok=True)
    feat=torch.zeros(24,5); feat[:13,0]=1; feat[13:,1]=1
    adj=torch.eye(24); cap=torch.ones(24)
    rows=[]
    for channels in CHANNELS:
        for depth in DEPTHS:
            for haar in HAAR:
                tag=f'c{channels}_d{depth}_h{haar}'
                cfg=expected_config()
                cfg['experiment'].update(
                    name=f'station24_lightweight_tail_capacity_{tag}_168h',
                    variant=f'geo_history_actual_lightweight_tail_capacity_{tag}',
                    description='Frozen Station-24 Lightweight Tail structure/capacity search candidate')
                cfg['model'].update(
                    joint_multiresidual_training_version=CAPACITY_SEARCH_VERSION,
                    joint_multiresidual_channels=channels,
                    joint_multiresidual_fusion_layers=depth,
                    joint_multiresidual_haar_levels=haar)
                path=out/f'{tag}.yaml'
                path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
                module=JointMultiresolutionResidualTail(32,feat,adj,cap,channels=channels,
                                                        groups=8,haar_levels=haar,
                                                        fusion_layers=depth)
                tail=sum(p.numel() for p in module.parameters())
                full=Station24DiffusionModel(cfg['model'],feat,adj,cap,adj)
                total=sum(p.numel() for p in full.parameters())
                raw=total-sum(p.numel() for p in full.denoiser.joint_multiresidual_tail.parameters())
                rows.append(dict(candidate=tag,channels=channels,fusion_layers=depth,
                                 haar_levels=haar,haar_block_hours=2**haar,
                                 raw_parameters=raw,tail_parameters=tail,total_parameters=total,
                                 tail_fraction_of_total=tail/total,
                                 config=str(path.relative_to(root)),
                                 config_sha256=sha256(path),formal_status='NOT_RUN'))
    manifest=root/'station24_tail_capacity_search_manifest.csv'
    with manifest.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    reference=next(r for r in rows if r['candidate']=='c24_d2_h3')
    if reference['tail_parameters'] != 20588:
        raise AssertionError(reference)
    print(f'SEARCH_MANIFEST_COMPLETE candidates={len(rows)} reference_params={reference["tail_parameters"]}')


if __name__=='__main__': main()
