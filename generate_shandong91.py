"""Generate Shandong91 scenarios from a formal Raw Body checkpoint."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, yaml
from datasets.shandong91_reliable import Shandong91ReliableDataset
from train_shandong91 import build_model, move_batch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--config',default='configs/shandong91/raw_body_heterogeneous_formal_v1.yaml'); p.add_argument('--split',choices=('validation','test'),default='test')
    p.add_argument('--n-samples',type=int,default=500); p.add_argument('--seed',type=int,default=424242); p.add_argument('--member-chunk',type=int,default=5); p.add_argument('--inference-steps',type=int,default=None)
    a=p.parse_args(); out=Path(a.output_dir)
    if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
    if not torch.cuda.is_available(): raise RuntimeError('formal generation requires CUDA')
    cfg=yaml.safe_load(Path(a.config).read_text(encoding='utf-8')); ds=Shandong91ReliableDataset(cfg['data']['data_path'],a.split)
    device=torch.device('cuda:0'); model=build_model(cfg,ds,device); ckpt=Path(a.run_dir)/'checkpoints'/'best.pt'
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    if saved.get('model_identifier') != model.denoiser.architecture: raise ValueError('checkpoint model identifier mismatch')
    model.load_state_dict(saved['model_state_dict'],strict=True); model.eval()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    steps=int(a.inference_steps or cfg['sampling']['inference_steps']); shape=(len(ds),a.n_samples,168,91,3)
    scenarios=np.empty(shape,dtype=np.float32); valid=np.empty((len(ds),168,91,3),dtype=bool)
    actual=np.empty((len(ds),168,91,3),dtype=np.float32); forecast=np.empty_like(actual)
    node=ds.node_type_mask.to(device).view(1,1,91,3); train=ds.channel_train_mask.to(device).view(1,1,91,3)
    for issue in range(len(ds)):
        raw=ds[issue]; batch=move_batch({k:v.unsqueeze(0) for k,v in raw.items()},device)
        publication=node & batch['forecast_valid_mask'].bool(); generation=publication & train
        actual[issue]=ds.denormalize_mw(batch['actual'])[0].cpu().numpy(); forecast[issue]=ds.denormalize_mw(batch['forecast'])[0].cpu().numpy(); valid[issue]=publication[0].cpu().numpy()
        for start in range(0,a.n_samples,a.member_chunk):
            count=min(a.member_chunk,a.n_samples-start)
            f=batch['forecast'].expand(count,-1,-1,-1); fv=batch['forecast_valid_mask'].expand(count,-1,-1,-1); tm=batch['time_mark'].expand(count,-1,-1)
            gm=generation.expand(count,-1,-1,-1); noise=torch.randn_like(f)
            with torch.inference_mode(), torch.autocast(device_type='cuda',dtype=torch.float16):
                residual=model.sample_ddim(f,fv,tm,gm,inference_steps=steps,initial_noise=noise).float()
            generated=(ds.denormalize_mw(f)+ds.denormalize_mw(residual)).masked_fill(~publication.expand_as(residual),0.0)
            scenarios[issue,start:start+count]=generated.cpu().numpy()
        print(json.dumps({'issue':issue+1,'total':len(ds)}),flush=True)
    out.mkdir(parents=True); np.save(out/'actual_scenarios_mw.npy',scenarios); np.save(out/'actual_mw.npy',actual); np.save(out/'forecast_mw.npy',forecast); np.save(out/'valid_mask.npy',valid)
    meta={'checkpoint':str(ckpt),'checkpoint_epoch':saved['epoch'],'split':a.split,'n_samples':a.n_samples,'seed':a.seed,'shape':list(shape),'inference_steps':steps,'residual_sign':'actual=forecast+residual','inactive_policy':cfg['sampling']['inactive_policy']}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
if __name__=='__main__': main()
