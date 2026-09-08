"""Offline checkpoint branch audit; no optimizer step or formal generation.

Run with python -m tools.audit_station24_branch_gradients --run-dir ... --output ...
Uses validation labels only for a diagnostic forward/backward, never as a claim
of causal generation performance.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from station_dataset import get_station_dataloader, load_station_static_data
from station_graph_prior import load_generation_graphs
from station_jstd_targets import build_station_jstd_target_arrays
from src.models.station_conditioned_diffusion import Station24DiffusionModel
from src.models.station_joint_decomposed_tail import same_length_average


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data-path', default='diffusion_input_station')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2027)
    run = Path(args.run_dir)
    config = yaml.safe_load((run / 'config_used.yaml').read_text(encoding='utf-8'))['model']
    checkpoint = torch.load(run / 'checkpoints/model_best.pt', map_location='cpu', weights_only=False)
    static = load_station_static_data(args.data_path)
    primary, secondary, _ = load_generation_graphs(Path(args.data_path), run, config, checkpoint)
    model = Station24DiffusionModel(config, static['station_features'], primary,
                                   static['station_capacities'], secondary)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.configure_jstd_training()
    model.eval()
    thresholds = json.loads((run / 'jstd_event_targets.json').read_text(encoding='utf-8'))['thresholds']
    targets = build_station_jstd_target_arrays(args.data_path, 'val', thresholds)
    _, dataset = get_station_dataloader(args.data_path, 'val', checkpoint['residual_scale'],
        batch_size=2, seed=2027, num_workers=0, condition_config=config,
        state_thresholds=checkpoint.get('state_thresholds'), jstd_targets=targets)
    selected = [i for i, active in enumerate(targets.event_active) if active > 0]
    batch = next(iter(DataLoader(Subset(dataset, selected), batch_size=len(selected))))
    captured = []
    handle = model.denoiser.jstd_tail.register_forward_hook(lambda m, a, out: captured.append(out))
    rows = []
    for step in (10, 100, 300):
        model.zero_grad(set_to_none=True)
        captured.clear()
        torch.manual_seed(2027)
        loss = model(batch, timestep=torch.full((len(selected),), step, dtype=torch.long),
                     noise=torch.randn_like(batch['residual_target']))
        loss.backward()
        out = captured[-1]
        slow, fast = out.slow_correction.detach(), out.fast_correction.detach()
        def energy(x): return float(x.square().mean())
        def rms(x): return energy(x) ** .5
        def grad(prefix):
            values = [p.grad.square().sum() for n, p in model.denoiser.jstd_tail.named_parameters()
                      if n.startswith(prefix) and p.grad is not None]
            return float(torch.stack(values).sum().sqrt()) if values else 0.
        denominator = (energy(slow) * energy(fast)) ** .5
        rows.append(dict(step=step, loss=float(loss.detach()), slow_rms=rms(slow), fast_rms=rms(fast),
            slow_fast_cosine=float((slow*fast).mean()) / max(denominator, 1e-12),
            combined_to_separate_energy=energy(slow+fast)/max(energy(slow)+energy(fast),1e-12),
            fast_low12_rms_ratio=rms(same_length_average(fast,12))/max(rms(fast),1e-12),
            slow_raw_gradient=grad('slow_raw.'), fast_raw_gradient=grad('fast_raw.'),
            slow_mask_gradient=grad('slow_mask.'), fast_mask_gradient=grad('fast_mask.'),
            frozen_parameters_with_gradient=[n for n,p in model.named_parameters()
                                             if not p.requires_grad and p.grad is not None]))
    handle.remove()
    result = dict(run=str(run), checkpoint_epoch=checkpoint['epoch'], event_issues=selected,
        scope='label-conditioned offline epsilon forward/backward; not final-sample efficacy',
        scale_parameterization=model.denoiser.jstd_tail.segment_scale_parameterization,
        final_sample_branch_ablation='NOT RUN', cuda_amp='NOT RUN', rows=rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
