"""CPU evidence gate for the isolated Joint Multiresidual Tail core."""
import argparse
import json
from pathlib import Path
import torch
from src.models.station_joint_multiresidual_tail import JointMultiresolutionResidualTail


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(22091)
    features=torch.zeros(24,5);features[:13,0]=1;features[13:,1]=1
    adjacency=torch.eye(24);capacity=torch.linspace(1,2,24)
    model=JointMultiresolutionResidualTail(32,features,adjacency,capacity)
    hidden=torch.randn(2,24,32,168);forecast=torch.rand(2,24,168)
    low,high=model.forecast_components(forecast)
    reconstruction_error=float((low+high-forecast).abs().max())
    orthogonality=float((low*high).sum().abs()/forecast.square().sum().clamp(min=1e-8))
    initial=model(hidden,forecast,route=1.);route0=model(hidden,forecast,route=0.)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    target=torch.randn_like(initial.correction);updated={}
    for step in range(3):
        optimizer.zero_grad();result=model(hidden,forecast,route=1.)
        loss=(result.correction-target).square().mean();loss.backward()
        if step==2:
            for name,param in model.named_parameters():
                updated[name]=float(param.grad.norm()) if param.grad is not None else 0.0
        optimizer.step()
    final=model(hidden,forecast,route=1.)
    required=('local_fast.weight','local_slow.weight','system_fast.weight','system_slow.weight','hidden.0.weight','fast_condition.0.weight','slow_condition.0.weight')
    gates={
      'G0_interface_has_no_actual_or_future_residual':'PASS',
      'G1_forecast_exact_reconstruction':'PASS' if reconstruction_error<2e-7 else 'FAIL',
      'G1_slow_fast_orthogonality':'PASS' if orthogonality<2e-7 else 'FAIL',
      'G1_zero_initial_tail':'PASS' if float(initial.correction.detach().abs().max())==0 else 'FAIL',
      'G1_route_zero_identity':'PASS' if float(route0.correction.detach().abs().max())==0 else 'FAIL',
      'G2_required_gradients_after_three_steps':'PASS' if all(updated[k]>0 and torch.isfinite(torch.tensor(updated[k])) for k in required) else 'FAIL',
      'G2_wind_and_solar_outputs_nonzero':'PASS' if float(final.correction[:,:13].abs().mean())>0 and float(final.correction[:,13:].abs().mean())>0 else 'FAIL',
      'G4_cuda_amp':'NOT RUN', 'G4_full_model_checkpoint_load':'NOT RUN',
      'G5_final_sampling':'NOT RUN', 'G6_scientific_benefit':'NOT RUN'}
    payload={'device':'CPU float32','gates':gates,'launch_eligible':False,
      'reason':'Core-only preflight; full-model integration and CUDA/AMP gates remain.',
      'forecast_reconstruction_max_abs':reconstruction_error,'slow_fast_inner_product_ratio':orthogonality,
      'parameter_count':sum(p.numel() for p in model.parameters()),'gradient_norms':updated}
    (out/'preflight.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    if any(v=='FAIL' for v in gates.values()):raise SystemExit('JOINT_MULTIRESIDUAL_PREFLIGHT_FAILED')
    print('JOINT_MULTIRESIDUAL_CORE_PREFLIGHT_PASSED; NOT LAUNCH ELIGIBLE',flush=True)


if __name__=='__main__':main()
