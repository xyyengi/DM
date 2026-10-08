# Shandong91 heterogeneous Raw Body V2 faithful24

## Decision and evidence

The migration source is the pure Station-24 Raw Body experiment
`station24_geo_history_actual_dual_168h` (`geo_history_actual_dual`), not External
Tail and not a Body+Tail mixture. Its best checkpoint is:

`outputs_shandong/station24/historical_dual_graph_20260817_200113/training/20260817_200225_station24_geo_history_actual_dual_20260817_200113_seed2027/checkpoints/model_best.pt`

The recorded source epoch is 240 and validation epsilon MSE is
0.13130853033584097. The 500-member validation comparison reports station Wind
CRPS 0.09072 and coverage90 0.87185, aggregate Wind CRPS 127.95553 MW and
coverage90 0.83696, daylight Solar CRPS 0.05563 and coverage90 0.89903, Energy
Score 6.62480 and spatial-correlation RMSE 0.08054.

A later Body-tail-MoE container evaluated through its *raw route* is marginally
better on some metrics (for example Wind CRPS 0.09058 and Energy Score 6.59278)
but has a worse validation epsilon MSE (0.38196) and is not a distinct pure-Raw
training result. It is therefore evidence that the frozen body remained strong,
not the migration checkpoint selected here. This avoids manufacturing a single
"best" from metrics with different winners.

Primary evidence:

- `configs/station24_geo_history_actual_dual_168h.yaml`
- `outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/body_tail_initialization.json`
- `outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/comparisons/history_vs_body_tail_raw/comparison_report.md`
- `outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242/metrics.json`

## Code-level comparison and classification

| Area | Mature Station-24 Raw | Shandong91 V1 | V2 decision | Class |
|---|---|---|---|---|
| Temporal body | 3-level ResUNet, widths 32/64/128, timestep embedding 128, dropout 0.1 | Same basic temporal body | Preserved | inherited |
| Condition encoder | Forecast, validity, calendar, lead, static station data; normalized/refined multi-scale down blocks | Forecast/calendar/static projection with lighter one-convolution down blocks | Restored the mature normalized/refined down blocks, except semantically invalid lead | mature detail recovered |
| Recent state | Previous 24-hour observed error with availability gate | Absent | Restored from hours strictly before issue time | potential omission recovered |
| Forecast state | Four low/high/ramp state features, multi-scale encoder, global state and FiLM in ResBlocks | Absent | Restored as four features per resource (12 total), thresholds fitted on train-only unique hourly actuals, future state computed from forecast only | potential omission recovered / necessary heterogeneous adaptation |
| FiLM | Timestep + condition and state FiLM through encoder/bottleneck/decoder | Timestep/condition FiLM | State FiLM restored at every scale | potential omission recovered |
| Graph | Geographic graph plus train-only historical-actual graph; early parallel graph fusion and bottleneck propagation | Early parallel fusion and bottleneck propagation on one physical graph | Existing propagation retained; one published 91-node physical graph retained | inherited / second graph not migratable |
| Resources | Station-specific Wind/Solar scalar output | Explicit Wind/Solar/Load with shared projections and type flags | Explicit `[B,168,91,3]`, shared projections, resource-aware state pooling | necessary adaptation |
| Diffusion | epsilon target, 500 linear betas 1e-4 to 0.04, posterior DDPM | Same forward schedule, formal DDIM50 generation | Same target/schedule; full DDPM500 is formal default | mature mechanism restored |
| Objective | masked epsilon MSE | elementwise `effective_mask` MSE | Unchanged elementwise `effective_mask` mean; no resource balancing | contract preserved |
| Training | AdamW, lr 1e-4, wd 1e-4, effective batch 16, clip 1, EMA .999, validation/5, patience40 | Adam, batch2, no EMA/early stopping | AdamW, accumulation 8 (batch2, effective16), clip1, EMA, validation/5, patience40 | mature training restored; memory adaptation |
| Target | `actual-current_forecast`, condition-dependent residual scale | `actual-forecast`, fixed node/resource scales | 91 residual sign and published node/resource scaling retained | necessary contract adaptation |
| Generation | posterior DDPM and physical projection including astronomical Solar rule | DDIM50; no extra projection | DDPM500 EMA default; no Solar clipping/projection | mature sampler restored / projection not migratable |

The relevant implementation anchors are
`src/models/station_conditioned_diffusion.py`,
`configs/station24_geo_history_actual_dual_168h.yaml`,
`src/models/shandong91_conditioned_diffusion.py`, and
`configs/shandong91/raw_body_heterogeneous_formal_v1.yaml`.

## Explicitly not migrated

- **Lead condition:** Station-24 has a real lead encoding. Shandong91
  `window_position` is not forecast horizon or issuance lead, so V2 has no lead
  argument or substitute.
- **Historical-actual second graph:** there is no validated heterogeneous
  Shandong91 counterpart. The published physical graph is used alone; no
  correlation graph is invented, especially while Solar negative-value semantics
  remain unresolved.
- **Station-24 conditional residual scaling:** the 91-node node/resource
  normalization and inverse-normalization contract is retained.
- **Astronomical Solar clipping:** negative Solar observations are not silently
  modified. Evaluation reports physical violations read-only.
- Tail, event-balanced sampling, auxiliary losses, self-localization, Protected
  Partial and resource-balanced loss remain disabled.

## Independent V2 package

- Model: `src/models/shandong91_faithful24_diffusion.py`
- Causal data view: `datasets/shandong91_faithful24.py`
- Config: `configs/shandong91/raw_body_v2_faithful24.yaml`
- Training: `train_shandong91_v2.py`
- Generation: `generate_shandong91_v2.py`
- Evaluation: `tools/evaluate_shandong91_v2.py`
- Bounded gate: `tools/preflight_shandong91_raw_body_v2.py`
- Formal pipeline/finalize: `run_shandong91_raw_body_v2_faithful24_pipeline.sh` and
  `run_shandong91_raw_body_v2_faithful24_finalize.sh`

All paths and model identifiers are distinct from V1. Formal generation writes
raw 500-member validation scenarios plus publication and effective masks. The
evaluator reports resource CRPS, coverage, interval width/score, residual tails,
conditional spread, node/system metrics, ramp CRPS, trajectory-comparable ACF,
spatial-correlation RMSE, joint Energy Score and read-only physical violations.

## Verification state

Local CPU checks pass for the base data contract, causal condition contract,
forward/backward, finite and non-zero Wind/Solar/Load gradients, inactive-mask
invariance, a core optimizer update, save/reload identity, bounded sampler shape
and finiteness, inactive residual zeroing, residual sign/inverse normalization,
and V1/V2 isolation. Formal training and full 500-member generation are **NOT
RUN**. CUDA/AMP preflight is mandatory and fail-closed in the formal launcher.
