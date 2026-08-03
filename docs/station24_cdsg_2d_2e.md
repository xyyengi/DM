# Station24 Experiments 2D and 2E

## Purpose

These validation-only experiments isolate two wind deficiencies observed in 2B:

- condition-dependent under/over-dispersion;
- attenuated or mistimed 1 h, 3 h, and 6 h ramps.

The existing 2B result is reused. Test data remain locked.

## Experiment 2D

2D keeps the complete 2B model backbone and changes only residual normalization.
Solar stations retain their original per-station standard deviation. Wind stations
use a train-fitted factorized conditional scale based on information available at
generation time:

```text
station intercept
  x forecast-level factor
  x lead-day factor
  x absolute 3 h forecast-ramp factor
  x absolute forecast-revision factor
```

The factors are fitted exclusively from `train_residual.npy`. Bin edges and
factors are stored in `residual_scale.json` and the checkpoint. Missing forecast
revision uses a neutral multiplier of 1. Validation, test, and deployment never
refit the scale.

The scale is built as `[station, 168]` for every issuance. It is used both for
the training target and the generation inverse transform:

```text
standardized target = residual / conditional scale
generated residual = generated standardized residual * conditional scale
```

No model parameter or power channel is added.

## Experiment 2E

2E is exactly 2D plus a small physical-power ramp reconstruction objective at
1 h, 3 h, and 6 h. Its total weight is `0.05`, with lag weights
`[0.5, 0.3, 0.2]`. A signal-to-noise-ratio weight suppresses unstable clean-data
reconstruction at very noisy diffusion steps.

The auxiliary objective operates on reconstructed physical normalized power,
not on condition channels. It adds no model parameter and does not use future
actual power as an inference condition. Actual power is used only as the
training target, as in the ordinary diffusion loss.

## Automatic server pipeline

```bash
cd /root/autodl-tmp/DM
bash run_station24_cdsg_2d_2e_pipeline.sh
```

The launcher immediately returns after starting a detached process. The worker:

1. verifies the branch, clean worktree, CUDA, data, and config parity;
2. finds the latest matching 2B validation result, unless
   `REFERENCE_2B_RESULT` is explicitly supplied;
3. trains 2D and generates 80 validation members;
4. trains 2E and generates 80 validation members;
5. evaluates 80%, 90%, and 95% intervals and all existing joint/extreme metrics;
6. compares 2B vs 2D, 2D vs 2E, and 2B vs 2E;
7. runs the wind 1 h/3 h/6 h event-timing diagnostic;
8. writes one downloadable `.tar.gz` archive.

The status, log, stop command, result paths, and archive path are printed by the
launcher and recorded under `logs/station24/`.
