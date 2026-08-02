# Station24 State V1 experiment

This experiment is a new, paired validation experiment. It does not overwrite or
extend the earlier spatial or condition ablation output directories.

## Question being tested

Does a compact, forecast-derived state representation improve ramp and extreme
event scenario quality beyond feeding raw 3 h and 6 h forecast ramps directly?

Both runs use the same residual target, Fixed Graph, recent 24 h realized-error
context, training split, seed, optimizer, stopping rule, 80 ensemble members, 500
reverse steps, physical projection, and validation data.

| Variant | Additional condition |
|---|---|
| `ramp36_control` | Raw forecast ramps at 3 h and 6 h |
| `state_v1_fixed_graph` | Four node states: low output, high output, upward ramp, downward ramp |

The State V1 run removes the raw ramp branch, so the two representations are not
duplicated in the treatment model.

## Causal data flow

```text
current issued 168 h station forecast
  -> train-fitted per-station state thresholds
  -> node_state [batch, 24, 4, 168]
  -> shared lightweight temporal state encoder
  -> multiscale node embeddings
  -> gated state FiLM in the residual U-Net

station forecast + time/lead/type + recent 24 h known errors
  -> existing condition encoder
  -> existing FiLM

geographic adjacency
  -> unchanged Fixed Graph bottleneck

diffusion target
  -> future actual minus current issued forecast
```

Thresholds are fitted from training actuals only. Validation/test state tensors are
then computed from the current issued forecast and astronomical daylight mask only.
Future actuals and future residuals are never used as generation-time conditions.

## State definitions

- Low output severity: forecast shortfall below the station's training 20th percentile.
- High output severity: forecast excess above the training 90th percentile.
- Upward ramp severity: positive 3 h/6 h forecast ramps normalized by training ramp scales.
- Downward ramp severity: negative 3 h/6 h forecast ramps normalized by training ramp scales.

Solar output states are masked outside astronomical daylight. Values are clipped at
3 to reduce sensitivity to rare denominator effects. Volatility and dynamic graph
conditions are intentionally excluded from this first version.

## Run on the server

```bash
cd /root/autodl-tmp/DM
bash run_station24_state_v1_pipeline.sh
```

The command detaches one background pipeline. It trains and validates the control,
then trains and validates State V1, generates the paired comparison, and finally
creates a compressed archive. Outputs are isolated under:

```text
outputs_shandong/station24/state_v1_<timestamp>/
outputs_shandong/station24/station24_state_v1_<timestamp>.tar.gz
logs/station24/station24_state_v1_<timestamp>.*
```

The test split remains locked.
