# Shandong91 Solar nonnegative modeling data version

## Status and semantics

`reliable_channel_training_v2_solar_nonnegative` is an isolated modeling data
version derived from `reliable_channel_training_v1`. It applies the declared
rule below; it does **not** claim that the original negative `REAL_POWER`
measurements were erroneous:

```text
on actual_valid & node_type & channel_train Solar observations, in 15-minute MW space:
projected_actual = max(raw_actual, 0)
```

Solar forecast and every validity mask are unchanged. Wind and Load values are
unchanged. Invalid storage cells remain invalid and are never activated.

## Rebuilt chain

The builder projects 15-minute Solar actual MW, recomputes Solar normalized
actual and residual, aggregates projected observations to hourly using the
published >=2-of-4 validity rule, then rebuilds train/validation/test 168-hour
actual and residual arrays. Consequently both diffusion supervision and the
strictly pre-window 24-hour recent-error condition use the same projected data.
Forecast-state thresholds continue to be fitted at runtime from projected
training-period hourly actual only and are reused for validation/test/generation.

Every modified observation is recorded in
`reports/solar_nonnegative_projection_records.npz` using time index, node id,
raw value, projected value, correction and reason code. The full data directory
is intentionally excluded from Git.

## Projection statistics

- Modified unique 15-minute observations: 710,231
- Total upward correction over MW samples: 227,294.734375 MW-samples
- Equivalent 15-minute correction energy: 56,823.683594 MWh
- Absolute correction: q50 0.16 MW, q90 0.87 MW, q95 1.15 MW,
  q99 2.16 MW, maximum 4.27 MW
- Unique effective hourly Solar negatives: 169,191 before, 0 after
- Overlapping 168-hour window elements corrected:
  - train: 928,164
  - validation: 88,829
  - test: 98,672

Node 48 retains the existing 172 MW unit-detail-sum normalization scale. The
alternate reported value 254.56 MW remains documented and unresolved; no
capacity was re-inferred.

## Verification

Full data checks pass:

- effective Solar actual is nonnegative at 15-minute, hourly and all splits;
- all actual/forecast/residual validity masks are identical to V1;
- Wind and Load actual arrays are identical to V1;
- Solar forecast arrays are identical to V1;
- MW residual identity is exact;
- normalized residual identity and inverse normalization pass float tolerance;
- projection record count matches the projection mask;
- node 48 scale is exactly 172 MW;
- output manifest hashes validate;
- validation sample 0 recent error exactly equals projected hourly residual for
  2025-10-31 00:00 through 23:00, strictly before the 2025-11-01 00:00 window;
- projected-data forecast-state threshold hash is
  `d8dee9e0bde92bdc0dfc44ebfdb5336b53f90117088c4d5450094b955ce2cec5`;
- bounded V2 CPU forward/backward, three-resource gradients, optimizer update,
  checkpoint reload and sampler checks pass;
- one-batch/one-epoch checkpoint plus bounded two-member generation/evaluation
  integration passes. This was an engineering probe, not formal training.

Target-server full-model CUDA/AMP preflight remains mandatory and **NOT RUN**
locally. Formal training and formal 500-member generation were **NOT RUN**.

## Build and launch

Build/verify the untracked data directory:

```bash
python tools/build_shandong91_solar_nonnegative.py \
  --source reliable_channel_training_v1 \
  --output reliable_channel_training_v2_solar_nonnegative
```

Formal all-in-one entry after the server CUDA gate:

```bash
export OUTPUT_ROOT=/root/autodl-tmp/shandong91_v2_solar_nonnegative_$(date +%Y%m%d_%H%M%S)
bash run_shandong91_raw_body_v2_solar_nonnegative_pipeline.sh
```

The dedicated config is
`configs/shandong91/raw_body_v2_faithful24_solar_nonnegative.yaml`. Training and
generation manifests record the data identifier, transformation rule, node 48
capacity convention and SHA-256 of the data output manifest.
