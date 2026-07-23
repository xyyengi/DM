# V5 physical-feasibility projection

Generated residuals are unconstrained even when all cleaned observations are
non-negative. The physical scenario reconstruction

`actual = forecast - generated_residual`

can therefore leave the feasible power domain.

The projection is deterministic:

- wind is clipped to `[0, wind_upper_bound]`;
- solar is clipped to `[0, solar_upper_bound]`;
- load is floored at zero;
- solar is set to zero outside the clock-hour support fitted on train data.

The wind and solar upper bounds are the channel scales recorded in
`denormalization_used.json`.

The data are provincial aggregates and do not contain site coordinates. Their
hour labels also do not align closely enough with a site-level astronomical
sunrise rule at the two boundary hours. The Stage-1 default therefore
reconstructs the train split's unique-hour solar series, computes the train-only
95th percentile at each local clock hour, and marks an hour as allowable solar
support when that percentile exceeds a small train-peak-relative tolerance.
Validation and test actuals are never used to fit the mask. For the current
export this produces the conservative support `05:00` through `18:00`.

This rule avoids both failure modes of the earlier proxy: it does not need a
fictional site coordinate for a province-wide aggregate, and it does not mistake
a cloudy daytime forecast near zero for night.

For backward-compatible sensitivity checks only, set:

```yaml
evaluation:
  solar_night_mode: forecast_threshold
  solar_night_threshold_mw: 1.0
```

The Stage-1 default is:

```yaml
evaluation:
  solar_night_mode: train_support
```

`astronomical_shandong` remains available as a sensitivity mode. It evaluates
a deliberately broad Shandong envelope (34--39 degrees north, 114--123 degrees
east) and classifies a timestamp as night only when all nine reference points
are below apparent sunrise. It is not the default because the exported
province-level hourly labels show a measurable dawn/dusk offset from this
site-style calculation.

## Artifact semantics

Generation preserves the unmodified artifacts for audit:

- `actual_scenarios.npy`: raw reconstructed scenarios;
- `metrics.json`: raw metrics;
- `samples/scenarios_raw.npz`: raw UC-format scenarios.

It also writes:

- `actual_scenarios_constrained.npy`;
- `actual_scenarios_constrained_normalized.npy`;
- `metrics_constrained.json`;
- `physical_projection.json`;
- `samples/scenarios.npz`: constrained operational scenarios.

For already saved results, run:

```bash
python tools/project_saved_scenarios.py RESULT_DIR [RESULT_DIR ...]
```

This never overwrites the raw arrays. It refuses to replace existing projection
artifacts unless `--overwrite` is explicitly supplied. For these pre-existing
runs the operational archive is written as
`samples/scenarios_constrained.npz`, so the old `samples/scenarios.npz` remains
available for audit.
