# Auxiliary timestep compensation preflight

Only ramp/shape/slow receive a fixed per-sample timestep multiplier before reduction; base epsilon is unchanged.
The [0.25, 2.0] clipped sqrt(SNR) schedule is normalized to mean one and is not a search space.
Device: cpu; formal training: NOT RUN; generation: NOT RUN; optimizer step: NOT RUN.

| Check | Status |
|---|---|
| raw_initialization_off | PASS |
| raw_initialization_on | PASS |
| trainable_parameter_count | PASS |
| raw_frozen | PASS |
| checkpoint_state_compatible | PASS |
| schedule_mean_one | PASS |
| epsilon_unchanged_t83 | PASS |
| epsilon_unchanged_t250 | PASS |
| epsilon_unchanged_t416 | PASS |
| scaled_ramp_t83 | PASS |
| scaled_shape_t83 | PASS |
| scaled_slow_t83 | PASS |
| scaled_ramp_t250 | PASS |
| scaled_shape_t250 | PASS |
| scaled_slow_t250 | PASS |
| scaled_ramp_t416 | PASS |
| scaled_shape_t416 | PASS |
| scaled_slow_t416 | PASS |
| high_t_reduced_not_zero | PASS |
| low_t_bounded | PASS |
| raw_and_tail_state_unchanged | PASS |
| CPU_forward_backward | PASS |
| CUDA_AMP_forward_backward | NOT RUN |
