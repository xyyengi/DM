# Station-24 Lightweight Tail Staged Structure/Capacity Search Protocol

**FROZEN BEFORE CANDIDATE TRAINING — revised 2026-10-12**

This staged protocol supersedes the earlier plan to fully train all 24
combinations. No candidate had been trained when this revision was frozen. The
24 configs remain reproducible artifacts; they are not 24 discovered good
models and are no longer the default execution list. This bounded search does
not establish a global optimum. Test remains locked throughout selection.

## What exists and what it proves

The 24 configs are `4 widths × 3 fusion depths × 2 Haar levels`. Implementation,
exact parameter counting, CPU shape/route/Haar and functional-branch update
tests, and restartable launch wiring are complete. These establish structural
consistency and executable CPU code. They do not establish CUDA/AMP safety,
real-task learning, probability quality, extreme-event skill, recovery skill,
or superiority of any candidate.

The reference Original Lightweight / Joint Multiresolution Residual Tail is
`channels=24`, `fusion_layers=2`, `haar_levels=3`, with exactly 20,588 newly
added trainable Tail parameters. They were not selected from Raw. Raw stays
frozen/eval. Depth 2 is historical; depth 1 removes only its second dilated
convolution; depth 3 appends a fixed `k=3` layer (Fast `d=4`, Slow `d=8`). No
internal residual skip is added and GroupNorm remains 8.

### Parameter-count objects must not be mixed

`779,248` is the frozen parameter inventory instantiated by the Lightweight/Raw
body-tail configuration. It includes inherited legacy body-tail expert modules
because `use_body_tail_experts=true`; those remain frozen during JMRT training.
`772,290` is the entire trainable model instantiated by Full Independent Tail
V1/V2 with `use_body_tail_experts=false` and
`independent_joint_tail_training=true`. Their 6,958 difference is a module-
inventory/configuration difference, not an error bar and not Tail compression.
Report efficiency using the newly trainable Lightweight Tail count; never mix
these two whole-model objects in a compression ratio.

## Frozen common recipe

All stages use the same Raw checkpoint, train/validation split, normalization,
valid masks, 60% event-balanced sampling, all-valid epsilon objective,
ramp/shape/slow weights `0.18/0.14/0.10`, optimizer, LR, batch/accumulation,
diffusion schedule, conditions and event definition. Auxiliary timestep
compensation is OFF. Loss weights, sampling ratio and architecture outside the
JMRT search variables are not searched. Test is unused until one final recipe,
member ratio and evaluation protocol are frozen, and cannot reopen selection.

## Stage 0: mandatory gates

Before paid work: `c24_d2_h3=20,588`; shape/mask/graph and Haar checks; Tail-off
equals Raw; every functional Tail branch updates; optimizer contains Tail only;
Raw parameters/buffers unchanged; no future actual/residual generation input;
checkpoint save/reload; target-server CUDA/AMP forward/backward. The server also
verifies complete Raw member arrays, checkpoint, validation windows, generation
seed and member ordering. Any mandatory failure stops training. `NOT RUN` is not
`PASS`; no Station-91 process may be interrupted or competed with.

## Stage 1: six-candidate one-third-budget screen

Default candidates are:

- `c16_d2_h3` — smaller capacity;
- `c24_d2_h3` — matched reference;
- `c32_d2_h3` — moderate width;
- `c48_d2_h3` — larger width;
- `c24_d3_h3` — depth-only contrast;
- `c24_d2_h2` — Haar-only contrast.

The currently loaded formal Station-24 train split has 290 windows, batch 8 and
accumulation 2: `ceil(290/8)=37` batches and 19 optimizer updates/epoch (the
final partial accumulation is stepped). The full 240-epoch cap is 4,560 updates; screening
pauses after 1,520 updates, exactly 80 completed epochs. This is a pause, not a
compressed schedule. The project has no LR scheduler (constant AdamW LR), so
scheduler state is explicitly `none`; it must not be invented. Screening does
not early-stop before the registered pause. The pause checkpoint contains model,
optimizer, AMP scaler, EMA, update count, best validation state, history,
Python/NumPy/Torch/CUDA RNG and train-sampler generator state. Resume continues
toward the original cap and reinstates the original validation/patience rule.

The Stage-1 reference is the matched `c24_d2_h3` 1,520-update checkpoint. The
historical fully trained reference is context only, not its replacement.

### Lightweight validation evaluation

Reuse the same verified Raw pool and fixed 120-member prefix. Generate 30 Tail
members with seed 424242 and merge `120 Raw + 30 Tail` (20%). Use every existing
validation window and identical Raw member ordering. The exploratory 30-Tail
hit rate is not directly comparable to formal 100-Tail hits, and absence of one
rare hit cannot alone eliminate a candidate.

Retain numeric evaluation only: wind/solar/renewable CRPS; 90% coverage error,
width and interval score jointly; persistent-event depth/onset/duration plus
descriptive hits; wind 1/3/6 h ramps; frozen-anchor strong-drop 3/6 h half-
recovery probability, time and amplitude. No bulk plots or archive in Stage 1.
Training loss alone is never a selection criterion.

## Pre-registered promotion rule

Apply integrity checks, then the registered finite-search Body guardrails:
relative to Raw, renewable CRPS <=2% degradation, Energy Score <=1%, wind CRPS
<=2%, solar-daylight CRPS <=2%, and overall spatial/wind-solar correlation RMSE
<=5%. These are engineering screens, not universal statistical thresholds.
Coverage error, width and interval score are read jointly.

No arbitrary weighted score is allowed. Retain the descriptive Pareto set over
ordinary quality, persistent events, ramp/recovery and Tail parameters. One
event, sign or lag cannot decide promotion. Select at most two promising non-
baseline candidates plus `c24_d2_h3` for full budget. If ranks are close,
families disagree, or learning is still materially improving at update 1,520,
extend at most three non-baseline candidates to 3,040 updates (160 epochs) before
deciding. Do not force elimination. If indistinguishable, prefer fewer
parameters, then the recipe closest to reference.

## Stage 2: full budget and formal evaluation

Resume promoted checkpoints; do not retrain. Continue toward 240 epochs with
the original validation interval, patience and checkpoint selection semantics.
A historical reference is reusable only if config, budget, initialization and
evaluation hashes match; otherwise complete the matched reference.

Formal evaluation restores `400 Raw + 100 Tail`, the frozen seed and complete
chain: ordinary/calibration metrics, persistent events, joint/spatial structure,
ramps, drop-recovery, necessary plots, summary and archive.

## Targeted additions, repeats and stop

Never return automatically to 24 full runs and never start a loss-weight search.
Only if evidence clearly points to the width boundary, or separately supports
width and depth, may at most two targeted configs be added after writing their
rationale. Select one candidate and the reference for final stability. Each
totals training seeds `2027`, `3407`, `7919` with paired generation seeds
`424242`, `271828`, `314159`; report the two sources of variation separately.

If no stable advantage exists, retain the reference and stop. Full V2 differs
in adaptation mechanism as well as capacity. Limited independent validation
extremes bound generalization claims.

After every candidate and stage, report what should continue, why, remaining
uncertainty and the next question. Final delivery contains a summary CSV, only
necessary plots and `FINAL_SELECTION.md`, answering capacity limitation,
depth/Haar effects, replacement evidence and the stopping basis.
