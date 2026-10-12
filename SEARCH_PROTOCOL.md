# Station-24 Lightweight Tail Structure/Capacity Search Protocol

**FROZEN BEFORE CANDIDATE TRAINING — 2026-10-12**

This protocol is frozen before any new candidate result is inspected. It is a
bounded empirical search, not a claim of global optimality. Test remains locked.

## Verified reference

The reference is Original Lightweight / Joint Multiresolution Residual Tail
(`station24_lightweight_joint_tail_v2_fair_168h`). Raw is frozen and kept in
evaluation mode. The additive Tail has exactly 20,588 trainable parameters at
`channels=24`, `fusion_layers=2`, `haar_levels=3`. Those are newly instantiated
Tail parameters, not parameters selected from Raw. The reference fusion stack is:

1. `Conv1d(2C,C,k=5,d=1,p=2) + GroupNorm(8) + SiLU`;
2. Fast: `Conv1d(C,C,k=3,d=2,p=2) + SiLU`; Slow: the same with `d=4,p=4`.

There are no residual skip connections inside this stack.

## Frozen search space

- Tail channels: `16, 24, 32, 48`.
- Fast/Slow fusion convolution count: `1, 2, 3`.
- Haar levels: `2, 3` (4 h and 8 h block projections).
- Total: 24 candidates, including the reference `24/2/3`.

Only Fast/Slow fusion depth changes. Hidden projection, condition encoders,
system branch, output heads, graph/state encoders, and all conditions remain
unchanged. Depth 1 uses only item 1 above. Depth 2 is byte/key compatible with
the historical reference. Depth 3 appends one `C->C, k=3` convolution: Fast
uses `d=4,p=4`; Slow uses `d=8,p=8`; it is followed by SiLU. No depth has an
internal residual skip. GroupNorm remains exactly 8 groups and all channel
choices are divisible by 8.

If and only if the first-round non-dominated evidence clearly points to the
48-channel boundary, at most two 64-channel candidates may be added, using the
best boundary depth/Haar settings. No other boundary expansion is permitted.

## Fixed training and generation conditions

The following must match the formal reference config and artifacts: Raw
checkpoint; train/validation split; normalization and valid masks; 60% event-
balanced sampling; all-valid epsilon objective; auxiliary ramp/shape/slow
weights `0.18/0.14/0.10`; all other loss switches; optimizer, LR, batch,
gradient accumulation, 240-epoch cap, validation interval, patience and
checkpoint selection; diffusion schedule; condition inputs; event definition;
generation seed and evaluation code. Auxiliary timestep compensation is OFF.
Each formal comparison is `400 Raw + 100 Tail`. Loss weights, sampling ratio,
Tail ratio and member count are not search dimensions.

Validation alone is used for search. Test may run once only after one final
recipe, all hyperparameters, member ratio and evaluation protocol are frozen;
test cannot trigger another search.

## Evidence gates

Before paid training: exact `24/2/3 = 20,588` reproduction; shape/mask/graph
checks; exact slow+fast Haar reconstruction; disabled Tail equals Raw; every
trainable branch receives gradients and updates; optimizer contains Tail only;
Raw parameters and buffers are unchanged; no future actual/residual is a
generation condition; CPU forward/backward/reload; target-server CUDA/AMP
forward/backward/reload. Any mandatory failure stops training. NOT RUN is not
PASS.

Each candidate receives the complete common budget and early-stop rule, then
generation, 400+100 merge, complete evaluation, plots, summary and archive.
Candidates are never removed using early training loss. Interrupted stages are
resumed idempotently without overwriting prior outputs.

## Measurements

Body: wind/solar/renewable CRPS, Energy Score, 90% coverage error together with
interval width and interval score, overall spatial and wind-solar correlation
RMSE. Persistent events: any-hit, member-hit, onset, duration and depth using
the existing frozen event standard. Local dynamics: wind 1/3/6 h ramp
distribution/std and existing strong-drop anchored 3/6 h half-recovery,
recovery time and amplitude. Also record exact Raw/Tail/total parameters,
wall-clock train/generation/evaluation time, peak CUDA memory and hardware.

## Pre-registered screening and selection

No arbitrary weighted total score is permitted. First apply the already frozen
engineering Body guardrails: relative to Raw, renewable CRPS <=2% degradation,
Energy Score <=1%, wind CRPS <=2%, solar-daylight CRPS <=2%, and overall
spatial/wind-solar correlation RMSE <=5%. These are finite-search engineering
screens, not universal or theoretical thresholds. Calibration improvement must
not be explained solely by wider intervals: coverage error to 0.90, width and
interval score are reported jointly and paired where available.

Among candidates passing integrity and Body screens, retain the Pareto frontier
over ordinary quality, persistent-event behavior, ramp/recovery behavior and
Tail parameter count. An extreme-family improvement cannot be declared from a
single event, solar case, sign or lag; event-wise, sign/lag-stratified and paired
bootstrap evidence is reported. If matched baseline-repeat variation is
insufficient, the result is explicitly descriptive and is resolved only by the
pre-registered repeat stage, not by changing thresholds.

Choose at most three representatives: overall compromise, recovery-oriented,
and parameter-efficient (one candidate may fill multiple roles). If performance
is practically indistinguishable, choose fewer parameters, then the recipe
closest to the reference. The reference always remains a candidate.

## Repeat and stopping rules

Each selected representative and the reference use training seeds `2027` plus
two fixed additional seeds `3407` and `7919`. Each uses the same paired
generation seeds `424242`, `271828`, and `314159`. Training-seed and generation-
seed variability are reported separately. Existing runs are reused only when
their full manifest hashes match this protocol.

Stop after the 24 candidates, optional maximum two 64-channel candidates, and
the registered repeats. Do not proceed to loss, sampling, member-ratio or new-
module searches. If larger models lack stable gain, costs rise materially, or
models are indistinguishable, select the compact candidate. Keeping the
reference is a valid outcome. Full Independent V2 differs in adaptation method
as well as capacity, so its gap cannot be attributed solely to parameter count.
The small number of independent validation extremes limits generalization
claims.
