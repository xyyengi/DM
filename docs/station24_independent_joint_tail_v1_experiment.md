# Station-24 Independent Joint Tail V1

## Scope and hypothesis

This experiment tests a full independent 24-station conditional diffusion tail
expert initialized from the locked Raw checkpoint. All 772,290 parameters are
trainable; the historical Raw checkpoint, configuration and result ensemble
remain separate and unchanged. The new expert has no body-tail adapter, event
classifier, oracle event conditioning or retrieval input. Historical configs
omit the independent flag and retain their prior behavior; the new slow loss
defaults to zero. No historical checkpoint is reinterpreted.

The full wind/solar residual is supervised by epsilon loss plus 0.06 times the
1/3/6 h ramp loss and 0.08 times the 12/24 h moving-average loss. Missing
observations are excluded from averaging neighborhoods. These are overlapping
output objectives, not orthogonal components or separate slow/fast heads.
No cancellation-free or causal-benefit conclusion follows from this design.

Train-fitted event labels only set training sampler weights. The expected
fraction of event-bearing issuance draws is 60%; finite epochs need not have
exactly 60%. Validation uses the natural issuance distribution and records the
same joint objective without inverse replay weighting. The first CPU epoch
drew 168 event-bearing windows out of 290 draws (57.93%).

The fixed output budgets are development 80 Raw + 20 tail = 100 and formal
400 Raw + 100 tail = 500. Prefixes reuse the locked Raw result. Independent
sampling with the same seed does not imply paired stochastic trajectories.
The small local 3+2 merge is an interface check, not the development gate.

## Evidence manifest: 2026-09-10

Machine-readable evidence, commands, source/config/checkpoint/graph hashes,
per-parameter gradients and loss values are in
[evidence/station24_independent_joint_tail_v1_cpu_20260910.json](evidence/station24_independent_joint_tail_v1_cpu_20260910.json).
The containing Git commit identifies the reviewed implementation.
Environment: Windows, dm_preflight, Python 3.10.20, PyTorch 2.14.0+cpu.

| Check | Status | Evidence / scope |
|---|---|---|
| Raw initialization compatibility | PASS | 387 state tensors, 772,290 trainable parameters |
| Gradients and optimizer updates | PASS | All 181 unique parameter tensors finite, nonzero, updated |
| Persistent buffers | PASS | Unchanged; nonpersistent runtime counters explicitly listed |
| Synthetic fixed-batch learning | PASS | 12 updates; epsilon, ramp and slow losses recorded |
| Real fixed-batch learning | PASS | Wind/solar x positive/negative plus ordinary windows, 12 updates per stratum; diffusion objectives only |
| Train-only event sampler | PASS | Expected event probability 0.60; actual epoch draw count 168/290 |
| Future-label invariance | PASS | Fixed-seed full 500-step generation unchanged after replacing labels |
| Serialization and generation | PASS | Unit roundtrip equality plus saved real training checkpoint reloaded for 23 validation windows x 2 members |
| Training/validation/logging | PASS | 1 CPU epoch, 19 optimizer updates; train 0.1204202, validation 0.1245236 |
| Member merge/evaluation | PASS | Real 3+2 merge; prefix/route/budget and rejection tests |
| Regression tests | PASS | 62 distinct tests across full suite (61) and final model suite (6, one additional test) |
| Shell syntax and whitespace | PASS | Both Bash scripts; git diff --check |
| CUDA/FP16 GradScaler | NOT RUN | Must match target training path; CPU does not substitute |
| Target performance/memory probe | NOT RUN | Required before paid training |
| Separate event count/onset/duration/depth heads | NOT APPLICABLE | Full diffusion model has no such heads |
| Frozen body parameters within new expert | NOT APPLICABLE | All new expert parameters trainable; historical files preserved externally |
| Slow/fast branch overlap audit | NOT APPLICABLE | No separate branches; overlapping loss scales must still be evaluated |
| Event-quality, ramp, calibration and quota ablations | NOT RUN | Required for scientific conclusions |
| Development 80+20 and formal 400+100 | NOT RUN | No paid run or formal generation launched |

**Launch eligibility: false.** Local engineering checks pass. CUDA/AMP and
performance evidence remain mandatory. Fixed-batch loss reduction and a
one-epoch smoke run do not establish sustained-event quality, generalization,
or scientific benefit. Failure of a later run must distinguish implementation
errors, learning failure and insufficient evidence; it cannot establish a
universal capacity or information limit.

## Pipeline behavior

`run_station24_independent_joint_tail_v1_development.sh` rejects an existing
pipeline directory, executes preflight, then checks explicit launch eligibility
before training. The present preflight deliberately emits `launch_eligible=false`
because target performance evidence is not implemented/executed. The pipeline
therefore stops before paid training. Completing those server gates requires
a subsequent evidence-backed implementation update, not editing a PASS flag.

After an eligible development run, the timestamped training directory is saved
in `train_run.txt`. Formal continuation reuses that checkpoint and does not
train again. `development_decision.json` must contain `status: PASS`,
`formal_launch_eligible: true`, the checkpoint SHA-256, the SHA-256 of each
`development_tail_n20` and `development_mixture_body80_tail20_n100`
`generation_metadata.json` (keys end in `_metadata_sha256`), and a nonempty
`evidence` explanation. This file is not produced by the local smoke test.

A future development decision must report non-degenerate wind/solar scenarios,
physical projection checks, sustained-event and 1/3/6 h ramp results, ordinary
CRPS/coverage/width/Energy, and all/body/tail metrics. Numeric acceptance
thresholds must be agreed and recorded before interpreting the gate; no
threshold has been relaxed or invented in this local implementation review.
