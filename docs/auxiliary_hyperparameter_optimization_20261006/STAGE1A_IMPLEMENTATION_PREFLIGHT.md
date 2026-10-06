# Stage 1A implementation and preflight record

Date: 2026-10-06  
Scope: Auxiliary-strength Stage 1A only (`alpha=0.65/1.00/1.35`)  
Formal paid training: **NOT RUN locally**

## Frozen comparison

- `alpha=0.65`: new run; effective ramp/shape/slow = `0.117/0.091/0.065`.
- `alpha=1.00`: reuse the completed Lightweight fair control; no retraining.
- `alpha=1.35`: new run; effective ramp/shape/slow = `0.243/0.189/0.135`.
- Architecture, Raw initialization/freeze, 60% event-balanced sampling, optimizer,
  training budget, diffusion configuration, validation split, generation seed and
  `400 Raw + 100 Tail` member composition remain fixed.
- Stage 1B, test evaluation, extra alpha points and multi-seed confirmation are not
  authorized by this launcher.

## Local evidence (`dm_preflight`, CPU only)

| Gate | alpha=0.65 | alpha=1.35 |
|---|---|---|
| Config materialization: only declared alpha/weight/identity fields change | PASS | PASS |
| CPU forward/backward, six optimizer steps | PASS | PASS |
| Trainable parameter count | PASS (20,588) | PASS (20,588) |
| Raw parameters and buffers unchanged | PASS | PASS |
| Every Tail module updates | PASS | PASS |
| ramp/shape/slow each has a nonzero Tail gradient | PASS | PASS |
| Disabled Tail exactly recovers Raw | PASS | PASS |
| Save/reload and causal-generation checks | PASS | PASS |
| CUDA/AMP | NOT RUN | NOT RUN |

Latest local reports:

- `local_checks/stage1a_local_preflight_20261006_022706/preflight_065_cpu_frozen_20261006_132508/`
- `local_checks/stage1a_local_preflight_20261006_022706/preflight_135_cpu_frozen_20261006_132508/`

The targeted regression suite completed 22 tests successfully. Local CPU success
does not replace the target-server CUDA/AMP gate.

## Added paid-run gates

Before either optimizer is created, the formal launcher now requires:

1. at least 12 GiB free space;
2. shell syntax and Python CLI checks;
3. successful reads and exact `[23,500,168,24]` shapes for all five reused Raw and
   control member arrays;
4. exact equality of the 400 reused Raw members across all five arrays;
5. the exact completed alpha=1 control recipe, validation split, seed 424242,
   400+100 quota and test lock;
6. a target CUDA/AMP preflight whose every mandatory check is PASS.

The local downloaded control arrays are incomplete; the new immutable-input gate
correctly reports FAIL on them. They are not used to claim readiness. The server's
canonical artifacts must independently pass the gate before paid training starts.

## Formal lifecycle and recovery

`run_station24_auxiliary_alpha_stage1a.sh` owns immutable-input audit, config
materialization, CUDA preflight, alpha=0.65 training, alpha=1.35 training,
generation, 400+100 merge, complete ordinary/event/joint/extreme/timing evaluation,
representative plots, paired bootstrap/leave-one-event-out/Pareto selection,
summary and one complete archive.

`run_station24_auxiliary_alpha_stage1a_finalize.sh` is training-free and idempotent.
It reuses complete artifacts, preserves incomplete directories under retry names,
records the actual artifact paths, and can finish generation/evaluation/archive
after an interruption without retraining a completed checkpoint.

## Current launch status

- Local implementation and CPU evidence: **PASS**.
- Target-server immutable-input audit: **NOT RUN**.
- Target-server CUDA/AMP preflight: **NOT RUN**.
- Formal alpha=0.65 training: **NOT RUN**.
- Formal alpha=1.35 training: **NOT RUN**.
- Launch eligibility: **pending the two mandatory target-server gates**.

