# Station-24 Tail Capacity Search Execution Status

Updated: 2026-10-12 (Asia/Shanghai)

## Completed locally

- Baseline implementation and formal recipe audited from source/config.
- Search protocol frozen before candidate results.
- 24 configs materialized and validated.
- Exact instantiated Raw/Tail/total parameter counts recorded.
- `c24_d2_h3` reproduced 20,588 Tail parameters and historical state-dict keys.
- Structural CPU tests passed for all 24 combinations: shape, route-off identity,
  functional-branch optimizer updates, and exact Haar reconstruction.
- Existing Lightweight wiring/400+100 merger tests passed.
- Formal candidate and sequential restart-safe launchers implemented.

## Revised staged execution

- The default launcher now runs only six registered candidates.
- Each pauses at 1,520 optimizer updates (80 epochs) without compressing a
  learning-rate schedule; the recipe uses constant AdamW and has no scheduler.
- Stage-1 evaluation is 120 fixed Raw + 30 Tail, numeric-only, with no bulk plots
  or archive.
- Full-24 execution is retained but fails closed unless explicitly overridden;
  it is outside the current protocol.
- Epoch-boundary continuation was added and verified on real Station-24 data:
  interrupted vs uninterrupted two-epoch CPU runs ended with bitwise-identical
  model and EMA tensors and the same train loss/update count. CUDA resume remains
  a mandatory target-server gate.

## Evidence not completed

- Full data/checkpoint CPU audit: **BLOCKED** because the local downloaded Raw
  result lacks `generated_stochastic_residual_standardized.npy`; no result was
  fabricated. The target server must contain the complete formal Raw artifact.
- CUDA/AMP gate: **NOT RUN** (this Windows host has CPU-only PyTorch).
- Shell syntax on Linux: **NOT RUN** (no local Bash executable).
- Six Stage-1 training/generation/evaluation runs: **NOT RUN** until the target
  CUDA/AMP gate passes.
- 64-channel boundary extension: **NOT RUN**, conditional on first-round data.
- Training-seed/generation-seed repeats: **NOT RUN**, conditional on selection.
- Final selection: **NOT RUN**; `FINAL_SELECTION.md` must not be populated before
  the registered evidence exists.

## Time and storage planning

Historical evidence gives about 45 minutes for a full 240-epoch Lightweight
training; a registered 80-epoch screen is therefore initially estimated at
15–25 minutes/candidate. Tail-30 generation is estimated at 4–8 minutes and the
numeric evaluation suite at 8–25 minutes depending mostly on CPU/RAM. Thus six
sequential screens are provisionally **2.7–5.8 server hours**. These components
are separately estimated, not promised acceleration. Replace the operational
ETA with the first candidate's measured timings without changing the protocol.
