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

## Evidence not completed

- Full data/checkpoint CPU audit: **BLOCKED** because the local downloaded Raw
  result lacks `generated_stochastic_residual_standardized.npy`; no result was
  fabricated. The target server must contain the complete formal Raw artifact.
- CUDA/AMP gate: **NOT RUN** (this Windows host has CPU-only PyTorch).
- Shell syntax on Linux: **NOT RUN** (no local Bash executable).
- 24 formal training/generation/evaluation runs: **NOT RUN** until the target
  CUDA/AMP gate passes.
- 64-channel boundary extension: **NOT RUN**, conditional on first-round data.
- Training-seed/generation-seed repeats: **NOT RUN**, conditional on selection.
- Final selection: **NOT RUN**; `FINAL_SELECTION.md` must not be populated before
  the registered evidence exists.

## Time and storage planning

The closest historical Lightweight formal run used roughly 45 minutes for
training on its prior GPU. Generation, merge and the complete evaluation suite
have previously added roughly 20–60 minutes depending on CPU/RAM and I/O.
Therefore the defensible planning range is about **26–42 sequential GPU/server
hours for 24 candidates**, before conditional repeats; this is an estimate, not
a measured promise for a new GPU. Candidate logs record actual wall time and the
first completed candidate should be used to revise only the operational ETA,
never the frozen scientific protocol. Allow substantial output storage; archive
and remove/retrieve completed candidate arrays progressively rather than filling
the server disk.
