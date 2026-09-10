# Station-24 independent joint tail V2 manifest

## Scientific change

V1 already used one full trainable denoiser for all 13 wind and 11 solar stations, but its 1/3/6 h and 12/24 h auxiliary losses averaged ordinary and extreme locations together. V2 preserves the Raw initialization, causal conditions, dual fixed graphs, state/FiLM pathway, 500 diffusion steps, data split and final 400/100 member budget. It replaces the two globally averaged auxiliaries with:

- top-10% target-ramp plus continuous-event-context supervision at 1/3/6 h;
- event-support station depth/shape plus wind, solar and renewable-system aggregate supervision;
- event-and-recovery-context slow supervision at 12/24 h.

The continuous event labels are train/validation targets only. They are never passed to generation. V2 adds no classifier, event gate, third expert, retrieval condition, oracle hint or hard time mask.

## Evidence gates before paid training

| Gate | Status | Evidence |
|---|---|---|
| Historical checkpoint semantics preserved | PASS | New configuration fields default to zero; nonpersistent capacity buffers do not enter checkpoints. |
| V1 baseline files modified | PASS | V1 configuration and historical results are unchanged. |
| Python syntax | PASS | `py_compile` on model, train, generate, preflight, merge, plot and summary tools. |
| Unit/regression tests | PASS | 42 tests passed locally, including V2 nonzero losses, full-model gradients and causal generation invariance. |
| Real-data CPU preflight | PASS | Raw checkpoint loaded 387 tensors; 181 trainable parameter tensors received finite nonzero gradients; optimizer changed 362 tensors; fixed buffers unchanged; event-stratified fixed-batch objectives decreased; save/reload and full-chain causal invariance passed. |
| Postprocessing smoke | PASS | Existing 23x500 validation results completed wind/solar/renewable metrics, same-member correlation, 3x3 joint plot and Markdown summary. |
| Bash syntax on local Windows | NOT RUN | No Git Bash or installed WSL is available locally. Server invokes the script with Bash. |
| Target CUDA/AMP, memory and timing | NOT RUN | Mandatory server preflight; the paid pipeline stops before training unless it passes. |
| Formal V2 event/ordinary quality | NOT RUN | Requires training and 23x500 validation generation. |

Local CPU PASS does not establish CUDA/AMP safety or scientific benefit. The server preflight is launch-eligible only after CUDA/AMP gradients, optimizer updates, real event strata, save/reload, causal invariance, two full reverse chains, peak memory and elapsed time all complete.

## Primary decision after the run

V2 succeeds only if 1/3 h extreme-ramp coverage or error and continuous-event depth/duration improve without a material loss in Raw-body ordinary CRPS, Energy Score, renewable aggregate calibration or wind-solar correlation. More tail members alone are not treated as an improvement.
