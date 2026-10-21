# Shandong91 Route A/B low-cost gate report

## Scope and isolation

- Branch: `experiment/shandong91-route-ab-low-cost`
- Base commit: `a185474` (V2 Solar-nonnegative baseline)
- V3/V3.1 common-factor code and checkpoints are not imported.
- V2 data, checkpoint and 500-member outputs are read-only; new outputs must use route-specific directories.
- This report covers CPU Gate 0 only. CUDA/AMP Gate 1, 5-epoch throughput, 30-epoch short training and paired 2-window × 64-member evaluation are **NOT RUN** locally.

## Route A: one system Load coordinate

- Wind/Solar retain the V2 node-level denoiser coordinates.
- Load is collapsed to one normalized system residual per forecast time.
- One shared Gaussian Load noise coordinate is expanded to 77 nodes using fixed train-only shares; Load supervision is counted once per valid time, not 77 times.
- Shares were recovered from 7,296 complete unique training hours only.
- Active Load nodes: 77; coefficient sum: `1.0000000084`.
- Train-only system scale: q99.5 of system actual = `120458.5547 MW`.
- Fixed-share reconstruction: max absolute node error `0.004883 MW`, RMSE `0.000293 MW`, maximum error relative to system peak `3.79e-8`.
- Contract SHA-256: `306ba700b1aa55589d0a4dbf2b1582528e380636b2b96a02fefe290c169940d8`.
- Interpretation is limited to the mathematical structure of the current dataset; it does not establish that real node loads are physically fixed shares.

## Route B: V2 multi-scale physical-graph propagation

- V2 forecast/calendar/recent-error/state conditions, FiLM, epsilon target, 500-step linear schedule, mask and physical adjacency are retained.
- The only architectural change is fixed-graph propagation after every encoder and decoder temporal scale.
- No second graph, PCA factor, Tail, event loss, resource-balanced loss or other mechanism is introduced.

## Gate 0 results

| Check | Route A | Route B |
|---|---|---|
| Real data contract/load mapping | PASS | PASS |
| Forward shape and finite values | PASS | PASS |
| Backward finite/non-zero gradient | PASS | PASS |
| New graph parameters receive gradient | N/A | PASS |
| Optimizer update | PASS | PASS |
| Checkpoint save and epoch-boundary resume | PASS | PASS |
| V2 causal/mask/inverse-normalization regressions | PASS | PASS |
| CUDA/AMP | NOT RUN | NOT RUN |
| 5-epoch throughput | NOT RUN | NOT RUN |
| 30-epoch short train | NOT RUN | NOT RUN |
| Paired Gate 2 metrics | NOT RUN | NOT RUN |

CPU one-batch smoke timings (not an epoch estimate): Route A first pass `1.04 s`, resumed pass `0.56 s`; Route B first pass `1.03 s`, resumed pass `0.80 s`.

## Launch eligibility

Both routes are eligible only for the target-server CUDA/AMP Gate 1. Neither route is yet eligible for full training or 500-member generation. Run A before B if resources are limited. A 30-epoch run is allowed only after its 5-epoch probe records finite loss, finite gradients, checkpoint resume, real epoch time and peak VRAM.

