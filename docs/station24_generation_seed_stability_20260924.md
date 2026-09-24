# Station-24 checkpoint-frozen generation-seed stability

## Question

Does the sustained-event improvement of Lightweight Joint Tail over Raw keep the
same direction when only the reverse-diffusion generation seed changes?

This is a generation-only sensitivity study. It does not train or modify Raw,
Full Independent V2, or Lightweight Joint Tail.

## Locked protocol

- Historical seed: `424242`.
- New seeds registered before generation: `271828`, `314159`.
- Checkpoints: existing Raw, Full Independent V2, and Lightweight Joint Tail
  `model_best.pt` files; SHA-256 values are recorded by server preflight.
- Validation set: the same 23 windows.
- Raw evaluation: 500 Raw members.
- Candidate evaluation: first 400 members from the same-seed Raw pool plus 100
  same-seed Tail members.
- Checkpoint state: `raw` for all three generators.
- No self-localization, risk gate, event-hypothesis condition, new event label,
  loss change, parameter update, or hyperparameter search.
- Existing event and joint evaluation programs are reused without changes.

Each new seed therefore generates exactly 700 members per issue: Raw500, Full
V2 Tail100, and Lightweight Tail100. The two candidate mixtures share the same
400 Raw members within a seed.

## Required evidence

The report contains per-seed values and mean, sample standard deviation, minimum
and maximum for ordinary CRPS/coverage/width/Energy Score, sustained-event
any-hit and member hit ratio, onset/duration/depth, 1/3/6-hour wind and solar
ramps, overall spatial correlation RMSE, wind-solar spatial RMSE, and same-member
wind-solar power/residual correlation RMSE.

Direction consistency is evaluated pairwise within each seed. For depth, the
comparison uses absolute distance of the depth ratio from 1, rather than assuming
that a larger ratio is always better.

## Engineering gates

| Gate | Status |
|---|---|
| Local Python syntax | PASS |
| Synthetic three-seed aggregation | PASS |
| Existing merge and Lightweight regression tests | PASS |
| Historical result/checkpoint audit | PASS (CPU-only) |
| No training command in formal runner | PASS |
| Target-server shell syntax | NOT RUN |
| Target-server CUDA generation smoke | NOT RUN |
| Formal two-seed generation and evaluation | NOT RUN |

Local evidence does not replace the server gates. The formal runner stops before
the 500-member generation if the checkpoint/result audit, free-space check,
CUDA matrix probe, or three-model two-member generation smoke fails.

## Entry points

- `launch_station24_generation_seed_stability.sh`: detached one-command launch.
- `run_station24_generation_seed_stability.sh`: idempotent complete worker.
- `run_station24_generation_seed_stability_finalize.sh`: continuation using the
  same root; completed generation/evaluation stages are reused.
- `tools/audit_station24_generation_seed_stability.py`: immutable input gate.
- `tools/summarize_station24_generation_seed_stability.py`: cross-seed report.

The pipeline produces a complete archive and a much smaller report-only archive.
No SHA-256 sidecar is computed.
