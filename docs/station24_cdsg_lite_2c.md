# Station24 Experiment 2C: geographic-prior hybrid dynamic graph

## Question

Experiment 2C changes only the adjacency used by the 168 h parallel spatial
branch from Experiment 2B.  It tests whether an issue-specific graph improves
wind-event timing and cross-station dependence without discarding the physical
geographic prior.

## Known-at-issue graph condition

The graph generator uses only information available when the seven-day
forecast is issued:

- the current 168 h issued forecast condition embedding;
- forecast-derived State V1 embeddings;
- the available recent-error embedding;
- static station features (wind/solar type, position/capacity features).

Future actual power and future residuals are never graph-generator inputs.

For station `i`, temporal mean and standard deviation pool the 168 h known
condition into a small dynamic embedding.  A projected static station embedding
is then added.  Symmetric scaled dot-product similarities produce a sparse
top-k dynamic graph.  The graph used for propagation is

```text
A_hybrid = (1 - sigmoid(alpha)) * A_geo
           + sigmoid(alpha) * A_dynamic
```

`alpha` starts at `-3`, so the initial dynamic share is about 4.7%.  The model
must earn a larger departure from the geographic graph during training.

## Controlled comparison

- State V1 baseline: bottleneck geographic graph only.
- Experiment 2B: baseline plus 168 h parallel branch using the fixed graph.
- Experiment 2C: exactly the 2B structure, but that parallel branch uses the
  hybrid dynamic graph.

All three use the same split, residual target, State V1 features, recent error,
FiLM, diffusion schedule, seed, 80 validation members and physical projection.

The generator saves the learned validation-average hybrid adjacency and its
between-condition standard deviation as `.npy` files for graph auditing.

## One-command server pipeline

```bash
cd /root/autodl-tmp/DM
bash run_station24_cdsg_2b_2c_pipeline.sh
```

The command immediately returns after launching one detached process.  That
process trains 2B, generates 80 validation members, trains 2C, generates 80
validation members, then runs baseline-vs-2B, baseline-vs-2C and 2B-vs-2C
comparisons plus all three wind-event timing diagnostics.  It finally creates
one `station24_cdsg_2b_2c_*.tar.gz` archive.  The sealed test split is not read.
