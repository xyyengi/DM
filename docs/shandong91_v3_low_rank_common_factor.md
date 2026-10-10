# Shandong91 V3 low-rank common-factor decision

## Scope and data assumptions

- Data version remains `reliable_channel_training_v2_solar_nonnegative`; V1/V2 data and results are not modified.
- The 77 active Load channels are a strict rank-1 data fact to numerical tolerance: train-only MW PCA PC1 explains `0.9999999999984581` of variance and rank-1 RMSE is `5.11e-5 MW`.
- Fixed-proportion allocation is therefore a highly credible inference, but the unavailable upstream workbook/builder prevents provenance confirmation. The 77 channels must not be described as 77 independent measured loads.
- Solar nodes 80 and 83 have exactly repeated actual, forecast, residual and masks in the earliest available arrays. Both are retained as distinct physical nodes; this is an unresolved mapping risk, not grounds for deletion or merging.

## Minimal architecture

The fixed train-only PCA factor counts are Wind=5, Solar=1 and Load=1. PCA is fitted only to unique hourly rows ending at `2025-10-31T23:00:00+08:00`; validation and test are never used or refitted. The serialized factor contract is hashed and embedded in checkpoints.

The reverse process evolves seven standardized common factors jointly with the Wind/Solar local residual field. At every diffusion step the same V2 temporal UNet receives the reconstructed node-resource state plus forecast, calendar, strictly causal 24-hour recent errors, forecast-state FiLM and the fixed physical graph. Its joint output is projected into fixed factor coordinates and the PCA-orthogonal local complement. Load has no local latent. This prevents common variance from being learned twice while preserving cross-resource interaction inside one conditioned backbone and one reverse chain.

The V2 backbone is unchanged: `777,152` trainable parameters in both V2 and V3. V3 adds fixed buffers and coordinate transforms, not trainable heads. Loss remains the unweighted element-wise `effective_mask` mean in reconstructed node space; no resource weighting, Tail, event loss, full graph or new calibration module is introduced.

## Evidence gates completed locally

- 16 data/V2/V3 unit tests: PASS.
- Full-width CPU forward/backward and finite gradients: PASS.
- Common-factor gradients: Wind `5.71e-4`, Solar `5.21e-4`, Load `1.92e-4` L2 in the full-width CPU gate.
- PCA decomposition/reconstruction max normalized error: `5.07e-7`; projected local overlap: `1.17e-6`; Load local maximum: exactly `0`.
- Optimizer core-backbone update: PASS; mask invariance: PASS.
- Save/reload plus paired two-step DDIM output maximum delta: `0`.
- Sampled centered Load rank-1 relative error: `7.04e-8`.
- One-batch bounded epoch-boundary resume: epoch/global-step `1/1` to `2/2`, PASS.
- Two-member, one-window bounded generation and the V3 evaluation entry: PASS. These untrained outputs are engineering artifacts and have no scientific meaning.
- Formal CUDA/AMP gate: NOT RUN locally. The formal launcher executes the full-width CUDA/AMP preflight first and stops before paid training on failure.

The evaluator reports strict-mask node and system CRPS, 90% coverage and width, system interval score, member-conditional covariance amplification, residual spatial correlation, temporal ACF and joint Energy Score. Cross-time truth correlation and member-conditional generated correlation are explicitly labeled as different estimands.

## Server entry

```bash
cd /root/autodl-tmp/DM
git fetch origin
git switch experiment/shandong91-v3-low-rank-common-factor
git pull --ff-only origin experiment/shandong91-v3-low-rank-common-factor
export CONFIG=configs/shandong91/raw_body_v3_low_rank_common_factor.yaml
export OUTPUT_ROOT="/root/autodl-tmp/DM/outputs_shandong/shandong91_v3_low_rank_$(date +%Y%m%d_%H%M%S)"
bash run_shandong91_raw_body_v3_low_rank_pipeline.sh
```

The configured data directory `reliable_channel_training_v2_solar_nonnegative` must already exist under the repository root. The pipeline does not overwrite an existing output root and keeps all V1, V2 and Station-24 outputs isolated.
