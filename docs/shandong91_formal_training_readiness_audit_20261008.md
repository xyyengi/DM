# Shandong91 heterogeneous Raw Body formal-training readiness audit

Date: 2026-10-08  
Audited commit: `0df0c08aa1a55dab87e7f0496caa21f2ec45a5b8`  
Decision: **BLOCKED — do not start paid formal training yet**

## Scope locked by the architecture decision

- Interface remains `[B,168,91,3]`; no architectural flattening.
- Target remains normalized `actual - forecast` residual.
- Formal objective remains element-wise `effective_mask` mean.
- Legacy lead condition, External Tail, auxiliary loss, event-balanced sampling,
  self-localization, and Protected Partial remain disabled.
- Resource-balanced mean remains diagnostic only. Gate 0 showed only a 0.4282%
  difference from the formal element-wise mean, so this audit does not alter the
  objective.

## Evidence gates

| Check | Status | Evidence |
|---|---|---|
| Data contract, masks, inverse normalization | PASS | Five Shandong91 contract/Gate 0 tests passed; 20 published data checks pass. |
| Temporal split order | PASS | Train: Jan–Oct 2025; validation: Nov 2025; test: Dec 2025. Window boundaries do not cross split boundaries. |
| Architecture and forbidden-branch audit | PASS | Versioned no-lead Raw Body; 91-node physical graph; forbidden switches rejected. |
| CPU Gate 0 | PASS | Shape, gradients by resource, mask invariance, optimizer update, save/reload, finite tensors. |
| CUDA/AMP Gate 0 | PASS | RTX 3080 Ti; FP16 autocast; all Gate 0 checks passed. |
| CPU Gate 1 fixed-batch learning | PASS | Overall 1.0882→0.2518; Wind/Solar/Load all decreased; 38 regressions passed. |
| CUDA/AMP Gate 1 fixed-batch learning | PASS | Overall 1.0694→0.2440; Wind 1.0962→0.3368; Solar 0.9975→0.2441; Load 1.1007→0.1747. |
| Formal train/validation entry point | FAIL | No Shandong91 epoch trainer, validation loop, early stopping, or best-checkpoint selection exists. |
| Resume-safe checkpoint contract | FAIL | No optimizer/scaler/epoch/RNG resume checkpoint for Shandong91 exists. |
| Reverse-diffusion sampler | FAIL | `Shandong91MaskedDiffusion` currently implements training noise/error only. |
| Target-server generation smoke | NOT RUN | Cannot run until a versioned sampler exists. |
| Formal manifest and fail-closed launcher | FAIL | No immutable run manifest or launcher ties preflight, training, validation and artifacts together. |
| Formal training | NOT RUN | Correctly not started. |

## Resource objective diagnostic

The fixed Gate 0 batch contained Wind 9,653 (30.33%), Solar 9,237 (29.02%), and
Load 12,936 (40.65%) valid elements. Their objective shares were 31.09%, 27.07%,
and 41.84%. Load is the largest contributor because it has more valid elements,
but its share is close to its valid-element share. The element-wise mean was
1.0693684 and the diagnostic resource-balanced mean was 1.0647893. This is not
evidence requiring an objective change; the formal objective stays element-wise.

## Blocking rationale

Gate 1 proves optimization can occur on one fixed batch. It does not establish
train/validation learning, checkpoint selection, recoverability, or generation.
Using the Gate 1 script for a long run would discard validation and resume
semantics and therefore must not be treated as formal training.

## Single next action

Implement one versioned Shandong91 formal package containing: epoch training and
validation with per-resource metrics, EMA/raw checkpoint selection, atomic
resume checkpoints including optimizer/scaler/RNG state, a reverse-diffusion
sampler, an immutable manifest, and a fail-closed launcher. Then run its bounded
CUDA/AMP train→save→reload→generate preflight. Only a PASS from that package may
set launch eligibility to true.
