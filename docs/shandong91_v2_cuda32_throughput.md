# Shandong91 V2 CUDA throughput gate

The original Solar-nonnegative formal configuration used microbatch 2,
gradient accumulation 8, two DataLoader workers, and generation member chunk
2.  Those conservative values were never throughput-selected on the target
32 GiB vGPU.  The interrupted server run completed only two epochs and its
latest recoverable checkpoint may be older because the trainer previously
saved `last.pt` only on validation epochs.

This change does not alter the model, data, diffusion schedule, loss, optimizer,
effective batch size, validation rule, or evaluation mask.  It adds a bounded
target-GPU engineering gate that:

- compares training microbatches 2/4/8/16 while holding effective batch 16;
- compares DataLoader worker counts 2/4/8;
- compares generation member chunks 2/4/8/16/32/64;
- rejects OOM candidates and candidates reserving more than 90% of device RAM;
- selects the measured highest-throughput eligible candidate;
- writes the selected values into an isolated `resolved_config.yaml`;
- runs the existing full-model CUDA/AMP preflight before formal training.

The probe performs one timed effective-batch optimizer update per training
candidate after a single-microbatch warmup.  Generation timing measures repeated
full denoiser calls, which are the operation repeated at every reverse diffusion
step.  This is a short hardware-selection probe, not a learning experiment.

The trainer now atomically writes `last.pt` after every completed epoch.
`best.pt` is still updated only after the existing EMA validation gate.  Resume
also trims history rows newer than the checkpoint and rejects changes to batch,
optimizer, AMP, clipping, or EMA semantics.

## Server launch

Use a new output directory; do not resume the batch-2 checkpoint with the
autotuned config.

```bash
export OUTPUT_ROOT=/root/autodl-tmp/shandong91_v2_solar_nonnegative_opt_$(date +%Y%m%d_%H%M%S)
bash run_shandong91_raw_body_v2_solar_nonnegative_autotuned.sh
```

The tuning evidence is stored at `${OUTPUT_ROOT}_throughput_tuning`.  The formal
run manifest contains the resolved configuration.  Training, 500-member DDPM
generation, evaluation, and archive remain one fail-closed pipeline.

## Evidence status

- Local CPU unit and bounded resume checks: PASS.
- Target CUDA/AMP throughput selection: NOT RUN locally.
- Target full-model CUDA/AMP preflight: NOT RUN locally.
- Formal optimized training/generation/evaluation: NOT RUN.
