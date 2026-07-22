# V5 Stage-1 paired server workflow

This workflow compares:

1. `v4_legacy` reproducible V4-RS baseline;
2. `v5_t`, using a three-channel noisy state plus the correct diffusion timestep;
3. `v5_tf`, additionally using forecast/calendar/relative-position sequence FiLM.

The scripts never fall back to full CPU training. They require the
`experiment/v5-risk-conditioned-film` branch, a clean tracked worktree,
`nvidia-smi`, and `torch.cuda.is_available() == True`.

## Defaults shared by all three models

- data: `diffusion_npy_normalized`;
- output root: `outputs_shandong/v5_stage1`;
- training seed: `2026`;
- fixed validation-noise seed: `314159`;
- train batch: `64`;
- epoch budget: `150`;
- early-stopping patience: `15`;
- top checkpoints: `3`;
- diffusion steps: `500`, linear schedule;
- reverse variance: `posterior`;
- validation ensemble: `20`;
- validation generation seed: `424242`;
- validation split only; test is not used for selection.

`BATCH`, `GEN_BATCH`, `DATA`, `OUTPUTS_DIR`, and `PYTHON_BIN` can be set in the
environment. Any batch-size change must be applied identically to all three
training scripts.

## Launch training

After pulling the committed V5 branch and activating the GPU Conda environment:

```bash
bash run_v4rs_repro_stage1_server.sh
bash run_v5_t_stage1_server.sh
bash run_v5_tf_stage1_server.sh
```

Each command launches one `nohup` worker and immediately prints its unique log,
PID file, status file, monitoring command, and safe stop command. Run the three
training jobs sequentially unless separate GPUs have been deliberately assigned.

The worker records branch, full commit, `git status`, Python, PyTorch, CUDA,
`nvidia-smi`, GPU name and GPU memory. The same record is copied into the run's
`logs/server_environment.txt`; duration and protocol fields are written to
`logs/server_run_record.env`.

Typical monitoring:

```bash
tail -f logs/v5_stage1/<job>.log
cat logs/v5_stage1/<job>.status
ps -fp "$(cat logs/v5_stage1/<job>.pid)"
nvidia-smi
```

Stop only the selected job:

```bash
kill "$(cat logs/v5_stage1/<job>.pid)"
```

## Top-3 validation

After all three training status files report `state=completed`, copy the three
`RUN_DIR=` values from their logs and launch one validation job:

```bash
bash run_v5_stage1_validation.sh \
  outputs_shandong/v5_stage1/<v4rs-run> \
  outputs_shandong/v5_stage1/<v5-t-run> \
  outputs_shandong/v5_stage1/<v5-tf-run>
```

The script generates the top three checkpoints for each model on `val`, always
with posterior variance, 20 members, and seed 424242. For V5-TF rank 1 it also
generates forecast-zero and calendar-zero inference ablations. The original
forecast remains unchanged for residual-to-actual reconstruction.

Completed validation outputs are never overwritten. If the validation worker
stops after completing some result directories, reuse only complete artifacts:

```bash
RESUME_VALIDATION=1 bash run_v5_stage1_validation.sh \
  outputs_shandong/v5_stage1/<v4rs-run> \
  outputs_shandong/v5_stage1/<v5-t-run> \
  outputs_shandong/v5_stage1/<v5-tf-run>
```

An incomplete existing directory still causes a fail-fast stop and must be
inspected rather than silently reused or deleted.

## Comparison outputs

Validation automatically calls `tools/compare_v5_stage1_results.py` and writes
a unique comparison directory under:

```text
outputs_shandong/v5_stage1/comparisons/<timestamp>/
```

It contains CSV, JSON, and Markdown comparisons covering validation epsilon
MSE, CRPS, multivariate Energy Score, 80/90/95% calibration and widths, ACF,
1h/6h ramps, net-load error, cross-variable correlation, raw physical-boundary
violations, parameter count, training time, generation time, and condition
ablations. The tool deliberately does not auto-select a checkpoint.

## Preserve and download

Preserve all three training run directories, their top-3 checkpoints and
manifests, the validation result directories, comparison directory, launcher
logs/PID/status files, `config_used.yaml`, `generation_config_used.yaml`,
`validation_metadata.json`, `metrics.json`, and environment/commit records.
Do not commit data, outputs, checkpoints, or logs.
