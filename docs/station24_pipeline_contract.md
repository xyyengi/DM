# Station-24 formal pipeline contract

Every paid Station-24 experiment must use a single top-level launcher that owns
the complete lifecycle. A training-only command is not a formal pipeline.

## Required lifecycle

1. Validate immutable inputs and activate `dm_env` explicitly.
2. Run the target-server CUDA/AMP preflight and stop on any FAIL or NOT RUN gate.
3. Train under `<pipeline-root>/training/` and persist `<pipeline-root>/train_run.txt`.
4. Discover the run from `checkpoints/model_best.pt`; verify the checkpoint exists.
5. Generate the declared member budget with fixed split, seed, and checkpoint state.
6. Merge body/tail members when applicable and validate metadata/member counts.
7. Run ordinary, event, timing, extreme-tail, and joint wind-solar evaluations.
8. Produce representative plots and a Markdown result summary.
9. Archive the complete pipeline root only after all required artifacts exist.

## Operational guarantees

- The launcher must use `set -Eeuo pipefail`, unbuffered Python, one unified log,
  and a machine-readable status file containing `state`, `phase`, `root`, and log.
- An ERR trap must record the failing phase and nonzero exit code.
- Every output path must live below one pipeline root. Training must use the same
  `TRAIN_ROOT` that downstream checkpoint discovery searches.
- Checkpoint discovery must convert `.../checkpoints/model_best.pt` to its parent
  run directory, not to the `checkpoints` directory.
- A separate idempotent finalize script must resume from a valid checkpoint
  without retraining. Complete generation/merge/evaluation artifacts may be reused;
  incomplete directories must never be silently overwritten.
- CLI signatures must be checked against the actual scripts before launch. Reuse
  a previously successful pipeline command sequence for shared evaluation tools.
- Completion means generation, merge, all required evaluations, plots, summary,
  and archive succeeded. `TRAIN_COMPLETE` alone is not experiment completion.
- Do not compute a SHA256 sidecar unless the experiment manifest requests it.

## Required final artifacts

- `train_run.txt` and `checkpoints/model_best.pt`
- generated tail/result arrays, metadata, and metrics
- merged ensemble arrays, metadata, and metrics when applicable
- ordinary comparison, continuous-event evaluation, joint wind-solar evaluation
- extreme-event plots, timing diagnostics, representative joint plots
- `RESULT_SUMMARY.md`
- one nonempty `.tar.gz` archive and a final `state=complete` status file

Before handing a command to the user, run shell syntax checks, Python CLI-help
checks for every called tool, and a dry-run/static path audit. Record checks that
need server data or CUDA as NOT RUN rather than assuming success.
