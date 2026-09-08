# Experiment engineering rules

Before changing or launching a deep-learning experiment, read
`docs/deep_learning_experiment_checks.md` and follow its evidence gates.
These rules apply to any assistant or human preparing Station-24 experiments.

For local CPU checks on this Windows workspace, reuse the isolated Conda
environment `dm_preflight` at
`C:\Users\mila2\miniconda3\envs\dm_preflight`. Do not install experiment
packages into `base` or `paperread`. If the environment is missing, recreate it
according to `docs/deep_learning_experiment_checks.md`. Local CPU success never
replaces the target-server CUDA/AMP gate.

- Preserve historical checkpoints, configurations, and formal baselines.
- Version any change that alters checkpoint inference semantics; absent fields
  must retain historical behavior. Never silently reinterpret old checkpoints.
- Passing shape tests, finite loss, or a generation smoke test does not establish
  learning or scientific benefit. Verify gradients AND optimizer updates for
  every new head; verify frozen parameters and buffers remain unchanged.
- Audit slow/fast separately and measure their overlap/cancellation. Do not call
  moving-average decomposition orthogonal or claim causal benefit without ablation.
- Record unexecuted checks as NOT RUN. CPU tests do not establish CUDA/AMP safety.
- Stop the paid pipeline before training when mandatory checks fail. Do not
  bypass checks, relax event thresholds, or launch long generation to hide failure.
- Research conclusions must distinguish implementation defects, learning failure,
  and insufficient evidence. No universal impossibility claim from one failed run.
- Do not launch paid training merely because this document exists. The experiment
  manifest must enumerate passed checks, unresolved issues, and launch eligibility.
