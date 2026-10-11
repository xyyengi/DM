# Shandong91 V3 failure and V3.1 correction

## Decision

V3 is not a usable probabilistic generator.  Its formal node-space objective
attenuated errors in standardized PCA-factor coordinates by the squared energy
of the fixed normalized loadings.  V3.1 therefore keeps the data, PCA factors,
V2 backbone, legal conditions, graph, diffusion schedule and sampler unchanged,
but versions the training objective and checkpoint identifier.

## Evidence from the completed V3 run

- The normalized loading energies were Wind `0.52058`, Solar `0.38035`, and
  Load `0.030566`.  Consequently, a unit Load-factor error contributed only a
  small node-space error to the V3 formal loss.
- At diffusion step 499 the EMA checkpoint predicted Load-factor epsilon with
  RMS `0.698` for a target RMS near `0.981`.  The prediction was correlated
  with the target, so this was amplitude under-learning rather than a completely
  disconnected branch.
- With the same initial noise, the reverse-chain Load-factor RMS grew from
  `0.981` to `24.648` under DDPM500.  DDIM50 ended at `34.377`; reducing the
  number of denoising steps did not repair the learned score scale.
- The completed V3 result had Load system 90% width `149028 MW`, Load system
  CRPS `17616 MW`, and joint Energy Score `263345 MW`, all severe regressions
  from V2.  Coverage equal to one was therefore over-dispersion, not improved
  calibration.

## V3.1 objective

The formal objective is

`mean(error^2 over the 7 standardized PCA-factor coordinates)`

plus

`effective-mask mean(error^2 over the projected Wind/Solar local field)`.

This is coordinate-block supervision, not resource-balanced node weighting.
The factor dimensionalities remain the train-only evidence-based values
Wind=5, Solar=1, Load=1.  Load retains no local branch.  The local field is
still projected away from the fixed PCA subspaces, so common variance is not
double-counted.

## Gates

The paid launcher runs a full-model CUDA/AMP preflight before training.  The
gate verifies the exact objective, finite forward/backward, finite non-zero
Wind/Solar/Load factor gradients, absence of PCA-energy gradient attenuation,
fixed-noise loss decrease after an optimizer update, mask invariance, backbone
update, checkpoint reload, sampler output, causal conditions and Load rank-1
reconstruction.  Any failed assertion stops before formal training.

Local CPU checks passed, including 12 unit tests, the V3.1 preflight, a bounded
three-update train/resume probe, two-member/two-step generation, complete
evaluation and plots.  These checks establish engineering behavior only.
Full-model CUDA/AMP safety and scientific improvement are **NOT RUN** locally
and must be established on the target server and by the completed paired V2
comparison.
