#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-diffusion_npy_normalized}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong}
SOURCE_RUN=${SOURCE_RUN:-20260718_232509_v4s_residual_event_sampler_no_guidance_168h}
NSAMPLES=${NSAMPLES:-50}
GEN_BATCH=${GEN_BATCH:-4}
SEED=${SEED:-2026}

if [[ ! -f "${OUTPUTS_DIR}/${SOURCE_RUN}/checkpoints/model_best.pt" ]]; then
  echo "Missing checkpoint: ${OUTPUTS_DIR}/${SOURCE_RUN}/checkpoints/model_best.pt" >&2
  exit 1
fi

for variance_type in beta posterior; do
  result_dir="${OUTPUTS_DIR}/${SOURCE_RUN}_regen_${variance_type}_n${NSAMPLES}_seed${SEED}"
  echo "Generating ${variance_type}: ${result_dir}"
  python generate.py \
    --save_path "${OUTPUTS_DIR}" \
    --exp_name "${SOURCE_RUN}" \
    --data_path "${DATA}" \
    --n_samples "${NSAMPLES}" \
    --batch_size "${GEN_BATCH}" \
    --seed "${SEED}" \
    --reverse_variance_type "${variance_type}" \
    --output_dir "${result_dir}"
done

echo "Variance ablation finished."
echo "beta:      ${OUTPUTS_DIR}/${SOURCE_RUN}_regen_beta_n${NSAMPLES}_seed${SEED}"
echo "posterior: ${OUTPUTS_DIR}/${SOURCE_RUN}_regen_posterior_n${NSAMPLES}_seed${SEED}"
