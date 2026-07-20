#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-diffusion_npy_normalized}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong}
CONFIG=${CONFIG:-configs/v4rs_residual_standardized_no_guidance_168h.yaml}
EXP_NAME=${EXP_NAME:-v4rs_residual_standardized_no_guidance_168h}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-15}
BATCH=${BATCH:-64}
NSAMPLES=${NSAMPLES:-50}
GEN_BATCH=${GEN_BATCH:-4}
SEED=${SEED:-2026}
VARIANCE_TYPES=${VARIANCE_TYPES:-posterior}

python train.py \
  --config "${CONFIG}" \
  --data_path "${DATA}" \
  --save_path "${OUTPUTS_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch_size "${BATCH}" \
  --exp_name "${EXP_NAME}"

run_id=$(find "${OUTPUTS_DIR}" -maxdepth 1 -type d -name "*_${EXP_NAME}" -printf '%T@ %f\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)

if [[ -z "${run_id}" ]]; then
  echo "Could not locate the new V4-RS run" >&2
  exit 1
fi

echo "Generating variance type(s) [${VARIANCE_TYPES}] from ${run_id}"
for variance_type in ${VARIANCE_TYPES}; do
  if [[ "${variance_type}" != "beta" && "${variance_type}" != "posterior" ]]; then
    echo "Unsupported reverse variance type: ${variance_type}" >&2
    exit 1
  fi
  result_dir="${OUTPUTS_DIR}/${run_id}_${variance_type}_n${NSAMPLES}_seed${SEED}"
  python generate.py \
    --save_path "${OUTPUTS_DIR}" \
    --exp_name "${run_id}" \
    --data_path "${DATA}" \
    --n_samples "${NSAMPLES}" \
    --batch_size "${GEN_BATCH}" \
    --seed "${SEED}" \
    --reverse_variance_type "${variance_type}" \
    --output_dir "${result_dir}"
done

echo "V4-RS finished: ${run_id}"
