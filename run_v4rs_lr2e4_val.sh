#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-diffusion_npy_normalized}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong}
CONFIG=${CONFIG:-configs/v4rs_lr2e4_no_guidance_168h.yaml}
EXP_NAME=${EXP_NAME:-v4rs_lr2e4_no_guidance_168h}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-15}
BATCH=${BATCH:-64}
NSAMPLES=${NSAMPLES:-20}
GEN_BATCH=${GEN_BATCH:-4}
SEED=${SEED:-2026}

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
  echo "Could not locate the new V4-RS low-learning-rate run" >&2
  exit 1
fi

result_dir="${OUTPUTS_DIR}/${run_id}_val_posterior_n${NSAMPLES}_seed${SEED}"
echo "Generating validation ensemble from ${run_id}"
python generate.py \
  --save_path "${OUTPUTS_DIR}" \
  --exp_name "${run_id}" \
  --data_path "${DATA}" \
  --split val \
  --n_samples "${NSAMPLES}" \
  --batch_size "${GEN_BATCH}" \
  --seed "${SEED}" \
  --reverse_variance_type posterior \
  --output_dir "${result_dir}"

echo "Training run: ${OUTPUTS_DIR}/${run_id}"
echo "Validation result: ${result_dir}"
