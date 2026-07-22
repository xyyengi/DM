#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-diffusion_npy_normalized}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong}
CONFIG=${CONFIG:-configs/v4rs_reproducible_no_guidance_168h.yaml}
SEED=${SEED:?Set SEED to the training seed, for example 2026 or 2027}
GEN_SEED=${GEN_SEED:-424242}
EXP_NAME=${EXP_NAME:-v4rs_repro_seed${SEED}_no_guidance_168h}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-15}
BATCH=${BATCH:-64}
NSAMPLES=${NSAMPLES:-20}
GEN_BATCH=${GEN_BATCH:-4}

python train.py \
  --config "${CONFIG}" \
  --data_path "${DATA}" \
  --save_path "${OUTPUTS_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch_size "${BATCH}" \
  --seed "${SEED}" \
  --exp_name "${EXP_NAME}"

run_id=$(find "${OUTPUTS_DIR}" -maxdepth 1 -type d -name "*_${EXP_NAME}" -printf '%T@ %f\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)

if [[ -z "${run_id}" ]]; then
  echo "Could not locate reproducible V4-RS run for seed ${SEED}" >&2
  exit 1
fi

result_dir="${OUTPUTS_DIR}/${run_id}_val_posterior_n${NSAMPLES}_genseed${GEN_SEED}"
python generate.py \
  --save_path "${OUTPUTS_DIR}" \
  --exp_name "${run_id}" \
  --data_path "${DATA}" \
  --split val \
  --n_samples "${NSAMPLES}" \
  --batch_size "${GEN_BATCH}" \
  --seed "${GEN_SEED}" \
  --reverse_variance_type posterior \
  --output_dir "${result_dir}"

echo "Training seed: ${SEED}"
echo "Fixed validation seed: 314159"
echo "Common generation seed: ${GEN_SEED}"
echo "Training run: ${OUTPUTS_DIR}/${run_id}"
echo "Validation result: ${result_dir}"
