#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-diffusion_npy_normalized}
EPOCHS=${EPOCHS:-200}
PATIENCE=${PATIENCE:-20}
BATCH=${BATCH:-64}
NSAMPLES=${NSAMPLES:-20}
GEN_BATCH=${GEN_BATCH:-16}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong}
START_FROM=${START_FROM:-v3}
VMIX_GUIDANCE_SCALE=${VMIX_GUIDANCE_SCALE:-}

run_one () {
  local key="$1"
  local exp="$2"
  local config="$3"
  local -a guidance_args=()

  if [[ "${START_FROM}" == "vmix" && "${key}" == "v3" ]]; then
    echo "Skipping V3 because START_FROM=vmix"
    return
  fi

  if [[ "${key}" == "vmix" && -n "${VMIX_GUIDANCE_SCALE}" ]]; then
    guidance_args=(--guidance_scale "${VMIX_GUIDANCE_SCALE}")
  fi

  echo "Running ${key}: data=${DATA}, outputs=${OUTPUTS_DIR}"
  python train.py \
    --config "${config}" \
    --data_path "${DATA}" \
    --save_path "${OUTPUTS_DIR}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH}" \
    --exp_name "${exp}" \
    "${guidance_args[@]}"

  local run_id
  run_id=$(find "${OUTPUTS_DIR}" -maxdepth 1 -type d -name "*_${exp}" -printf '%T@ %f\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-)

  python generate.py \
    --save_path "${OUTPUTS_DIR}" \
    --exp_name "${run_id}" \
    --data_path "${DATA}" \
    --n_samples "${NSAMPLES}" \
    --batch_size "${GEN_BATCH}" \
    "${guidance_args[@]}"
}

run_one v3 shandong_v3_actual_forecast_time_encoding_168h \
  configs/v3_actual_forecast_time_encoding_168h.yaml
run_one vmix shandong_vmix_residual_forecast_concat_guidance \
  configs/v_mix_residual_forecast_concat_guidance.yaml

python src/eval/collect_experiments.py --outputs_dir "${OUTPUTS_DIR}"
echo "Finished. Summary: ${OUTPUTS_DIR}/experiment_summary.csv"
