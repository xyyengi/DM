#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-input_4.27}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-15}
BATCH=${BATCH:-64}
NSAMPLES=${NSAMPLES:-20}
GEN_BATCH=${GEN_BATCH:-16}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs}
START_FROM=${START_FROM:-}
SHOULD_RUN=1

if [[ -n "${START_FROM}" ]]; then
  SHOULD_RUN=0
fi

run_one () {
  local exp="$1"
  local config="$2"

  if [[ "${SHOULD_RUN}" == "0" ]]; then
    if [[ "${exp}" == "${START_FROM}" ]]; then
      SHOULD_RUN=1
    else
      echo "Skipping ${exp}; waiting for START_FROM=${START_FROM}"
      return
    fi
  fi

  echo "============================================================"
  echo "Running ${exp}"
  echo "CONFIG=${config}"
  echo "DATA=${DATA}"
  echo "EPOCHS=${EPOCHS}"
  echo "PATIENCE=${PATIENCE}"
  echo "BATCH=${BATCH}"
  echo "NSAMPLES=${NSAMPLES}"
  echo "GEN_BATCH=${GEN_BATCH}"
  echo "OUTPUTS_DIR=${OUTPUTS_DIR}"
  echo "START_FROM=${START_FROM}"
  echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  python train.py \
    --config "${config}" \
    --data_path "${DATA}" \
    --save_path "${OUTPUTS_DIR}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH}" \
    --exp_name "${exp}"

  local run_id
  run_id=$(ls -td "${OUTPUTS_DIR}"/*_"${exp}" | head -n 1 | xargs basename)
  echo "RUN_ID=${run_id}"

  python generate.py \
    --save_path "${OUTPUTS_DIR}" \
    --exp_name "${run_id}" \
    --data_path "${DATA}" \
    --n_samples "${NSAMPLES}" \
    --batch_size "${GEN_BATCH}"

  python src/eval/collect_experiments.py --outputs_dir "${OUTPUTS_DIR}"

  echo "FINISH_TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "Finished ${exp}"
  echo
}

run_one v0_uncond_ddpm_actual_168h configs/v0_uncond_ddpm_actual_168h.yaml
run_one v1_2023_guidance_actual_168h configs/v1_2023_guidance_actual_168h.yaml
run_one v2_csdi_cond_actual_given_forecast_168h configs/v2_csdi_cond_actual_given_forecast_168h.yaml
run_one v_mix_residual_forecast_concat_guidance configs/v_mix_residual_forecast_concat_guidance.yaml

echo "All runs completed. Summary: ${OUTPUTS_DIR}/experiment_summary.csv"
