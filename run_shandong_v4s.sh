#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-diffusion_npy_normalized}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong}
EPOCHS=${EPOCHS:-150}
PATIENCE=${PATIENCE:-15}
BATCH=${BATCH:-64}
EVENT_FRACTION=${EVENT_FRACTION:-0.20}
MAX_DRAWS_PER_EVENT=${MAX_DRAWS_PER_EVENT:-8}
NSAMPLES=${NSAMPLES:-50}
GEN_BATCH=${GEN_BATCH:-4}
EXP_NAME=${EXP_NAME:-v4s_residual_event_sampler_no_guidance_168h}
CONFIG=${CONFIG:-configs/v4s_residual_event_sampler_no_guidance_168h.yaml}

echo "V4-s sampler-only experiment"
echo "data=${DATA}, outputs=${OUTPUTS_DIR}, epochs=${EPOCHS}, patience=${PATIENCE}"
echo "batch=${BATCH}, targeted_event_fraction=${EVENT_FRACTION}, max_draws_per_event=${MAX_DRAWS_PER_EVENT}"
echo "generation n_samples=${NSAMPLES}, generation_batch=${GEN_BATCH}"

mkdir -p logs
python -c 'import torch, yaml; print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available()); print("PyYAML: OK")'
python tools/audit_v4s_sampler.py \
  --data_path "${DATA}" \
  --batch_size "${BATCH}" \
  --event_fraction "${EVENT_FRACTION}" \
  --max_draws_per_event "${MAX_DRAWS_PER_EVENT}" \
  > logs/v4s_sampler_preflight.json
echo "Sampler preflight: logs/v4s_sampler_preflight.json"

python train.py \
  --config "${CONFIG}" \
  --data_path "${DATA}" \
  --save_path "${OUTPUTS_DIR}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch_size "${BATCH}" \
  --event_fraction "${EVENT_FRACTION}" \
  --max_draws_per_event "${MAX_DRAWS_PER_EVENT}" \
  --exp_name "${EXP_NAME}"

run_id=$(find "${OUTPUTS_DIR}" -maxdepth 1 -type d -name "*_${EXP_NAME}" -printf '%T@ %f\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "${run_id}" ]]; then
  echo "Cannot find completed V4-s run directory" >&2
  exit 1
fi

echo "Generating saved scenarios for ${run_id}"
python generate.py \
  --save_path "${OUTPUTS_DIR}" \
  --exp_name "${run_id}" \
  --data_path "${DATA}" \
  --n_samples "${NSAMPLES}" \
  --batch_size "${GEN_BATCH}"

python src/eval/collect_experiments.py --outputs_dir "${OUTPUTS_DIR}"
echo "V4-s finished: ${OUTPUTS_DIR}/${run_id}"
