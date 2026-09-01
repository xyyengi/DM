#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOG_ROOT="logs/station24"
OUTPUT_ROOT="outputs_shandong/station24"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

if [[ "${STATION24_TAIL_RATIO_SWEEP_WORKER:-0}" != "1" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_ROOT}/station24_tail_ratio_sweep_${stamp}.log"
  pidfile="${LOG_ROOT}/station24_tail_ratio_sweep_${stamp}.pid"
  status="${LOG_ROOT}/station24_tail_ratio_sweep_${stamp}.status"
  nohup env STATION24_TAIL_RATIO_SWEEP_WORKER=1 JOB_STAMP="${stamp}" \
    bash "$0" >"${log}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${pidfile}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" >"${status}"
  echo "Started Station24 Raw body-tail routing-ratio sweep"
  echo "PID=${pid}"
  echo "LOG=${log}"
  echo "STATUS=${status}"
  echo "Monitor: tail -f '${log}'"
  exit 0
fi

stamp="${JOB_STAMP:?missing JOB_STAMP}"
status="${LOG_ROOT}/station24_tail_ratio_sweep_${stamp}.status"
trap 'code=$?; if [[ $code -eq 0 ]]; then state=completed; else state=failed; fi; printf "state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n" "$state" "$$" "$(date --iso-8601=seconds)" "$code" >"${status}"' EXIT

PYTHON_BIN="${PYTHON_BIN:-python}"
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the 23-window x 500-member sweep")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

mapfile -t runs < <(find "${OUTPUT_ROOT}" -path '*/training/*_station24_body_tail_moe_*_seed2027' -type d | sort)
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "ERROR: Raw body-tail training run was not found under ${OUTPUT_ROOT}"
  exit 1
fi
RUN_DIR="${runs[-1]}"
[[ -f "${RUN_DIR}/checkpoints/model_best.pt" ]] || { echo "missing model_best.pt"; exit 1; }

BASELINE_RESULT=""
while IFS= read -r metadata; do
  candidate="$(dirname "${metadata}")"
  if "${PYTHON_BIN}" - "${metadata}" "${RUN_DIR}" <<'PY'
import json, sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text())
ok = (
    meta.get("split") == "val"
    and int(meta.get("n_samples", 0)) == 500
    and int(meta.get("generation_seed", -1)) == 424242
    and meta.get("checkpoint_state_source") == "raw"
    and meta.get("trained_condition_variant") == "geo_history_actual_body_tail_moe"
    and Path(meta.get("checkpoint", "")).resolve()
        == (Path(sys.argv[2]) / "checkpoints/model_best.pt").resolve()
)
raise SystemExit(0 if ok else 1)
PY
  then
    BASELINE_RESULT="${candidate}"
    break
  fi
done < <(find "${OUTPUT_ROOT}" -path '*/validation_results/*' -name generation_metadata.json -type f | sort -r)

PIPELINE_ROOT="${OUTPUT_ROOT}/tail_ratio_sweep_${stamp}"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
DIAGNOSTIC_ROOT="${PIPELINE_ROOT}/sustained_drop_coverage"
mkdir -p "${RESULT_ROOT}"

if [[ -z "${BASELINE_RESULT}" ]]; then
  BASELINE_RESULT="${RESULT_ROOT}/raw_body_tail_baseline_val_n500_seed424242"
  echo "BASELINE_GENERATION_START"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${RUN_DIR}" \
    --data-path diffusion_input_station \
    --output-dir "${BASELINE_RESULT}" \
    --split val --n-samples 500 --seed 424242 \
    --checkpoint-state raw \
    --result-variant raw_body_tail_baseline \
    --energy-score-member-limit 80
else
  echo "BASELINE_REUSED=${BASELINE_RESULT}"
fi

declare -A ratios=( [tail15]=0.15 [tail20]=0.20 [tail30]=0.30 )
for label in tail15 tail20 tail30; do
  ratio="${ratios[$label]}"
  result="${RESULT_ROOT}/raw_body_tail_${label}_val_n500_seed424242"
  echo "GENERATION_START label=${label} tail_probability=${ratio}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${RUN_DIR}" \
    --data-path diffusion_input_station \
    --output-dir "${result}" \
    --split val --n-samples 500 --seed 424242 \
    --checkpoint-state raw \
    --result-variant "raw_body_tail_${label}" \
    --tail-route-probability "${ratio}" \
    --energy-score-member-limit 80
done

echo "SUSTAINED_DROP_DIAGNOSTIC_START"
"${PYTHON_BIN}" tools/diagnose_station24_sustained_drop_tail_sweep.py \
  --run-dir "${RUN_DIR}" \
  --data-path diffusion_input_station \
  --output-dir "${DIAGNOSTIC_ROOT}" \
  --result "baseline=${BASELINE_RESULT}" \
  --result "tail15=${RESULT_ROOT}/raw_body_tail_tail15_val_n500_seed424242" \
  --result "tail20=${RESULT_ROOT}/raw_body_tail_tail20_val_n500_seed424242" \
  --result "tail30=${RESULT_ROOT}/raw_body_tail_tail30_val_n500_seed424242" \
  --top-events 5

ARCHIVE="${OUTPUT_ROOT}/station24_tail_ratio_sweep_${stamp}.tar.gz"
tar -czf "${ARCHIVE}" \
  -C "${PIPELINE_ROOT}" sustained_drop_coverage \
  -C "${PIPELINE_ROOT}" \
  validation_results/raw_body_tail_tail15_val_n500_seed424242/generation_metadata.json \
  validation_results/raw_body_tail_tail15_val_n500_seed424242/metrics.json \
  validation_results/raw_body_tail_tail15_val_n500_seed424242/tail_expert_route.npy \
  validation_results/raw_body_tail_tail20_val_n500_seed424242/generation_metadata.json \
  validation_results/raw_body_tail_tail20_val_n500_seed424242/metrics.json \
  validation_results/raw_body_tail_tail20_val_n500_seed424242/tail_expert_route.npy \
  validation_results/raw_body_tail_tail30_val_n500_seed424242/generation_metadata.json \
  validation_results/raw_body_tail_tail30_val_n500_seed424242/metrics.json \
  validation_results/raw_body_tail_tail30_val_n500_seed424242/tail_expert_route.npy
sha256sum "${ARCHIVE}" >"${ARCHIVE}.sha256"

echo "TAIL_RATIO_SWEEP_COMPLETE"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "RUN_DIR=${RUN_DIR}"
echo "BASELINE_RESULT=${BASELINE_RESULT}"
echo "ARCHIVE=${ARCHIVE}"
