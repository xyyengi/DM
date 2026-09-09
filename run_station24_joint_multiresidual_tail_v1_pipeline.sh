#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-dm_env}"
fi

EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-joint-multiresidual-tail-v1}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
SOURCE_RAW_ROOT=${SOURCE_RAW_ROOT:-${1:-}}
BASELINE_RESULT=${BASELINE_RESULT:-${2:-}}
FORMAL_MEMBERS=${FORMAL_MEMBERS:-500}
GEN_SEED=${GEN_SEED:-424242}
CONFIG=${CONFIG:-configs/station24_joint_multiresidual_tail_v1_168h.yaml}

die() { echo "ERROR: $*" >&2; exit 1; }

record_exit() {
  local code=$?
  trap - EXIT
  local state=completed
  [[ ${code} -eq 0 ]] || state=failed
  printf 'state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n' \
    "${state}" "${BASHPID}" "$(date --iso-8601=seconds)" "${code}" > "${STATUS_FILE}"
  exit "${code}"
}

launch_background() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_joint_multiresidual_tail_v1_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_joint_multiresidual_tail_v1_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_joint_multiresidual_tail_v1_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    STATION24_JMRT_WORKER=1 JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_RAW_ROOT="${SOURCE_RAW_ROOT}" BASELINE_RESULT="${BASELINE_RESULT}" \
    FORMAL_MEMBERS="${FORMAL_MEMBERS}" GEN_SEED="${GEN_SEED}" CONFIG="${CONFIG}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station-24 joint multiresidual tail V1"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_JMRT_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present; commit/pull first"
git diff --cached --quiet || die "staged changes are present; commit/pull first"
[[ "${FORMAL_MEMBERS}" -eq 500 ]] || die "formal protocol requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

echo "ENVIRONMENT_PREFLIGHT"
echo "conda_env=${CONDA_DEFAULT_ENV:-none} python=$(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -c "import torch; print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); assert torch.cuda.is_available()"

if [[ -z "${SOURCE_RAW_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_RAW_ROOT}" || "${candidate}" -nt "${SOURCE_RAW_ROOT}" ]]; then
      SOURCE_RAW_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'body_tail_moe_20*' -print0)
fi
[[ -n "${SOURCE_RAW_ROOT}" && -d "${SOURCE_RAW_ROOT}" ]] \
  || die "Raw body-tail pipeline root not found; pass it as argument 1"
shopt -s nullglob
source_runs=("${SOURCE_RAW_ROOT}"/training/*_station24_body_tail_moe_*_seed2027)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected exactly one Raw body-tail training run"
SOURCE_RUN=${source_runs[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
SECONDARY_ADJACENCY="${SOURCE_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing Raw checkpoint ${SOURCE_CHECKPOINT}"
[[ -f "${SECONDARY_ADJACENCY}" ]] || die "missing secondary graph ${SECONDARY_ADJACENCY}"

if [[ -z "${BASELINE_RESULT}" ]]; then
  candidate="${OUTPUT_ROOT}/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242"
  [[ -f "${candidate}/metrics.json" ]] && BASELINE_RESULT=${candidate}
fi
[[ -n "${BASELINE_RESULT}" && -f "${BASELINE_RESULT}/metrics.json" ]] \
  || die "Raw 500-member baseline result not found; pass it as argument 2"

PIPELINE_ROOT="${OUTPUT_ROOT}/joint_multiresidual_tail_v1_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
PREFLIGHT="${PIPELINE_ROOT}/preflight"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"

echo "JMRT_PREFLIGHT_START"
"${PYTHON_BIN}" -m py_compile train_station24.py generate_station24.py \
  src/models/station_joint_multiresidual_tail.py \
  src/models/station_conditioned_diffusion.py \
  tools/audit_station24_joint_multiresidual_preflight.py \
  tools/audit_station24_joint_multiresidual_full.py \
  tools/evaluate_station24_diffusion_ts.py \
  tools/evaluate_station24_jstd_events.py
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_joint_multiresidual_tail \
  tests.test_station24_jstd_tail \
  tests.test_station24_pipeline
"${PYTHON_BIN}" -m tools.audit_station24_joint_multiresidual_preflight \
  --output "${PREFLIGHT}_core"
"${PYTHON_BIN}" -m tools.audit_station24_joint_multiresidual_full \
  --config "${CONFIG}" --checkpoint "${SOURCE_CHECKPOINT}" \
  --secondary-adjacency "${SECONDARY_ADJACENCY}" --data-path "${DATA}" \
  --output-dir "${PREFLIGHT}" --device cuda
"${PYTHON_BIN}" -c "import json; p=json.load(open('${PREFLIGHT}/full_preflight.json')); assert p['launch_eligible'] is True, p"

echo "JMRT_TRAINING_START source=raw_body_tail_raw_state frozen_body=true trainable=joint_tail_only"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG}" --data-path "${DATA}" --output-root "${TRAIN_ROOT}" \
  --exp-name "station24_joint_multiresidual_tail_v1_${JOB_STAMP}" \
  --secondary-adjacency "${SECONDARY_ADJACENCY}" \
  --initialize-checkpoint "${SOURCE_CHECKPOINT}"

shopt -s nullglob
candidate_runs=("${TRAIN_ROOT}"/*_station24_joint_multiresidual_tail_v1_*_seed2027)
shopt -u nullglob
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one JMRT training run"
CANDIDATE_RUN=${candidate_runs[0]}
FORMAL_RESULT="${RESULT_ROOT}/joint_multiresidual_tail_v1_raw_val_n${FORMAL_MEMBERS}_seed${GEN_SEED}"

echo "JMRT_GENERATION_START members=${FORMAL_MEMBERS} fixed_tail_fraction=0.15 checkpoint_state=raw"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${FORMAL_RESULT}" --split val --n-samples "${FORMAL_MEMBERS}" \
  --seed "${GEN_SEED}" --issue-batch-size 2 --member-chunk-size 500 \
  --auto-tune-member-chunk --energy-score-member-limit 80 \
  --checkpoint-state raw --result-variant geo_history_actual_joint_multiresidual_tail_v1_raw

COMPARISON="${PIPELINE_ROOT}/comparisons/raw_body_tail_vs_joint_multiresidual_tail_v1"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
  --output-dir "${COMPARISON}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_joint_multiresidual_tail_v1_raw \
  --baseline-label "Raw body-tail" --candidate-label "Joint multiresidual tail V1" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Raw body-tail versus joint multiresidual tail V1" \
  --figure-prefix raw_body_tail_vs_joint_multiresidual_tail_v1

"${PYTHON_BIN}" tools/evaluate_station24_diffusion_ts.py \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --data "${DATA}" --output "${PIPELINE_ROOT}/joint_wind_solar_evaluation" \
  --candidate-label "Joint multiresidual tail V1"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${PIPELINE_ROOT}/continuous_event_evaluation" \
  --baseline-label "Raw body-tail" --candidate-label "Joint multiresidual tail V1"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${PIPELINE_ROOT}/extreme_wind_tail" \
  --top-issues 5 --baseline-label "Raw body-tail" \
  --candidate-label "Joint multiresidual tail V1"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${PIPELINE_ROOT}/wind_event_timing" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_joint_multiresidual_tail_v1_raw \
  --baseline-label "Raw body-tail" --candidate-label "Joint multiresidual tail V1"

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}").tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
RESULT_FILE="${LOG_FILE%.log}.results.env"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_RUN=${SOURCE_RUN}"
  echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
  echo "BASELINE_RESULT=${BASELINE_RESULT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "FORMAL_RESULT=${FORMAL_RESULT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "REPORTABLE_AS_CAUSAL_FORECAST=true"
  echo "JOINT_MULTIRESIDUAL_TAIL_V1_COMPLETE"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
