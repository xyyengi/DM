#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-dm_env}"
fi

PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
PIPELINE_ROOT=${PIPELINE_ROOT:-${1:-}}
BASELINE_RESULT=${BASELINE_RESULT:-${2:-}}

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
  log_file="${LOG_ROOT}/station24_jstd_tail_v1_finalize_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_jstd_tail_v1_finalize_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_jstd_tail_v1_finalize_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 STATION24_JSTD_FINALIZE_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" PID_FILE="${pid_file}" \
    STATUS_FILE="${status_file}" PIPELINE_ROOT="${PIPELINE_ROOT}" \
    BASELINE_RESULT="${BASELINE_RESULT}" PYTHON_BIN="${PYTHON_BIN}" DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" bash "$0" \
    > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started JSTD-Tail V1 finalization"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_JSTD_FINALIZE_WORKER:-0}" != "1" ]]; then
  [[ -n "${PIPELINE_ROOT}" ]] || die "pass the existing JSTD pipeline root as argument 1"
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

[[ -d "${PIPELINE_ROOT}" ]] || die "missing pipeline root ${PIPELINE_ROOT}"
shopt -s nullglob
candidate_runs=("${PIPELINE_ROOT}"/training/*_station24_jstd_tail_v1_*_seed2027)
shopt -u nullglob
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one JSTD training run"
CANDIDATE_RUN=${candidate_runs[0]}
FORMAL_RESULT="${PIPELINE_ROOT}/validation_results/jstd_tail_v1_raw_val_n500_seed424242"
[[ -f "${FORMAL_RESULT}/metrics.json" ]] || die "completed 500-member result not found"

INITIALIZATION_JSON="${CANDIDATE_RUN}/body_tail_initialization.json"
[[ -f "${INITIALIZATION_JSON}" ]] || die "missing JSTD initialization manifest"
SOURCE_CHECKPOINT=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
  "${INITIALIZATION_JSON}")
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing Raw source checkpoint ${SOURCE_CHECKPOINT}"

if [[ -z "${BASELINE_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${BASELINE_RESULT}" || "${candidate}" -nt "${BASELINE_RESULT}" ]]; then
      BASELINE_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path \
    '*/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242' -print0)
fi
[[ -n "${BASELINE_RESULT}" && -f "${BASELINE_RESULT}/metrics.json" ]] \
  || die "Raw 500-member baseline result not found; pass argument 2"

POST_ROOT="${PIPELINE_ROOT}/postprocess_resume_${JOB_STAMP}"
mkdir -p "${POST_ROOT}"
echo "JSTD_FINALIZE_START pipeline=${PIPELINE_ROOT}"

EVENT_EVAL="${POST_ROOT}/jstd_continuous_event_evaluation"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${EVENT_EVAL}"

TIMING="${POST_ROOT}/wind_event_timing"
"${PYTHON_BIN}" -m tools.diagnose_station24_wind_event_timing \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_jstd_tail_v1_raw \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-Tail V1"

TAIL="${POST_ROOT}/extreme_wind_tail"
"${PYTHON_BIN}" -m tools.plot_station24_extreme_tail \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-Tail V1"

LEAD_DAY="${POST_ROOT}/lead_day_analysis"
"${PYTHON_BIN}" -m tools.analyze_station24_lead_day_forecast_trust \
  --result-dir "${FORMAL_RESULT}" --data-path "${DATA}" --output-dir "${LEAD_DAY}"

RESULT_AUDIT="${POST_ROOT}/jstd_result_audit"
"${PYTHON_BIN}" -m tools.audit_station24_jstd_result \
  --source-checkpoint "${SOURCE_CHECKPOINT}" --candidate-run "${CANDIDATE_RUN}" \
  --candidate-result "${FORMAL_RESULT}" --output-dir "${RESULT_AUDIT}"

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}")_finalized_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
echo "JSTD_TAIL_V1_FINALIZE_COMPLETE"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "EVENT_EVAL=${EVENT_EVAL}"
echo "ARCHIVE=${ARCHIVE}"
