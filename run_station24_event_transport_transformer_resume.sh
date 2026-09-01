#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-dm_env}"
fi
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
DATA=${DATA:-diffusion_input_station}
PIPELINE_ROOT=${PIPELINE_ROOT:-${1:-}}

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
  log_file="${LOG_ROOT}/station24_event_transport_transformer_resume_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_event_transport_transformer_resume_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_event_transport_transformer_resume_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 \
    STATION24_EVENT_TRANSPORT_RESUME_WORKER=1 RESUME_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    PYTHON_BIN="${PYTHON_BIN}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    DATA="${DATA}" PIPELINE_ROOT="${PIPELINE_ROOT}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" bash "$0" \
    > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started event-transport post-processing resume"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_EVENT_TRANSPORT_RESUME_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
echo "conda_env=${CONDA_DEFAULT_ENV:-none} python=$(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -m py_compile tools/analyze_station24_body_tail_specialization.py \
  tools/analyze_station24_lead_day_forecast_trust.py
"${PYTHON_BIN}" -m unittest tests.test_station24_body_tail_specialization

if [[ -z "${PIPELINE_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${PIPELINE_ROOT}" || "${candidate}" -nt "${PIPELINE_ROOT}" ]]; then
      PIPELINE_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
    -name 'event_transport_transformer_20*' -print0)
fi
[[ -n "${PIPELINE_ROOT}" && -d "${PIPELINE_ROOT}" ]] \
  || die "event-transport pipeline root not found; pass it as argument 1"

shopt -s nullglob
runs=("${PIPELINE_ROOT}"/training/*_station24_event_transport_transformer_*_seed2027)
results=("${PIPELINE_ROOT}"/validation_results/event_transport_transformer_raw_val_n500_seed424242)
shopt -u nullglob
[[ ${#runs[@]} -eq 1 ]] || die "expected exactly one candidate training run"
[[ ${#results[@]} -eq 1 && -f "${results[0]}/metrics.json" ]] \
  || die "completed 500-member result not found"
CANDIDATE_RUN=${runs[0]}
FORMAL_RESULT=${results[0]}

SPECIALIZATION="${PIPELINE_ROOT}/body_tail_specialization_resume_${RESUME_STAMP}"
echo "BODY_TAIL_SPECIALIZATION_RESUME_START"
"${PYTHON_BIN}" tools/analyze_station24_body_tail_specialization.py \
  --run-dir "${CANDIDATE_RUN}" --result-dir "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${SPECIALIZATION}" --top-issues 5

LEAD_DAY="${PIPELINE_ROOT}/lead_day_analysis_resume_${RESUME_STAMP}"
echo "LEAD_DAY_ANALYSIS_RESUME_START"
"${PYTHON_BIN}" tools/analyze_station24_lead_day_forecast_trust.py \
  --result-dir "${FORMAL_RESULT}" --data-path "${DATA}" --output-dir "${LEAD_DAY}"

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}")_resumed_${RESUME_STAMP}.tar.gz"
echo "ARCHIVE_START"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"

RESULT_FILE="${LOG_FILE%.log}.results.env"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "FORMAL_RESULT=${FORMAL_RESULT}"
  echo "SPECIALIZATION=${SPECIALIZATION}"
  echo "LEAD_DAY=${LEAD_DAY}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "EVENT_TRANSPORT_TRANSFORMER_RESUME_COMPLETE"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
