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
  log_file="${LOG_ROOT}/station24_jstd_mechanism_audit_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_jstd_mechanism_audit_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_jstd_mechanism_audit_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    STATION24_JSTD_AUDIT_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    PIPELINE_ROOT="${PIPELINE_ROOT}" PYTHON_BIN="${PYTHON_BIN}" DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" bash "$0" \
    > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started JSTD mechanism audit"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_JSTD_AUDIT_WORKER:-0}" != "1" ]]; then
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
candidate_results=("${PIPELINE_ROOT}"/validation_results/jstd_tail_v1_raw_val_n500_seed424242)
shopt -u nullglob
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one JSTD training run"
[[ ${#candidate_results[@]} -eq 1 ]] || die "expected exactly one JSTD result"
RUN_DIR=${candidate_runs[0]}
RESULT_DIR=${candidate_results[0]}
[[ -f "${RUN_DIR}/checkpoints/model_best.pt" ]] || die "JSTD best checkpoint is missing"
[[ -f "${RESULT_DIR}/tail_expert_probability.npy" ]] || die "JSTD route audit is missing"

OUTPUT_DIR="${PIPELINE_ROOT}/jstd_mechanism_audit_${JOB_STAMP}"
echo "JSTD_MECHANISM_AUDIT_START pipeline=${PIPELINE_ROOT}"
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_jstd_tail \
  tests.test_station24_jstd_mechanism_audit
"${PYTHON_BIN}" -u tools/audit_station24_jstd_mechanism.py \
  --run-dir "${RUN_DIR}" \
  --result-dir "${RESULT_DIR}" \
  --data-path "${DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --checkpoint-state raw \
  --timesteps 50,150,300,450 \
  --false-positive-count 3 \
  --seed 20260904

ARCHIVE="${OUTPUT_ROOT}/station24_jstd_mechanism_audit_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"
echo "JSTD_MECHANISM_AUDIT_COMPLETE"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ARCHIVE=${ARCHIVE}"
