#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-dm_env}"
fi

EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-jstd-msep}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
PIPELINE_ROOT=${PIPELINE_ROOT:-${1:-}}
SOURCE_H1_ROOT=${SOURCE_H1_ROOT:-${2:-}}
BASELINE_RESULT=${BASELINE_RESULT:-${3:-}}
FORMAL_MEMBERS=${FORMAL_MEMBERS:-500}
GEN_SEED=${GEN_SEED:-424242}
ISSUE_BATCH=${ISSUE_BATCH:-2}
MEMBER_CHUNK=${MEMBER_CHUNK:-500}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}

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
  log_file="${LOG_ROOT}/station24_jstd_msep_finalize_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_jstd_msep_finalize_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_jstd_msep_finalize_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 STATION24_JSTD_MSEP_FINALIZE_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" PID_FILE="${pid_file}" \
    STATUS_FILE="${status_file}" PIPELINE_ROOT="${PIPELINE_ROOT}" \
    SOURCE_H1_ROOT="${SOURCE_H1_ROOT}" BASELINE_RESULT="${BASELINE_RESULT}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    FORMAL_MEMBERS="${FORMAL_MEMBERS}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started JSTD-MSEP generation/finalization"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_JSTD_MSEP_FINALIZE_WORKER:-0}" != "1" ]]; then
  [[ -n "${PIPELINE_ROOT}" ]] || die "pass the existing JSTD-MSEP pipeline root as argument 1"
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
[[ -d "${PIPELINE_ROOT}" ]] || die "missing JSTD-MSEP pipeline root ${PIPELINE_ROOT}"
[[ "${FORMAL_MEMBERS}" -eq 500 ]] || die "formal protocol requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

shopt -s nullglob
candidate_runs=("${PIPELINE_ROOT}"/training/*_station24_jstd_msep_*_seed2027)
shopt -u nullglob
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one JSTD-MSEP training run"
CANDIDATE_RUN=${candidate_runs[0]}
CHECKPOINT="${CANDIDATE_RUN}/checkpoints/model_best.pt"
[[ -f "${CHECKPOINT}" ]] || die "MSEP best checkpoint not found"
[[ -f "${CANDIDATE_RUN}/training_summary.json" ]] \
  || die "training_summary.json is absent; training did not finish"

INITIALIZATION_JSON="${CANDIDATE_RUN}/body_tail_initialization.json"
[[ -f "${INITIALIZATION_JSON}" ]] || die "missing MSEP initialization manifest"
SOURCE_CHECKPOINT=$("${PYTHON_BIN}" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
  "${INITIALIZATION_JSON}")
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing H1 source checkpoint ${SOURCE_CHECKPOINT}"

if [[ -z "${SOURCE_H1_ROOT}" ]]; then
  SOURCE_RUN=$(dirname "$(dirname "${SOURCE_CHECKPOINT}")")
  SOURCE_H1_ROOT=$(dirname "$(dirname "${SOURCE_RUN}")")
fi
H1_RESULT="${SOURCE_H1_ROOT}/validation_results/jstd_event_hypothesis_h1_oracle_raw_val_n500_seed424242"
[[ -f "${H1_RESULT}/metrics.json" ]] || die "missing H1 oracle result ${H1_RESULT}"

if [[ -z "${BASELINE_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${BASELINE_RESULT}" || "${candidate}" -nt "${BASELINE_RESULT}" ]]; then
      BASELINE_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path \
    '*/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242' -print0)
fi
[[ -n "${BASELINE_RESULT}" && -f "${BASELINE_RESULT}/metrics.json" ]] \
  || die "Raw 500-member baseline result not found; pass argument 3"

FORMAL_RESULT="${PIPELINE_ROOT}/validation_results/jstd_msep_causal_raw_val_n${FORMAL_MEMBERS}_seed${GEN_SEED}"
if [[ ! -f "${FORMAL_RESULT}/metrics.json" ]]; then
  if [[ -e "${FORMAL_RESULT}" ]]; then
    FORMAL_RESULT="${PIPELINE_ROOT}/validation_results/jstd_msep_causal_raw_val_n${FORMAL_MEMBERS}_seed${GEN_SEED}_resumed_${JOB_STAMP}"
  fi
  echo "JSTD_MSEP_GENERATION_RESUME_START output=${FORMAL_RESULT}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
    --output-dir "${FORMAL_RESULT}" --split val --n-samples "${FORMAL_MEMBERS}" \
    --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}" --auto-tune-member-chunk \
    --energy-score-member-limit "${ENERGY_MEMBERS}" --checkpoint-state raw \
    --result-variant geo_history_actual_jstd_msep_causal_raw
else
  echo "JSTD_MSEP_GENERATION_REUSED result=${FORMAL_RESULT}"
fi

POST_ROOT="${PIPELINE_ROOT}/postprocess_resume_${JOB_STAMP}"
mkdir -p "${POST_ROOT}"
COMPARISON="${POST_ROOT}/raw_body_tail_vs_jstd_msep"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
  --output-dir "${COMPARISON}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_jstd_msep_causal_raw \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-MSEP causal" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Raw body-tail versus causal JSTD-MSEP" \
  --figure-prefix raw_body_tail_vs_jstd_msep

EVENT_EVAL="${POST_ROOT}/continuous_event_evaluation"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${EVENT_EVAL}" \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-MSEP causal"

H1_EVENT_EVAL="${POST_ROOT}/h1_upper_bound_comparison"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${H1_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${H1_EVENT_EVAL}" \
  --baseline-label "H1 oracle upper bound" --candidate-label "JSTD-MSEP causal"

TAIL="${POST_ROOT}/extreme_wind_tail"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-MSEP causal"

RESULT_AUDIT="${POST_ROOT}/msep_result_audit"
"${PYTHON_BIN}" -m tools.audit_station24_jstd_msep_result \
  --raw-result "${BASELINE_RESULT}" --h1-result "${H1_RESULT}" \
  --candidate-result "${FORMAL_RESULT}" --event-eval "${EVENT_EVAL}" \
  --output-dir "${RESULT_AUDIT}"

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}")_finalized_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
echo "JSTD_MSEP_FINALIZE_COMPLETE"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "FORMAL_RESULT=${FORMAL_RESULT}"
echo "RESULT_AUDIT=${RESULT_AUDIT}"
echo "ARCHIVE=${ARCHIVE}"
echo "REPORTABLE_AS_CAUSAL_FORECAST=true"
