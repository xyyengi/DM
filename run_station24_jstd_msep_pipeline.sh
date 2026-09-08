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
SOURCE_H1_ROOT=${SOURCE_H1_ROOT:-${1:-}}
BASELINE_RESULT=${BASELINE_RESULT:-${2:-}}
FORMAL_MEMBERS=${FORMAL_MEMBERS:-500}
GEN_SEED=${GEN_SEED:-424242}
ISSUE_BATCH=${ISSUE_BATCH:-2}
MEMBER_CHUNK=${MEMBER_CHUNK:-500}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}
MSEP_CONFIG=${MSEP_CONFIG:-configs/station24_jstd_msep_168h.yaml}
MSEP_RUN_FAMILY=${MSEP_RUN_FAMILY:-jstd_msep}
MSEP_RESULT_VARIANT=${MSEP_RESULT_VARIANT:-geo_history_actual_jstd_msep_causal_raw}
MSEP_RESULT_LABEL=${MSEP_RESULT_LABEL:-JSTD-MSEP causal}

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
  log_file="${LOG_ROOT}/station24_jstd_msep_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_jstd_msep_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_jstd_msep_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    STATION24_JSTD_MSEP_WORKER=1 JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_H1_ROOT="${SOURCE_H1_ROOT}" BASELINE_RESULT="${BASELINE_RESULT}" \
    FORMAL_MEMBERS="${FORMAL_MEMBERS}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" MSEP_CONFIG="${MSEP_CONFIG}" \
    MSEP_RUN_FAMILY="${MSEP_RUN_FAMILY}" \
    MSEP_RESULT_VARIANT="${MSEP_RESULT_VARIANT}" \
    MSEP_RESULT_LABEL="${MSEP_RESULT_LABEL}" \
    CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station-24 JSTD-MSEP causal experiment"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_JSTD_MSEP_WORKER:-0}" != "1" ]]; then
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

if [[ -z "${SOURCE_H1_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_H1_ROOT}" || "${candidate}" -nt "${SOURCE_H1_ROOT}" ]]; then
      SOURCE_H1_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'jstd_event_hypothesis_h1_20*' -print0)
fi
[[ -n "${SOURCE_H1_ROOT}" && -d "${SOURCE_H1_ROOT}" ]] \
  || die "H1 pipeline root not found; pass argument 1"
shopt -s nullglob
source_runs=("${SOURCE_H1_ROOT}"/training/*_station24_jstd_event_hypothesis_h1_*_seed2027)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected exactly one H1 training run"
SOURCE_RUN=${source_runs[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
SECONDARY_ADJACENCY="${SOURCE_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing H1 checkpoint ${SOURCE_CHECKPOINT}"
[[ -f "${SECONDARY_ADJACENCY}" ]] || die "missing H1 graph ${SECONDARY_ADJACENCY}"
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
  || die "Raw 500-member baseline result not found; pass argument 2"

PIPELINE_ROOT="${OUTPUT_ROOT}/${MSEP_RUN_FAMILY}_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
PREFLIGHT="${PIPELINE_ROOT}/preflight"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"

echo "JSTD_MSEP_PREFLIGHT_START"
"${PYTHON_BIN}" -m py_compile station_jstd_targets.py station_dataset.py \
  train_station24.py generate_station24.py \
  src/models/station_joint_decomposed_tail.py \
  src/models/station_conditioned_diffusion.py \
  tools/audit_station24_jstd_msep_preflight.py \
  tools/audit_station24_jstd_msep_result.py \
  tools/evaluate_station24_jstd_events.py
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_jstd_targets tests.test_station24_jstd_tail \
  tests.test_station24_jstd_h1 tests.test_station24_jstd_msep
"${PYTHON_BIN}" -m tools.audit_station24_jstd_msep_preflight \
  --config "${MSEP_CONFIG}" \
  --checkpoint "${SOURCE_CHECKPOINT}" --data-path "${DATA}" \
  --output-dir "${PREFLIGHT}"

echo "JSTD_MSEP_TRAINING_START source_state=h1_raw causal_segment_prior=true"
"${PYTHON_BIN}" train_station24.py \
  --config "${MSEP_CONFIG}" \
  --data-path "${DATA}" --output-root "${TRAIN_ROOT}" \
  --exp-name "station24_${MSEP_RUN_FAMILY}_${JOB_STAMP}" \
  --secondary-adjacency "${SECONDARY_ADJACENCY}" \
  --initialize-checkpoint "${SOURCE_CHECKPOINT}"

shopt -s nullglob
candidate_runs=("${TRAIN_ROOT}"/*_station24_${MSEP_RUN_FAMILY}_*_seed2027)
shopt -u nullglob
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one MSEP training run"
CANDIDATE_RUN=${candidate_runs[0]}
FORMAL_RESULT="${RESULT_ROOT}/${MSEP_RUN_FAMILY}_causal_raw_val_n${FORMAL_MEMBERS}_seed${GEN_SEED}"

echo "JSTD_MSEP_CAUSAL_GENERATION_START members=${FORMAL_MEMBERS} fixed_tail_fraction=0.10"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${FORMAL_RESULT}" --split val --n-samples "${FORMAL_MEMBERS}" \
  --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}" --auto-tune-member-chunk \
  --energy-score-member-limit "${ENERGY_MEMBERS}" --checkpoint-state raw \
  --result-variant "${MSEP_RESULT_VARIANT}"

COMPARISON="${PIPELINE_ROOT}/comparisons/raw_body_tail_vs_jstd_msep"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
  --output-dir "${COMPARISON}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant "${MSEP_RESULT_VARIANT}" \
  --baseline-label "Raw body-tail" --candidate-label "${MSEP_RESULT_LABEL}" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Raw body-tail versus causal JSTD-MSEP" \
  --figure-prefix raw_body_tail_vs_jstd_msep

EVENT_EVAL="${PIPELINE_ROOT}/continuous_event_evaluation"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${EVENT_EVAL}" \
  --baseline-label "Raw body-tail" --candidate-label "${MSEP_RESULT_LABEL}"

H1_EVENT_EVAL="${PIPELINE_ROOT}/h1_upper_bound_comparison"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${H1_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${H1_EVENT_EVAL}" \
  --baseline-label "H1 oracle upper bound" --candidate-label "${MSEP_RESULT_LABEL}"

TAIL="${PIPELINE_ROOT}/extreme_wind_tail/raw_body_tail_vs_jstd_msep"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "${MSEP_RESULT_LABEL}"

RESULT_AUDIT="${PIPELINE_ROOT}/msep_result_audit"
"${PYTHON_BIN}" -m tools.audit_station24_jstd_msep_result \
  --raw-result "${BASELINE_RESULT}" --h1-result "${H1_RESULT}" \
  --candidate-result "${FORMAL_RESULT}" --event-eval "${EVENT_EVAL}" \
  --output-dir "${RESULT_AUDIT}"

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}").tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
RESULT_FILE="${LOG_FILE%.log}.results.env"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_RUN=${SOURCE_RUN}"
  echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
  echo "BASELINE_RESULT=${BASELINE_RESULT}"
  echo "H1_RESULT=${H1_RESULT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "FORMAL_RESULT=${FORMAL_RESULT}"
  echo "EVENT_EVAL=${EVENT_EVAL}"
  echo "H1_EVENT_EVAL=${H1_EVENT_EVAL}"
  echo "RESULT_AUDIT=${RESULT_AUDIT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "REPORTABLE_AS_CAUSAL_FORECAST=true"
  echo "JSTD_MSEP_COMPLETE"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
