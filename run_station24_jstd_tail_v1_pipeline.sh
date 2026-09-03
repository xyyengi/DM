#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_NAME:-dm_env}"
fi

EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-jstd-tail-v1}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
SOURCE_PIPELINE_ROOT=${SOURCE_PIPELINE_ROOT:-${1:-}}
BASELINE_RESULT=${BASELINE_RESULT:-${2:-}}
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
  log_file="${LOG_ROOT}/station24_jstd_tail_v1_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_jstd_tail_v1_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_jstd_tail_v1_${stamp}.status"
  nohup setsid env PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    STATION24_JSTD_WORKER=1 JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_PIPELINE_ROOT="${SOURCE_PIPELINE_ROOT}" BASELINE_RESULT="${BASELINE_RESULT}" \
    FORMAL_MEMBERS="${FORMAL_MEMBERS}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" CONDA_ENV_NAME="${CONDA_ENV_NAME:-dm_env}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started JSTD-Tail V1 experiment"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_JSTD_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
# Executing files under tools/ makes Python place tools/ rather than the
# repository root at sys.path[0].  Export the root explicitly so both direct
# script calls and module imports work in a clean SSH/conda environment.
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

for required in train_forecast.npy train_actual.npy train_residual.npy train_fill_mask.npy \
  train_issue_dates.csv val_forecast.npy val_actual.npy val_residual.npy val_fill_mask.npy \
  val_issue_dates.csv station_features.npy station_adjacency.npy station_order.csv \
  export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

if [[ -z "${SOURCE_PIPELINE_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_PIPELINE_ROOT}" || "${candidate}" -nt "${SOURCE_PIPELINE_ROOT}" ]]; then
      SOURCE_PIPELINE_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'body_tail_moe_20*' -print0)
fi
[[ -n "${SOURCE_PIPELINE_ROOT}" && -d "${SOURCE_PIPELINE_ROOT}" ]] \
  || die "Raw body-tail pipeline not found; pass argument 1"
shopt -s nullglob
source_runs=("${SOURCE_PIPELINE_ROOT}"/training/*_station24_body_tail_moe_*_seed2027)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected exactly one Raw body-tail training run"
SOURCE_RUN=${source_runs[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
SECONDARY_ADJACENCY="${SOURCE_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing Raw checkpoint ${SOURCE_CHECKPOINT}"
[[ -f "${SECONDARY_ADJACENCY}" ]] || die "missing historical graph ${SECONDARY_ADJACENCY}"

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

PIPELINE_ROOT="${OUTPUT_ROOT}/jstd_tail_v1_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
PREFLIGHT="${PIPELINE_ROOT}/preflight"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"

echo "JSTD_PREFLIGHT_START"
"${PYTHON_BIN}" -m py_compile station_jstd_targets.py station_dataset.py \
  train_station24.py generate_station24.py \
  src/models/station_joint_decomposed_tail.py \
  src/models/station_conditioned_diffusion.py \
  tools/audit_station24_jstd_preflight.py \
  tools/audit_station24_jstd_result.py \
  tools/evaluate_station24_jstd_events.py
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_jstd_targets tests.test_station24_jstd_tail
"${PYTHON_BIN}" -m tools.audit_station24_jstd_preflight \
  --config configs/station24_jstd_tail_v1_168h.yaml \
  --checkpoint "${SOURCE_CHECKPOINT}" --data-path "${DATA}" \
  --secondary-adjacency "${SECONDARY_ADJACENCY}" --output-dir "${PREFLIGHT}"

echo "JSTD_TRAINING_START source_state=raw"
"${PYTHON_BIN}" train_station24.py \
  --config configs/station24_jstd_tail_v1_168h.yaml \
  --data-path "${DATA}" --output-root "${TRAIN_ROOT}" \
  --exp-name "station24_jstd_tail_v1_${JOB_STAMP}" \
  --secondary-adjacency "${SECONDARY_ADJACENCY}" \
  --initialize-checkpoint "${SOURCE_CHECKPOINT}"

shopt -s nullglob
candidate_runs=("${TRAIN_ROOT}"/*_station24_jstd_tail_v1_*_seed2027)
shopt -u nullglob
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected exactly one JSTD training run"
CANDIDATE_RUN=${candidate_runs[0]}
FORMAL_RESULT="${RESULT_ROOT}/jstd_tail_v1_raw_val_n${FORMAL_MEMBERS}_seed${GEN_SEED}"

echo "JSTD_GENERATION_START members=${FORMAL_MEMBERS} checkpoint=raw"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${FORMAL_RESULT}" --split val --n-samples "${FORMAL_MEMBERS}" \
  --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}" --auto-tune-member-chunk \
  --energy-score-member-limit "${ENERGY_MEMBERS}" --checkpoint-state raw \
  --result-variant geo_history_actual_jstd_tail_v1_raw

COMPARISON="${PIPELINE_ROOT}/comparisons/raw_body_tail_vs_jstd_tail_v1"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
  --output-dir "${COMPARISON}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_jstd_tail_v1_raw \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-Tail V1" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Raw body-tail versus JSTD-Tail V1" \
  --figure-prefix raw_body_tail_vs_jstd_tail_v1

EVENT_EVAL="${PIPELINE_ROOT}/jstd_continuous_event_evaluation"
"${PYTHON_BIN}" -m tools.evaluate_station24_jstd_events \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --candidate-run "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${EVENT_EVAL}"

TIMING="${PIPELINE_ROOT}/wind_event_timing/raw_body_tail_vs_jstd_tail_v1"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${BASELINE_RESULT}" "${FORMAL_RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING}" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant geo_history_actual_jstd_tail_v1_raw \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-Tail V1"

TAIL="${PIPELINE_ROOT}/extreme_wind_tail/raw_body_tail_vs_jstd_tail_v1"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${BASELINE_RESULT}" --candidate "${FORMAL_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "JSTD-Tail V1"

LEAD_DAY="${PIPELINE_ROOT}/lead_day_analysis"
"${PYTHON_BIN}" tools/analyze_station24_lead_day_forecast_trust.py \
  --result-dir "${FORMAL_RESULT}" --data-path "${DATA}" --output-dir "${LEAD_DAY}"

RESULT_AUDIT="${PIPELINE_ROOT}/jstd_result_audit"
"${PYTHON_BIN}" -m tools.audit_station24_jstd_result \
  --source-checkpoint "${SOURCE_CHECKPOINT}" --candidate-run "${CANDIDATE_RUN}" \
  --candidate-result "${FORMAL_RESULT}" --output-dir "${RESULT_AUDIT}"

ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}").tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
RESULT_FILE="${LOG_FILE%.log}.results.env"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_RUN=${SOURCE_RUN}"
  echo "BASELINE_RESULT=${BASELINE_RESULT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "FORMAL_RESULT=${FORMAL_RESULT}"
  echo "EVENT_EVAL=${EVENT_EVAL}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "JSTD_TAIL_V1_COMPLETE"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
