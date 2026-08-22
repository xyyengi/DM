#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-forecast-anchor-relaxation}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
JOB_STAMP=${JOB_STAMP:-${1:-}}
SEED=${SEED:-2027}
NSAMPLES=${NSAMPLES:-500}
GEN_SEED=${GEN_SEED:-424242}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}

die() { echo "ERROR: $*" >&2; exit 1; }

record_exit() {
  local code=$?
  trap - EXIT
  local state=completed
  [[ ${code} -eq 0 ]] || state=failed
  printf 'state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n' \
    "${state}" "${BASHPID}" "$(date --iso-8601=seconds)" "${code}" \
    > "${STATUS_FILE}"
  exit "${code}"
}

launch_background() {
  [[ -n "${JOB_STAMP}" ]] \
    || die "usage: bash $0 ORIGINAL_JOB_STAMP (example: 20260819_215504)"
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local resume_stamp log_file pid_file status_file
  resume_stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_forecast_anchor_relaxation_${JOB_STAMP}_resume_${resume_stamp}.log"
  pid_file="${LOG_ROOT}/station24_forecast_anchor_relaxation_${JOB_STAMP}_resume_${resume_stamp}.pid"
  status_file="${LOG_ROOT}/station24_forecast_anchor_relaxation_${JOB_STAMP}_resume_${resume_stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_FORECAST_ANCHOR_RESUME_INTERNAL_WORKER=1 \
    JOB_STAMP="${JOB_STAMP}" RESUME_STAMP="${resume_stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SEED="${SEED}" NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 forecast-anchor resume pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_FORECAST_ANCHOR_RESUME_INTERNAL_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"

[[ -n "${JOB_STAMP}" ]] || die "JOB_STAMP is required"
command -v git >/dev/null || die "git is unavailable"
command -v "${PYTHON_BIN}" >/dev/null || die "python is unavailable: ${PYTHON_BIN}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present"
git diff --cached --quiet || die "staged changes are present"
[[ "${NSAMPLES}" -eq 500 ]] || die "formal comparison requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

PIPELINE_ROOT="${OUTPUT_ROOT}/forecast_anchor_relaxation_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing"
ATTRIBUTION_ROOT="${PIPELINE_ROOT}/forecast_event_attribution"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail"
ENVIRONMENT_FILE="${LOG_ROOT}/station24_forecast_anchor_relaxation_${JOB_STAMP}.environment.txt"

[[ -d "${PIPELINE_ROOT}" ]] || die "pipeline root not found: ${PIPELINE_ROOT}"
[[ -f "${ENVIRONMENT_FILE}" ]] || die "environment file not found: ${ENVIRONMENT_FILE}"

mapfile -d '' CANDIDATE_RUNS < <(
  find "${TRAIN_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "*_station24_forecast_anchor_relaxation_${JOB_STAMP}_seed${SEED}" \
    -print0
)
[[ ${#CANDIDATE_RUNS[@]} -eq 1 ]] \
  || die "expected one trained candidate run, found ${#CANDIDATE_RUNS[@]}"
CANDIDATE_RUN=${CANDIDATE_RUNS[0]}
[[ -f "${CANDIDATE_RUN}/checkpoints/model_best.pt" ]] \
  || die "candidate checkpoint is missing"

REFERENCE_HISTORY_RESULT=$(grep -m1 '^reference_history_result=' "${ENVIRONMENT_FILE}" | cut -d= -f2-)
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "reference historical result is missing: ${REFERENCE_HISTORY_RESULT}"

G100="${RESULT_ROOT}/forecast_anchor_g100_val_n${NSAMPLES}_seed${GEN_SEED}"
G075="${RESULT_ROOT}/forecast_anchor_g075_val_n${NSAMPLES}_seed${GEN_SEED}"
G050="${RESULT_ROOT}/forecast_anchor_g050_val_n${NSAMPLES}_seed${GEN_SEED}"

validate_result() {
  local result=$1
  local expected_scale=$2
  [[ -f "${result}/generation_metadata.json" && -f "${result}/metrics.json" ]] \
    || return 1
  "${PYTHON_BIN}" - "${result}" "${expected_scale}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_scale = float(sys.argv[2])
expected_members = int(sys.argv[3])
expected_seed = int(sys.argv[4])
metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "geo_history_actual_forecast_anchor_relaxation",
    "split": "val",
    "n_samples": expected_members,
    "generation_seed": expected_seed,
    "test_used": False,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"metadata mismatch {key}: {metadata.get(key)!r} != {value!r}")
actual_scale = float(metadata.get("forecast_guidance_scale", -1.0))
if abs(actual_scale - expected_scale) > 1e-12:
    raise SystemExit(f"guidance mismatch: {actual_scale} != {expected_scale}")
PY
}

preserve_incomplete() {
  local path=$1
  if [[ -e "${path}" ]]; then
    local backup="${path}.interrupted_${RESUME_STAMP}"
    [[ ! -e "${backup}" ]] || die "backup already exists: ${backup}"
    mv -- "${path}" "${backup}"
    echo "PRESERVED_INCOMPLETE source=${path} backup=${backup}"
  fi
}

validate_result "${G100}" 1.0 || die "completed g100 result is invalid"
validate_result "${G075}" 0.75 || die "completed g075 result is invalid"
echo "EXISTING_RESULTS_OK g100=${G100} g075=${G075}"

if validate_result "${G050}" 0.5; then
  echo "GENERATION_SKIP tag=g050 reason=already_complete"
else
  preserve_incomplete "${G050}"
  "${PYTHON_BIN}" - <<'PY'
import torch
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required to regenerate g050")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
  echo "GENERATION_START tag=g050 guidance=0.5 members=${NSAMPLES}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
    --output-dir "${G050}" --split val \
    --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
    --forecast-guidance-scale 0.5 \
    --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
    --energy-score-member-limit "${ENERGY_MEMBERS}"
  validate_result "${G050}" 0.5 || die "new g050 result failed validation"
  echo "GENERATION_RESUME_COMPLETE tag=g050"
fi

mkdir -p "${COMPARISON_ROOT}" "${TIMING_ROOT}" "${ATTRIBUTION_ROOT}" "${TAIL_ROOT}"

prepare_stage() {
  local output=$1
  local marker="${output}/.resume_complete"
  if [[ -f "${marker}" ]]; then
    echo "STAGE_SKIP output=${output} reason=already_complete"
    return 1
  fi
  preserve_incomplete "${output}"
  return 0
}

run_postprocess() {
  local tag=$1
  local scale=$2
  local result="${RESULT_ROOT}/forecast_anchor_${tag}_val_n${NSAMPLES}_seed${GEN_SEED}"
  local label="Anchor relaxation guidance ${scale}"
  local comparison="${COMPARISON_ROOT}/history_vs_${tag}"
  local timing="${TIMING_ROOT}/history_vs_${tag}"
  local attribution="${ATTRIBUTION_ROOT}/history_vs_${tag}"
  local tail_dir="${TAIL_ROOT}/history_vs_${tag}"

  validate_result "${result}" "${scale}" || die "invalid result for ${tag}"

  if prepare_stage "${comparison}"; then
    echo "COMPARISON_START tag=${tag}"
    "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
      "${REFERENCE_HISTORY_RESULT}" "${result}" \
      --data-path "${DATA}" --output-dir "${comparison}" \
      --baseline-variant geo_history_actual_dual \
      --candidate-variant geo_history_actual_forecast_anchor_relaxation \
      --baseline-label "Historical-spatial baseline" --candidate-label "${label}" \
      --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
      --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
      --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
      --title "Historical-spatial baseline versus ${label}" \
      --figure-prefix "history_vs_${tag}"
    touch "${comparison}/.resume_complete"
  fi

  if prepare_stage "${timing}"; then
    echo "WIND_EVENT_TIMING_START tag=${tag}"
    "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
      "${REFERENCE_HISTORY_RESULT}" "${result}" --data-path "${DATA}" \
      --output-dir "${timing}" \
      --baseline-variant geo_history_actual_dual \
      --candidate-variant geo_history_actual_forecast_anchor_relaxation \
      --baseline-label "Historical-spatial baseline" --candidate-label "${label}"
    touch "${timing}/.resume_complete"
  fi

  if prepare_stage "${attribution}"; then
    [[ -f "${timing}/event_records.csv" ]] \
      || die "timing event records missing for ${tag}"
    echo "FORECAST_EVENT_ATTRIBUTION_START tag=${tag}"
    "${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
      "${REFERENCE_HISTORY_RESULT}" "${result}" \
      --event-records "${timing}/event_records.csv" --data-path "${DATA}" \
      --output-dir "${attribution}"
    touch "${attribution}/.resume_complete"
  fi

  if prepare_stage "${tail_dir}"; then
    echo "SUSTAINED_TAIL_AUDIT_START tag=${tag}"
    "${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
      --baseline "${REFERENCE_HISTORY_RESULT}" --candidate "${result}" \
      --data-path "${DATA}" --output-dir "${tail_dir}" --top-issues 5 \
      --baseline-label "Historical-spatial baseline" --candidate-label "${label}"
    touch "${tail_dir}/.resume_complete"
  fi

  echo "POSTPROCESS_COMPLETE tag=${tag}"
}

run_postprocess g100 1.0
run_postprocess g075 0.75
run_postprocess g050 0.5

ARCHIVE="${OUTPUT_ROOT}/station24_forecast_anchor_relaxation_${JOB_STAMP}.tar.gz"
if [[ -e "${ARCHIVE}" ]]; then
  preserve_incomplete "${ARCHIVE}"
fi
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"

RESULT_FILE="${LOG_FILE%.log}.results.env"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "GUIDANCE_100_RESULT=${G100}"
  echo "GUIDANCE_075_RESULT=${G075}"
  echo "GUIDANCE_050_RESULT=${G050}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "FORECAST_ATTRIBUTION_DIR=${ATTRIBUTION_ROOT}"
  echo "EXTREME_TAIL_DIR=${TAIL_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_FORECAST_ANCHOR_RELAXATION_RESUME_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
