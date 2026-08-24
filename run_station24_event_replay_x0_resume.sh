#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-event-replay-x0}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT:-}
PIPELINE_ROOT=${PIPELINE_ROOT:-${1:-}}
NSAMPLES=${NSAMPLES:-500}
GEN_SEED=${GEN_SEED:-424242}

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
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_event_replay_x0_resume_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_event_replay_x0_resume_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_event_replay_x0_resume_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    STATION24_EVENT_REPLAY_X0_RESUME_WORKER=1 \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE_HISTORY_RESULT="${REFERENCE_HISTORY_RESULT}" \
    PIPELINE_ROOT="${PIPELINE_ROOT}" NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 B1 analysis resume"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
}

if [[ "${STATION24_EVENT_REPLAY_X0_RESUME_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present; commit/pull first"
git diff --cached --quiet || die "staged changes are present; commit/pull first"

if [[ -z "${PIPELINE_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${PIPELINE_ROOT}" || "${candidate}" -nt "${PIPELINE_ROOT}" ]]; then
      PIPELINE_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'event_replay_x0_*' -print0)
fi
[[ -n "${PIPELINE_ROOT}" && -d "${PIPELINE_ROOT}" ]] \
  || die "B1 pipeline root not found; pass it as the first argument"

B1_RESULT="${PIPELINE_ROOT}/validation_results/geo_history_actual_event_replay_x0_val_n${NSAMPLES}_seed${GEN_SEED}"
[[ -f "${B1_RESULT}/generation_metadata.json" && -f "${B1_RESULT}/metrics.json" ]] \
  || die "completed B1 generation result not found: ${B1_RESULT}"
shopt -s nullglob
run_matches=("${PIPELINE_ROOT}/training"/*_station24_event_replay_x0_*_seed2027)
shopt -u nullglob
[[ ${#run_matches[@]} -eq 1 ]] || die "expected one completed B1 training run"
B1_RUN=${run_matches[0]}
[[ -f "${B1_RUN}/checkpoints/model_best.pt" ]] || die "B1 checkpoint is missing"

if [[ -z "${REFERENCE_HISTORY_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_HISTORY_RESULT}" || "${candidate}" -nt "${REFERENCE_HISTORY_RESULT}" ]]; then
      REFERENCE_HISTORY_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/historical_dual_graph_*/validation_results/geo_history_actual_dual_val_n500_seed424242' -print0)
fi
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "historical-spatial reference result not found"

"${PYTHON_BIN}" - "${REFERENCE_HISTORY_RESULT}" "${B1_RESULT}" <<'PY'
import json
import sys
from pathlib import Path

checks = [
    (Path(sys.argv[1]), "geo_history_actual_dual"),
    (Path(sys.argv[2]), "geo_history_actual_event_replay_x0"),
]
for path, variant in checks:
    meta = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "condition_variant": variant,
        "split": "val",
        "n_samples": 500,
        "generation_seed": 424242,
        "test_used": False,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise SystemExit(f"metadata mismatch {path} {key}: {meta.get(key)!r}")
print("RESUME_INPUT_AUDIT_PASSED")
PY

COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons/history_vs_event_replay_x0"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing/history_vs_event_replay_x0"
ATTRIBUTION_ROOT="${PIPELINE_ROOT}/forecast_event_attribution/history_vs_event_replay_x0"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail/history_vs_event_replay_x0"

prepare_output() {
  local output=$1 marker=$2
  if [[ -f "${output}/${marker}" ]]; then
    return 1
  fi
  if [[ -d "${output}" ]]; then
    rmdir "${output}" 2>/dev/null \
      || die "incomplete non-empty output requires manual review: ${output}"
  fi
  mkdir -p "$(dirname "${output}")"
  return 0
}

if prepare_output "${COMPARISON_ROOT}" comparison_summary.csv; then
  echo "COMPARISON_RESUME_START"
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${REFERENCE_HISTORY_RESULT}" "${B1_RESULT}" \
    --data-path "${DATA}" --output-dir "${COMPARISON_ROOT}" \
    --baseline-variant geo_history_actual_dual \
    --candidate-variant geo_history_actual_event_replay_x0 \
    --baseline-label "Historical-spatial baseline" \
    --candidate-label "Independent event replay + x0" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "Historical-spatial baseline versus B1 event learning" \
    --figure-prefix history_vs_event_replay_x0
else
  echo "COMPARISON_ALREADY_COMPLETE"
fi

if prepare_output "${TIMING_ROOT}" event_records.csv; then
  echo "WIND_EVENT_TIMING_RESUME_START"
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${REFERENCE_HISTORY_RESULT}" "${B1_RESULT}" --data-path "${DATA}" \
    --output-dir "${TIMING_ROOT}" \
    --baseline-variant geo_history_actual_dual \
    --candidate-variant geo_history_actual_event_replay_x0 \
    --baseline-label "Historical-spatial baseline" \
    --candidate-label "Independent event replay + x0"
else
  echo "WIND_EVENT_TIMING_ALREADY_COMPLETE"
fi

if prepare_output "${ATTRIBUTION_ROOT}" attribution_summary.csv; then
  echo "FORECAST_EVENT_ATTRIBUTION_RESUME_START"
  "${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
    "${REFERENCE_HISTORY_RESULT}" "${B1_RESULT}" \
    --event-records "${TIMING_ROOT}/event_records.csv" --data-path "${DATA}" \
    --output-dir "${ATTRIBUTION_ROOT}"
else
  echo "FORECAST_EVENT_ATTRIBUTION_ALREADY_COMPLETE"
fi

if prepare_output "${TAIL_ROOT}" extreme_wind_tail_summary.json; then
  echo "SUSTAINED_TAIL_AUDIT_RESUME_START"
  "${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
    --baseline "${REFERENCE_HISTORY_RESULT}" --candidate "${B1_RESULT}" \
    --data-path "${DATA}" --output-dir "${TAIL_ROOT}" --top-issues 5 \
    --baseline-label "Historical-spatial baseline" \
    --candidate-label "Independent event replay + x0"
else
  echo "SUSTAINED_TAIL_AUDIT_ALREADY_COMPLETE"
fi

JOB_NAME=$(basename "${PIPELINE_ROOT}")
ARCHIVE="${OUTPUT_ROOT}/station24_${JOB_NAME}.tar.gz"
[[ ! -e "${ARCHIVE}" ]] || die "refusing to overwrite archive ${ARCHIVE}"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "${JOB_NAME}"
echo "B1_RESUME_COMPLETE"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "B1_RUN=${B1_RUN}"
echo "B1_RESULT=${B1_RESULT}"
echo "ARCHIVE=${ARCHIVE}"
