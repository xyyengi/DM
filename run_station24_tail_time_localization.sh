#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-tail-time-localization}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
SOURCE_PIPELINE_ROOT=${SOURCE_PIPELINE_ROOT:-${1:-}}
RAW_REFERENCE_RESULT=${RAW_REFERENCE_RESULT:-${2:-}}
REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT:-}
RESUME_PIPELINE_ROOT=${RESUME_PIPELINE_ROOT:-${3:-}}
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
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_tail_time_localization_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_tail_time_localization_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_tail_time_localization_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_TAIL_TIME_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_PIPELINE_ROOT="${SOURCE_PIPELINE_ROOT}" \
    RAW_REFERENCE_RESULT="${RAW_REFERENCE_RESULT}" \
    REFERENCE_HISTORY_RESULT="${REFERENCE_HISTORY_RESULT}" \
    RESUME_PIPELINE_ROOT="${RESUME_PIPELINE_ROOT}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 tail-time localization pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_TAIL_TIME_INTERNAL_WORKER:-0}" != "1" ]]; then
  launch_background
  exit 0
fi

trap record_exit EXIT
cd "${REPO_ROOT}"
command -v git >/dev/null || die "git is unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v "${PYTHON_BIN}" >/dev/null || die "${PYTHON_BIN} is unavailable"
[[ "$(git branch --show-current)" == "${EXPECTED_BRANCH}" ]] \
  || die "expected branch ${EXPECTED_BRANCH}, got $(git branch --show-current)"
git diff --quiet || die "tracked working-tree changes are present; commit/pull first"
git diff --cached --quiet || die "staged changes are present; commit/pull first"
[[ "${NSAMPLES}" -eq 500 ]] || die "formal comparison requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

for required in \
  train_forecast.npy train_actual.npy train_residual.npy train_time_mark.npy \
  train_lead_mark.npy train_fill_mask.npy train_issue_dates.csv \
  val_forecast.npy val_actual.npy val_residual.npy val_time_mark.npy \
  val_lead_mark.npy val_fill_mask.npy val_issue_dates.csv \
  station_features.npy station_adjacency.npy station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "TAIL_TIME_UNIT_TEST_START"
"${PYTHON_BIN}" -m py_compile \
  src/models/station_conditioned_diffusion.py train_station24.py \
  generate_station24.py tools/audit_station24_tail_time_localization.py
"${PYTHON_BIN}" -m unittest \
  tests.test_station24_pipeline.Station24ModelTests.test_tail_time_localizer_reuses_raw_tail_and_localizes_each_member

if [[ -z "${SOURCE_PIPELINE_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_PIPELINE_ROOT}" || "${candidate}" -nt "${SOURCE_PIPELINE_ROOT}" ]]; then
      SOURCE_PIPELINE_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'body_tail_moe_20*' -print0)
fi
[[ -n "${SOURCE_PIPELINE_ROOT}" && -d "${SOURCE_PIPELINE_ROOT}" ]] \
  || die "source body-tail pipeline not found; pass argument 1"

shopt -s nullglob
source_runs=("${SOURCE_PIPELINE_ROOT}"/training/*_station24_body_tail_moe_*_seed2027)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected one source body-tail run, found ${#source_runs[@]}"
SOURCE_RUN=${source_runs[0]}
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoints/model_best.pt"
SECONDARY_ADJACENCY="${SOURCE_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "missing source Raw checkpoint ${SOURCE_CHECKPOINT}"
[[ -f "${SECONDARY_ADJACENCY}" ]] || die "missing frozen historical graph ${SECONDARY_ADJACENCY}"

if [[ -z "${RAW_REFERENCE_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${RAW_REFERENCE_RESULT}" || "${candidate}" -nt "${RAW_REFERENCE_RESULT}" ]]; then
      RAW_REFERENCE_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242' -print0)
fi
[[ -n "${RAW_REFERENCE_RESULT}" && -d "${RAW_REFERENCE_RESULT}" ]] \
  || die "previous Raw 500-member reference not found; pass argument 2"

if [[ -z "${REFERENCE_HISTORY_RESULT}" ]]; then
  REFERENCE_SEARCH_ROOT=$(dirname "${OUTPUT_ROOT}")
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_HISTORY_RESULT}" || "${candidate}" -nt "${REFERENCE_HISTORY_RESULT}" ]]; then
      REFERENCE_HISTORY_RESULT=${candidate}
    fi
  done < <(find "${REFERENCE_SEARCH_ROOT}" -type d -path '*/historical_dual_graph_*/validation_results/geo_history_actual_dual_val_n500_seed424242' -print0)
fi
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "historical-spatial 500-member reference not found"

"${PYTHON_BIN}" - "${SOURCE_CHECKPOINT}" "${RAW_REFERENCE_RESULT}" \
  "${REFERENCE_HISTORY_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path
import torch

source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if source.get("condition_variant") != "geo_history_actual_body_tail_moe":
    raise SystemExit("source checkpoint variant mismatch")
if "model_state_dict" not in source:
    raise SystemExit("source checkpoint lacks Raw model_state_dict")
for path, variant in [
    (Path(sys.argv[2]), "geo_history_actual_body_tail_moe_raw"),
    (Path(sys.argv[3]), "geo_history_actual_dual"),
]:
    metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("condition_variant") != variant:
        raise SystemExit(f"reference variant mismatch: {path}")
    expected = {"split": "val", "n_samples": int(sys.argv[4]), "generation_seed": int(sys.argv[5]), "test_used": False}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise SystemExit(f"reference protocol mismatch {key}: {path}")
print("INPUT_AUDIT_PASSED")
PY

PIPELINE_ROOT=${RESUME_PIPELINE_ROOT:-"${OUTPUT_ROOT}/tail_time_localization_${JOB_STAMP}"}
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"

shopt -s nullglob
candidate_runs=("${TRAIN_ROOT}"/*_station24_tail_time_localized_*_seed2027)
shopt -u nullglob
if [[ ${#candidate_runs[@]} -eq 1 && ! -f "${candidate_runs[0]}/checkpoints/model_best.pt" ]]; then
  mkdir -p "${PIPELINE_ROOT}/incomplete_training"
  preserved_training="${PIPELINE_ROOT}/incomplete_training/$(basename "${candidate_runs[0]}")_${JOB_STAMP}"
  mv "${candidate_runs[0]}" "${preserved_training}"
  echo "PRESERVED_INCOMPLETE_TRAINING=${preserved_training}"
  candidate_runs=()
fi
if [[ ${#candidate_runs[@]} -eq 0 ]]; then
  echo "TAIL_TIME_TRAINING_START source_state=raw"
  "${PYTHON_BIN}" train_station24.py \
    --config configs/station24_geo_history_actual_body_tail_time_localized_168h.yaml \
    --data-path "${DATA}" --output-root "${TRAIN_ROOT}" \
    --exp-name "station24_tail_time_localized_${JOB_STAMP}" \
    --secondary-adjacency "${SECONDARY_ADJACENCY}" \
    --initialize-checkpoint "${SOURCE_CHECKPOINT}"
  shopt -s nullglob
  candidate_runs=("${TRAIN_ROOT}"/*_station24_tail_time_localized_*_seed2027)
  shopt -u nullglob
else
  echo "TAIL_TIME_TRAINING_SKIP existing_run=${candidate_runs[0]}"
fi
[[ ${#candidate_runs[@]} -eq 1 ]] || die "expected one candidate run, found ${#candidate_runs[@]}"
CANDIDATE_RUN=${candidate_runs[0]}

RAW_RESULT="${RESULT_ROOT}/geo_history_actual_body_tail_time_localized_raw_val_n${NSAMPLES}_seed${GEN_SEED}"
EMA_RESULT="${RESULT_ROOT}/geo_history_actual_body_tail_time_localized_ema_val_n${NSAMPLES}_seed${GEN_SEED}"

preserve_incomplete_generation() {
  local output=$1
  if [[ -d "${output}" && ! -f "${output}/metrics.json" ]]; then
    local preserved="${output}.incomplete_${JOB_STAMP}"
    [[ ! -e "${preserved}" ]] || die "incomplete preservation target exists: ${preserved}"
    mv "${output}" "${preserved}"
    echo "PRESERVED_INCOMPLETE_GENERATION=${preserved}"
  fi
}
preserve_incomplete_generation "${RAW_RESULT}"
preserve_incomplete_generation "${EMA_RESULT}"

if [[ ! -f "${RAW_RESULT}/metrics.json" ]]; then
  echo "LOCALIZED_RAW_GENERATION_START members=${NSAMPLES}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
    --output-dir "${RAW_RESULT}" --split val --n-samples "${NSAMPLES}" \
    --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}" \
    --energy-score-member-limit "${ENERGY_MEMBERS}" \
    --checkpoint-state raw \
    --result-variant geo_history_actual_body_tail_time_localized_raw
else
  echo "LOCALIZED_RAW_GENERATION_SKIP"
fi

if [[ ! -f "${EMA_RESULT}/metrics.json" ]]; then
  echo "LOCALIZED_EMA_GENERATION_START members=${NSAMPLES}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
    --output-dir "${EMA_RESULT}" --split val --n-samples "${NSAMPLES}" \
    --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}" \
    --energy-score-member-limit "${ENERGY_MEMBERS}" \
    --checkpoint-state ema \
    --result-variant geo_history_actual_body_tail_time_localized_ema
else
  echo "LOCALIZED_EMA_GENERATION_SKIP"
fi

AUDIT_RAW="${PIPELINE_ROOT}/tail_time_audit/raw"
AUDIT_EMA="${PIPELINE_ROOT}/tail_time_audit/ema"
if [[ ! -f "${AUDIT_RAW}/tail_time_localization_audit.json" ]]; then
  echo "TAIL_TIME_RAW_AUDIT_START"
  "${PYTHON_BIN}" tools/audit_station24_tail_time_localization.py \
    --source-checkpoint "${SOURCE_CHECKPOINT}" --candidate-run "${CANDIDATE_RUN}" \
    --candidate-result "${RAW_RESULT}" --data-path "${DATA}" \
    --output-dir "${AUDIT_RAW}" --top-issues 5
fi
if [[ ! -f "${AUDIT_EMA}/tail_time_localization_audit.json" ]]; then
  echo "TAIL_TIME_EMA_AUDIT_START"
  "${PYTHON_BIN}" tools/audit_station24_tail_time_localization.py \
    --source-checkpoint "${SOURCE_CHECKPOINT}" --candidate-run "${CANDIDATE_RUN}" \
    --candidate-result "${EMA_RESULT}" --data-path "${DATA}" \
    --output-dir "${AUDIT_EMA}" --top-issues 5
fi

compare_pair() {
  local baseline=$1 candidate=$2 output=$3 baseline_variant=$4 candidate_variant=$5 baseline_label=$6 candidate_label=$7 prefix=$8 title=$9
  if [[ -f "${output}/comparison_summary.csv" ]]; then
    echo "COMPARISON_SKIP output=${output}"
    return
  fi
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${baseline}" "${candidate}" --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${baseline_variant}" --candidate-variant "${candidate_variant}" \
    --baseline-label "${baseline_label}" --candidate-label "${candidate_label}" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "${title}" --figure-prefix "${prefix}"
}

echo "COMPARISONS_START"
compare_pair "${RAW_REFERENCE_RESULT}" "${RAW_RESULT}" \
  "${PIPELINE_ROOT}/comparisons/raw_tail_vs_time_localized_raw" \
  geo_history_actual_body_tail_moe_raw geo_history_actual_body_tail_time_localized_raw \
  "Raw body-tail" "Time-localized Raw" raw_vs_localized \
  "Raw body-tail versus stochastic time-localized tail"
compare_pair "${RAW_RESULT}" "${EMA_RESULT}" \
  "${PIPELINE_ROOT}/comparisons/localized_raw_vs_ema" \
  geo_history_actual_body_tail_time_localized_raw geo_history_actual_body_tail_time_localized_ema \
  "Time-localized Raw" "Time-localized warm-up EMA" localized_raw_vs_ema \
  "Same localized checkpoint: Raw versus warm-up EMA"
compare_pair "${REFERENCE_HISTORY_RESULT}" "${RAW_RESULT}" \
  "${PIPELINE_ROOT}/comparisons/history_vs_time_localized_raw" \
  geo_history_actual_dual geo_history_actual_body_tail_time_localized_raw \
  "Historical-spatial body" "Time-localized Raw" history_vs_localized \
  "Historical-spatial body versus time-localized tail"

echo "SPECIALIZATION_AND_EVENT_DIAGNOSTICS_START"
SPECIALIZATION="${PIPELINE_ROOT}/body_tail_specialization/localized_raw"
EXTREME_TAIL="${PIPELINE_ROOT}/extreme_wind_tail/raw_vs_time_localized_raw"
TIMING="${PIPELINE_ROOT}/wind_event_timing/history_vs_time_localized_raw"
ATTRIBUTION="${PIPELINE_ROOT}/forecast_event_attribution/history_vs_time_localized_raw"
if [[ ! -f "${SPECIALIZATION}/body_tail_specialization_summary.json" ]]; then
  "${PYTHON_BIN}" tools/analyze_station24_body_tail_specialization.py \
    --run-dir "${CANDIDATE_RUN}" --result-dir "${RAW_RESULT}" \
    --data-path "${DATA}" --output-dir "${SPECIALIZATION}" --top-issues 5
fi
if [[ ! -f "${EXTREME_TAIL}/extreme_wind_tail_summary.json" ]]; then
  "${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
    --baseline "${RAW_REFERENCE_RESULT}" --candidate "${RAW_RESULT}" \
    --data-path "${DATA}" --output-dir "${EXTREME_TAIL}" \
    --top-issues 5 --baseline-label "Raw body-tail" \
    --candidate-label "Time-localized Raw"
fi
if [[ ! -f "${TIMING}/timing_diagnostics.md" ]]; then
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${REFERENCE_HISTORY_RESULT}" "${RAW_RESULT}" --data-path "${DATA}" \
    --output-dir "${TIMING}" --baseline-variant geo_history_actual_dual \
    --candidate-variant geo_history_actual_body_tail_time_localized_raw \
    --baseline-label "Historical-spatial body" \
    --candidate-label "Time-localized Raw"
fi
if [[ ! -f "${ATTRIBUTION}/forecast_event_attribution.md" ]]; then
  "${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
    "${REFERENCE_HISTORY_RESULT}" "${RAW_RESULT}" \
    --event-records "${TIMING}/event_records.csv" --data-path "${DATA}" \
    --output-dir "${ATTRIBUTION}"
fi

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_$(basename "${PIPELINE_ROOT}").tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_PIPELINE_ROOT=${SOURCE_PIPELINE_ROOT}"
  echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
  echo "RAW_REFERENCE_RESULT=${RAW_REFERENCE_RESULT}"
  echo "CANDIDATE_RUN=${CANDIDATE_RUN}"
  echo "LOCALIZED_RAW_RESULT=${RAW_RESULT}"
  echo "LOCALIZED_EMA_RESULT=${EMA_RESULT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_TAIL_TIME_LOCALIZATION_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
