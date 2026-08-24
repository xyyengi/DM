#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-body-tail-moe}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
SOURCE_PIPELINE_ROOT=${SOURCE_PIPELINE_ROOT:-${1:-}}
REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT:-}
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
  log_file="${LOG_ROOT}/station24_body_tail_moe_raw_regeneration_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_body_tail_moe_raw_regeneration_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_body_tail_moe_raw_regeneration_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_BODY_TAIL_RAW_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SOURCE_PIPELINE_ROOT="${SOURCE_PIPELINE_ROOT}" \
    REFERENCE_HISTORY_RESULT="${REFERENCE_HISTORY_RESULT}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 body-tail raw-checkpoint regeneration"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_BODY_TAIL_RAW_INTERNAL_WORKER:-0}" != "1" ]]; then
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

if [[ -z "${SOURCE_PIPELINE_ROOT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_PIPELINE_ROOT}" || "${candidate}" -nt "${SOURCE_PIPELINE_ROOT}" ]]; then
      SOURCE_PIPELINE_ROOT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name 'body_tail_moe_20*' -print0)
fi
[[ -n "${SOURCE_PIPELINE_ROOT}" && -d "${SOURCE_PIPELINE_ROOT}" ]] \
  || die "source body-tail pipeline not found; pass SOURCE_PIPELINE_ROOT or argument 1"

shopt -s nullglob
source_runs=("${SOURCE_PIPELINE_ROOT}"/training/*_station24_body_tail_moe_*_seed2027)
source_results=("${SOURCE_PIPELINE_ROOT}"/validation_results/geo_history_actual_body_tail_moe_val_n500_seed424242)
shopt -u nullglob
[[ ${#source_runs[@]} -eq 1 ]] || die "expected one source body-tail run, found ${#source_runs[@]}"
[[ ${#source_results[@]} -eq 1 && -d "${source_results[0]}" ]] \
  || die "source 500-member EMA result is missing"
MOE_RUN=${source_runs[0]}
EMA_RESULT=${source_results[0]}
CHECKPOINT="${MOE_RUN}/checkpoints/model_best.pt"
[[ -f "${CHECKPOINT}" ]] || die "source best checkpoint is missing: ${CHECKPOINT}"

if [[ -z "${REFERENCE_HISTORY_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_HISTORY_RESULT}" || "${candidate}" -nt "${REFERENCE_HISTORY_RESULT}" ]]; then
      REFERENCE_HISTORY_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/historical_dual_graph_*/validation_results/geo_history_actual_dual_val_n500_seed424242' -print0)
fi
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "historical-spatial 500-member result not found; set REFERENCE_HISTORY_RESULT"

readarray -t verified < <("${PYTHON_BIN}" - \
  "${MOE_RUN}" "${EMA_RESULT}" "${REFERENCE_HISTORY_RESULT}" \
  "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path
import torch

run = Path(sys.argv[1])
ema_result = Path(sys.argv[2])
reference = Path(sys.argv[3])
n = int(sys.argv[4])
seed = int(sys.argv[5])
checkpoint = torch.load(run / "checkpoints/model_best.pt", map_location="cpu", weights_only=False)
if checkpoint.get("condition_variant") != "geo_history_actual_body_tail_moe":
    raise SystemExit("source checkpoint is not the body-tail MoE experiment")
for key in ("model_state_dict", "ema_model_state_dict"):
    if key not in checkpoint:
        raise SystemExit(f"source checkpoint lacks {key}")
ema_meta = json.loads((ema_result / "generation_metadata.json").read_text(encoding="utf-8"))
if ema_meta.get("condition_variant") != "geo_history_actual_body_tail_moe":
    raise SystemExit("source EMA result variant mismatch")
if ema_meta.get("checkpoint_state_source", "ema") != "ema":
    raise SystemExit("source result was not generated from EMA weights")
for key, value in {"split": "val", "n_samples": n, "generation_seed": seed, "test_used": False}.items():
    if ema_meta.get(key) != value:
        raise SystemExit(f"source EMA protocol mismatch {key}")
reference_meta = json.loads((reference / "generation_metadata.json").read_text(encoding="utf-8"))
if reference_meta.get("condition_variant") != "geo_history_actual_dual":
    raise SystemExit("historical-spatial reference variant mismatch")
for key, value in {"split": "val", "n_samples": n, "generation_seed": seed, "test_used": False}.items():
    if reference_meta.get(key) != value:
        raise SystemExit(f"reference protocol mismatch {key}")
print(reference_meta["run_dir"])
print(checkpoint["epoch"])
PY
)
REFERENCE_HISTORY_RUN=${verified[0]}
CHECKPOINT_EPOCH=${verified[1]}

PIPELINE_ROOT="${OUTPUT_ROOT}/body_tail_moe_raw_inference_${JOB_STAMP}"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
RAW_RESULT="${RESULT_ROOT}/geo_history_actual_body_tail_moe_raw_val_n${NSAMPLES}_seed${GEN_SEED}"
BASELINE_COMPARISON="${PIPELINE_ROOT}/comparisons/history_vs_body_tail_raw"
STATE_COMPARISON="${PIPELINE_ROOT}/comparisons/ema_vs_raw_checkpoint"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing/history_vs_body_tail_raw"
ATTRIBUTION_ROOT="${PIPELINE_ROOT}/forecast_event_attribution/history_vs_body_tail_raw"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail/history_vs_body_tail_raw"
EMA_RAW_TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail/ema_vs_raw_checkpoint"
AUDIT_ROOT="${PIPELINE_ROOT}/body_tail_moe_raw_audit"
EMA_SPECIALIZATION="${PIPELINE_ROOT}/body_tail_specialization/ema_checkpoint"
RAW_SPECIALIZATION="${PIPELINE_ROOT}/body_tail_specialization/raw_checkpoint"
mkdir -p "${RESULT_ROOT}" "$(dirname "${BASELINE_COMPARISON}")" \
  "$(dirname "${TIMING_ROOT}")" "$(dirname "${ATTRIBUTION_ROOT}")" \
  "$(dirname "${TAIL_ROOT}")" "$(dirname "${EMA_SPECIALIZATION}")"

echo "RAW_GENERATION_START checkpoint_epoch=${CHECKPOINT_EPOCH} members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${MOE_RUN}" --data-path "${DATA}" \
  --output-dir "${RAW_RESULT}" --split val \
  --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
  --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_MEMBERS}" \
  --checkpoint-state raw \
  --result-variant geo_history_actual_body_tail_moe_raw

echo "RAW_BODY_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/audit_station24_body_tail_moe.py \
  --body-run "${REFERENCE_HISTORY_RUN}" --body-result "${REFERENCE_HISTORY_RESULT}" \
  --candidate-run "${MOE_RUN}" --candidate-result "${RAW_RESULT}" \
  --candidate-checkpoint-state raw --output-dir "${AUDIT_ROOT}"

echo "EMA_SPECIALIZATION_START"
"${PYTHON_BIN}" tools/analyze_station24_body_tail_specialization.py \
  --run-dir "${MOE_RUN}" --result-dir "${EMA_RESULT}" \
  --data-path "${DATA}" --output-dir "${EMA_SPECIALIZATION}" --top-issues 5
echo "RAW_SPECIALIZATION_START"
"${PYTHON_BIN}" tools/analyze_station24_body_tail_specialization.py \
  --run-dir "${MOE_RUN}" --result-dir "${RAW_RESULT}" \
  --data-path "${DATA}" --output-dir "${RAW_SPECIALIZATION}" --top-issues 5

echo "HISTORICAL_VS_RAW_COMPARISON_START"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${REFERENCE_HISTORY_RESULT}" "${RAW_RESULT}" \
  --data-path "${DATA}" --output-dir "${BASELINE_COMPARISON}" \
  --baseline-variant geo_history_actual_dual \
  --candidate-variant geo_history_actual_body_tail_moe_raw \
  --baseline-label "Historical-spatial body" \
  --candidate-label "Body-tail MoE / raw checkpoint" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Historical-spatial body versus body-tail MoE raw checkpoint" \
  --figure-prefix history_vs_body_tail_raw

echo "EMA_VS_RAW_COMPARISON_START"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${EMA_RESULT}" "${RAW_RESULT}" \
  --data-path "${DATA}" --output-dir "${STATE_COMPARISON}" \
  --baseline-variant geo_history_actual_body_tail_moe \
  --candidate-variant geo_history_actual_body_tail_moe_raw \
  --baseline-label "Body-tail MoE / EMA checkpoint" \
  --candidate-label "Body-tail MoE / raw checkpoint" \
  --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
  --title "Same checkpoint: EMA versus raw parameter state" \
  --figure-prefix ema_vs_raw_checkpoint

echo "WIND_EVENT_TIMING_START"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${REFERENCE_HISTORY_RESULT}" "${RAW_RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING_ROOT}" \
  --baseline-variant geo_history_actual_dual \
  --candidate-variant geo_history_actual_body_tail_moe_raw \
  --baseline-label "Historical-spatial body" \
  --candidate-label "Body-tail MoE / raw checkpoint"

echo "FORECAST_EVENT_ATTRIBUTION_START"
"${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
  "${REFERENCE_HISTORY_RESULT}" "${RAW_RESULT}" \
  --event-records "${TIMING_ROOT}/event_records.csv" --data-path "${DATA}" \
  --output-dir "${ATTRIBUTION_ROOT}"

echo "SUSTAINED_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${REFERENCE_HISTORY_RESULT}" --candidate "${RAW_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL_ROOT}" --top-issues 5 \
  --baseline-label "Historical-spatial body" \
  --candidate-label "Body-tail MoE / raw checkpoint"
echo "EMA_RAW_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${EMA_RESULT}" --candidate "${RAW_RESULT}" \
  --data-path "${DATA}" --output-dir "${EMA_RAW_TAIL_ROOT}" --top-issues 5 \
  --baseline-label "Body-tail MoE / EMA checkpoint" \
  --candidate-label "Body-tail MoE / raw checkpoint"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_body_tail_moe_raw_inference_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "SOURCE_PIPELINE_ROOT=${SOURCE_PIPELINE_ROOT}"
  echo "REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT}"
  echo "MOE_RUN=${MOE_RUN}"
  echo "EMA_RESULT=${EMA_RESULT}"
  echo "RAW_RESULT=${RAW_RESULT}"
  echo "BASELINE_COMPARISON=${BASELINE_COMPARISON}"
  echo "EMA_RAW_COMPARISON=${STATE_COMPARISON}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_BODY_TAIL_RAW_REGENERATION_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
