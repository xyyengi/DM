#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-forecast-correction-event-learning}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT:-}
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
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_forecast_correction_event_learning_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_forecast_correction_event_learning_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_forecast_correction_event_learning_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_FORECAST_CORRECTION_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    REFERENCE_HISTORY_RESULT="${REFERENCE_HISTORY_RESULT}" SEED="${SEED}" \
    NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ENERGY_MEMBERS="${ENERGY_MEMBERS}" ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 forecast-correction event-learning pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_FORECAST_CORRECTION_INTERNAL_WORKER:-0}" != "1" ]]; then
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
  train_forecast.npy train_actual.npy train_residual.npy train_fill_mask.npy \
  train_issue_dates.csv val_forecast.npy val_actual.npy val_residual.npy \
  val_fill_mask.npy val_issue_dates.csv station_features.npy station_adjacency.npy \
  station_distance.npy station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

CONFIG_BASE=configs/station24_geo_history_actual_dual_168h.yaml
CONFIG_DIRECT=configs/station24_geo_history_actual_forecast_correction_direct_168h.yaml
CONFIG_DECOMPOSED=configs/station24_geo_history_actual_forecast_correction_decomposed_168h.yaml
for required in \
  "${CONFIG_BASE}" "${CONFIG_DIRECT}" "${CONFIG_DECOMPOSED}" \
  tools/diagnose_station24_event_predictability_tail_graph.py \
  tools/analyze_station24_forecast_correction.py; do
  [[ -f "${required}" ]] || die "missing experiment artifact ${required}"
done

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

if [[ -z "${REFERENCE_HISTORY_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_HISTORY_RESULT}" || "${candidate}" -nt "${REFERENCE_HISTORY_RESULT}" ]]; then
      REFERENCE_HISTORY_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/historical_dual_graph_*/validation_results/geo_history_actual_dual_val_n500_seed424242' -print0)
fi
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "historical actual dual-graph result not found; set REFERENCE_HISTORY_RESULT"

REFERENCE_HISTORY_RUN=$("${PYTHON_BIN}" - "${REFERENCE_HISTORY_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "geo_history_actual_dual",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
}
for key, value in expected.items():
    if meta.get(key) != value:
        raise SystemExit(f"reference mismatch {key}: {meta.get(key)!r} != {value!r}")
print(meta["run_dir"])
PY
)
SECONDARY_GRAPH="${REFERENCE_HISTORY_RUN}/graphs/secondary_adjacency.npy"
[[ -f "${REFERENCE_HISTORY_RUN}/checkpoints/model_best.pt" ]] \
  || die "reference checkpoint missing: ${REFERENCE_HISTORY_RUN}"
[[ -f "${SECONDARY_GRAPH}" ]] || die "historical graph missing: ${SECONDARY_GRAPH}"

"${PYTHON_BIN}" - "${CONFIG_BASE}" "${CONFIG_DIRECT}" "${CONFIG_DECOMPOSED}" "${DATA}" "${SECONDARY_GRAPH}" <<'PY'
import copy
import sys
import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data

base, direct, decomposed = [yaml.safe_load(open(path, encoding="utf-8")) for path in sys.argv[1:4]]
correction_keys = {
    "use_forecast_correction", "forecast_correction_mode",
    "forecast_correction_channels", "forecast_correction_use_revision",
    "forecast_correction_use_recent_error", "forecast_correction_max_abs",
    "forecast_correction_loss_weight", "forecast_correction_huber_beta",
}

def stripped(config):
    value = copy.deepcopy(config)
    value.pop("experiment")
    value["target"]["type"] = "residual"
    for key in correction_keys:
        value["model"].pop(key, None)
    return value

if stripped(base) != stripped(direct) or stripped(base) != stripped(decomposed):
    raise SystemExit("A1/A2 changed fields outside correction identity/configuration")
direct_core = copy.deepcopy(direct)
decomposed_core = copy.deepcopy(decomposed)
direct_core.pop("experiment")
decomposed_core.pop("experiment")
direct_core["model"].pop("forecast_correction_mode")
decomposed_core["model"].pop("forecast_correction_mode")
if direct_core != decomposed_core:
    raise SystemExit("A1 and A2 differ outside correction representation")

static = load_station_static_data(sys.argv[4])
secondary = torch.from_numpy(np.load(sys.argv[5]).astype(np.float32))
models = {
    name: Station24DiffusionModel(
        config["model"], static["station_features"], static["station_adjacency"],
        static["station_capacities"], secondary,
    )
    for name, config in [("baseline", base), ("direct", direct), ("decomposed", decomposed)]
}
counts = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
if not counts["baseline"] < counts["direct"] < counts["decomposed"]:
    raise SystemExit(f"unexpected correction parameter ordering: {counts}")
print(f"FORECAST_CORRECTION_PARITY_OK parameters={counts}")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/forecast_correction_event_learning_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
DIAGNOSTIC_ROOT="${PIPELINE_ROOT}/event_inventory"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail"
CORRECTION_ROOT="${PIPELINE_ROOT}/forecast_correction_audit"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" "${COMPARISON_ROOT}" \
  "${TIMING_ROOT}" "${TAIL_ROOT}"

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "reference_history_result=${REFERENCE_HISTORY_RESULT}"
  echo "reference_history_run=${REFERENCE_HISTORY_RUN}"
  echo "training_seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "test_used=false"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

echo "EVENT_INVENTORY_START train_fit=true validation_read_only=true"
"${PYTHON_BIN}" tools/diagnose_station24_event_predictability_tail_graph.py \
  --data-path "${DATA}" --output-dir "${DIAGNOSTIC_ROOT}" \
  --folds 5 --first-validation-index 122 --seed "${SEED}" --validation-check

DIRECT_NAME="station24_forecast_correction_direct_${JOB_STAMP}"
echo "TRAIN_START variant=geo_history_actual_forecast_correction_direct"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG_DIRECT}" --data-path "${DATA}" \
  --secondary-adjacency "${SECONDARY_GRAPH}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${DIRECT_NAME}" --seed "${SEED}"
shopt -s nullglob
matches=("${TRAIN_ROOT}"/*_"${DIRECT_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#matches[@]} -eq 1 ]] || die "expected one direct run, found ${#matches[@]}"
DIRECT_RUN=${matches[0]}
cp "${ENVIRONMENT_FILE}" "${DIRECT_RUN}/logs/server_environment.txt"
DIRECT_RESULT="${RESULT_ROOT}/geo_history_actual_forecast_correction_direct_val_n${NSAMPLES}_seed${GEN_SEED}"
echo "GENERATION_START variant=geo_history_actual_forecast_correction_direct members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${DIRECT_RUN}" --data-path "${DATA}" \
  --output-dir "${DIRECT_RESULT}" --split val --n-samples "${NSAMPLES}" \
  --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_MEMBERS}"

DECOMPOSED_NAME="station24_forecast_correction_decomposed_${JOB_STAMP}"
echo "TRAIN_START variant=geo_history_actual_forecast_correction_decomposed"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG_DECOMPOSED}" --data-path "${DATA}" \
  --secondary-adjacency "${SECONDARY_GRAPH}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${DECOMPOSED_NAME}" --seed "${SEED}"
shopt -s nullglob
matches=("${TRAIN_ROOT}"/*_"${DECOMPOSED_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#matches[@]} -eq 1 ]] || die "expected one decomposed run, found ${#matches[@]}"
DECOMPOSED_RUN=${matches[0]}
cp "${ENVIRONMENT_FILE}" "${DECOMPOSED_RUN}/logs/server_environment.txt"
DECOMPOSED_RESULT="${RESULT_ROOT}/geo_history_actual_forecast_correction_decomposed_val_n${NSAMPLES}_seed${GEN_SEED}"
echo "GENERATION_START variant=geo_history_actual_forecast_correction_decomposed members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${DECOMPOSED_RUN}" --data-path "${DATA}" \
  --output-dir "${DECOMPOSED_RESULT}" --split val --n-samples "${NSAMPLES}" \
  --seed "${GEN_SEED}" --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_MEMBERS}"

compare_pair() {
  local left=$1 right=$2 output=$3 left_variant=$4 right_variant=$5 left_label=$6 right_label=$7 title=$8
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${left}" "${right}" --data-path "${DATA}" --output-dir "${output}" \
    --baseline-variant "${left_variant}" --candidate-variant "${right_variant}" \
    --baseline-label "${left_label}" --candidate-label "${right_label}" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "${title}" --figure-prefix "$(basename "${output}")"
}

echo "PAIRED_COMPARISONS_START"
compare_pair "${REFERENCE_HISTORY_RESULT}" "${DIRECT_RESULT}" \
  "${COMPARISON_ROOT}/baseline_vs_direct" geo_history_actual_dual \
  geo_history_actual_forecast_correction_direct "Historical-spatial baseline" \
  "Direct correction center" "Baseline versus direct forecast correction"
compare_pair "${REFERENCE_HISTORY_RESULT}" "${DECOMPOSED_RESULT}" \
  "${COMPARISON_ROOT}/baseline_vs_decomposed" geo_history_actual_dual \
  geo_history_actual_forecast_correction_decomposed "Historical-spatial baseline" \
  "Decomposed correction center" "Baseline versus decomposed forecast correction"
compare_pair "${DIRECT_RESULT}" "${DECOMPOSED_RESULT}" \
  "${COMPARISON_ROOT}/direct_vs_decomposed" \
  geo_history_actual_forecast_correction_direct \
  geo_history_actual_forecast_correction_decomposed "Direct correction center" \
  "Decomposed correction center" "Direct versus decomposed forecast correction"

echo "FORECAST_CORRECTION_AUDIT_START"
"${PYTHON_BIN}" tools/analyze_station24_forecast_correction.py \
  --baseline "${REFERENCE_HISTORY_RESULT}" --direct "${DIRECT_RESULT}" \
  --decomposed "${DECOMPOSED_RESULT}" --data-path "${DATA}" \
  --output-dir "${CORRECTION_ROOT}" --top-events 5

for variant in direct decomposed; do
  if [[ "${variant}" == "direct" ]]; then
    candidate_result=${DIRECT_RESULT}
    candidate_variant=geo_history_actual_forecast_correction_direct
    candidate_label="Direct correction center"
  else
    candidate_result=${DECOMPOSED_RESULT}
    candidate_variant=geo_history_actual_forecast_correction_decomposed
    candidate_label="Decomposed correction center"
  fi
  echo "WIND_EVENT_TIMING_START variant=${variant}"
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${REFERENCE_HISTORY_RESULT}" "${candidate_result}" --data-path "${DATA}" \
    --output-dir "${TIMING_ROOT}/baseline_vs_${variant}" \
    --baseline-variant geo_history_actual_dual \
    --candidate-variant "${candidate_variant}" \
    --baseline-label "Historical-spatial baseline" \
    --candidate-label "${candidate_label}"
  echo "SUSTAINED_TAIL_AUDIT_START variant=${variant}"
  "${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
    --baseline "${REFERENCE_HISTORY_RESULT}" --candidate "${candidate_result}" \
    --data-path "${DATA}" --output-dir "${TAIL_ROOT}/baseline_vs_${variant}" \
    --top-issues 5 --baseline-label "Historical-spatial baseline" \
    --candidate-label "${candidate_label}"
done

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_forecast_correction_event_learning_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE_HISTORY_RUN=${REFERENCE_HISTORY_RUN}"
  echo "REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT}"
  echo "EVENT_INVENTORY_DIR=${DIAGNOSTIC_ROOT}"
  echo "DIRECT_RUN=${DIRECT_RUN}"
  echo "DIRECT_RESULT=${DIRECT_RESULT}"
  echo "DECOMPOSED_RUN=${DECOMPOSED_RUN}"
  echo "DECOMPOSED_RESULT=${DECOMPOSED_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "CORRECTION_AUDIT_DIR=${CORRECTION_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "EXTREME_TAIL_DIR=${TAIL_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_FORECAST_CORRECTION_EVENT_LEARNING_STAGE1_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
