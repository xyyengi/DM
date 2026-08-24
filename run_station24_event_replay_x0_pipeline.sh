#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-event-replay-x0}
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
  log_file="${LOG_ROOT}/station24_event_replay_x0_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_event_replay_x0_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_event_replay_x0_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
    STATION24_EVENT_REPLAY_X0_INTERNAL_WORKER=1 JOB_STAMP="${stamp}" \
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
  echo "Started Station24 B1 event-replay x0 pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_EVENT_REPLAY_X0_INTERNAL_WORKER:-0}" != "1" ]]; then
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
[[ "${NSAMPLES}" -eq 500 ]] || die "formal B1 comparison requires 500 members"
[[ "${GEN_SEED}" -eq 424242 ]] || die "generation seed must remain 424242"

for required in \
  train_forecast.npy train_actual.npy train_residual.npy train_time_mark.npy \
  train_lead_mark.npy train_fill_mask.npy train_issue_dates.csv \
  val_forecast.npy val_actual.npy val_residual.npy val_time_mark.npy \
  val_lead_mark.npy val_fill_mask.npy val_issue_dates.csv \
  station_features.npy station_adjacency.npy station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

CONFIG_BASE=configs/station24_geo_history_actual_dual_168h.yaml
CONFIG_B1=configs/station24_geo_history_actual_event_replay_x0_168h.yaml
[[ -f "${CONFIG_BASE}" && -f "${CONFIG_B1}" ]] \
  || die "B1 configuration is missing"

"${PYTHON_BIN}" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "B1_UNIT_TEST_START"
"${PYTHON_BIN}" -m unittest discover -s tests \
  -p 'test_station24_pipeline.py' -k event_replay

if [[ -z "${REFERENCE_HISTORY_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${REFERENCE_HISTORY_RESULT}" || "${candidate}" -nt "${REFERENCE_HISTORY_RESULT}" ]]; then
      REFERENCE_HISTORY_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d -path '*/historical_dual_graph_*/validation_results/geo_history_actual_dual_val_n500_seed424242' -print0)
fi
[[ -n "${REFERENCE_HISTORY_RESULT}" && -d "${REFERENCE_HISTORY_RESULT}" ]] \
  || die "historical-spatial 500-member result not found; set REFERENCE_HISTORY_RESULT"

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
[[ -f "${SECONDARY_GRAPH}" ]] \
  || die "reference historical graph missing: ${SECONDARY_GRAPH}"

"${PYTHON_BIN}" - "${CONFIG_BASE}" "${CONFIG_B1}" "${DATA}" "${SECONDARY_GRAPH}" <<'PY'
import copy
import sys
import numpy as np
import torch
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import fit_station_event_replay, load_station_static_data

base = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
candidate = yaml.safe_load(open(sys.argv[2], encoding="utf-8"))
b1_keys = {
    "forecast_correction_mode", "forecast_correction_loss_weight",
    "use_extreme_event_weighting", "use_event_replay_x0",
    "event_replay_window_hours", "event_replay_merge_gap_hours",
    "event_replay_quantiles", "event_replay_weights",
    "wind_common_event_loss_weight", "event_x0_magnitude_loss_weight",
    "event_x0_timing_loss_weight", "event_x0_sync_loss_weight",
    "event_x0_window_hours", "event_x0_error_scale",
    "event_x0_timing_temperature",
}
base_core = copy.deepcopy(base)
candidate_core = copy.deepcopy(candidate)
base_core.pop("experiment")
candidate_core.pop("experiment")
for key in b1_keys:
    candidate_core["model"].pop(key, None)
if base_core != candidate_core:
    raise SystemExit("B1 changed fields outside identity, replay, and x0 losses")
if candidate["model"].get("forecast_correction_mode") != "none":
    raise SystemExit("B1 must not use forecast correction")
if candidate["model"].get("use_extreme_event_weighting"):
    raise SystemExit("B1 must not use legacy hourly event weighting")

spec = fit_station_event_replay(sys.argv[3], candidate["model"])
if spec["independent_event_count"] <= 0:
    raise SystemExit("B1 found no independent events")
if spec["representative_issue_count"] > spec["overlapping_issue_count"]:
    raise SystemExit("B1 replay did not deduplicate overlapping issues")
static = load_station_static_data(sys.argv[3])
secondary = torch.from_numpy(np.load(sys.argv[4]).astype(np.float32))
models = [
    Station24DiffusionModel(
        config["model"], static["station_features"], static["station_adjacency"],
        static["station_capacities"], secondary,
    )
    for config in (base, candidate)
]
counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
if counts[0] != counts[1]:
    raise SystemExit(f"B1 loss-only experiment changed parameter count: {counts}")
print(
    "B1_PARITY_OK "
    f"parameters={counts[0]} events={spec['independent_event_count']} "
    f"q90={spec['q90_event_count']} overlapping_issues={spec['overlapping_issue_count']} "
    f"expected_event_draws={spec['expected_event_draws_per_epoch']:.2f}"
)
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/event_replay_x0_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparisons/history_vs_event_replay_x0"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing/history_vs_event_replay_x0"
ATTRIBUTION_ROOT="${PIPELINE_ROOT}/forecast_event_attribution/history_vs_event_replay_x0"
TAIL_ROOT="${PIPELINE_ROOT}/extreme_wind_tail/history_vs_event_replay_x0"
REPLAY_AUDIT_ROOT="${PIPELINE_ROOT}/event_replay_audit"
# Analysis tools intentionally refuse to overwrite an existing output
# directory.  Create only their parents here; each tool creates its own leaf.
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}" \
  "$(dirname "${COMPARISON_ROOT}")" \
  "$(dirname "${TIMING_ROOT}")" \
  "$(dirname "${ATTRIBUTION_ROOT}")" \
  "$(dirname "${TAIL_ROOT}")"

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

EXP_NAME="station24_event_replay_x0_${JOB_STAMP}"
echo "TRAIN_START variant=geo_history_actual_event_replay_x0"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG_B1}" --data-path "${DATA}" \
  --secondary-adjacency "${SECONDARY_GRAPH}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${EXP_NAME}" --seed "${SEED}"
shopt -s nullglob
matches=("${TRAIN_ROOT}"/*_"${EXP_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#matches[@]} -eq 1 ]] || die "expected one B1 run, found ${#matches[@]}"
B1_RUN=${matches[0]}
cp "${ENVIRONMENT_FILE}" "${B1_RUN}/logs/server_environment.txt"
B1_RESULT="${RESULT_ROOT}/geo_history_actual_event_replay_x0_val_n${NSAMPLES}_seed${GEN_SEED}"

echo "GENERATION_START variant=geo_history_actual_event_replay_x0 members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${B1_RUN}" --data-path "${DATA}" \
  --output-dir "${B1_RESULT}" --split val \
  --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
  --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
  --energy-score-member-limit "${ENERGY_MEMBERS}"

echo "EVENT_REPLAY_AUDIT_START"
"${PYTHON_BIN}" tools/audit_station24_event_replay_x0.py \
  --run-dir "${B1_RUN}" --result-dir "${B1_RESULT}" \
  --output-dir "${REPLAY_AUDIT_ROOT}"

echo "COMPARISON_START"
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

echo "WIND_EVENT_TIMING_START"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${REFERENCE_HISTORY_RESULT}" "${B1_RESULT}" --data-path "${DATA}" \
  --output-dir "${TIMING_ROOT}" \
  --baseline-variant geo_history_actual_dual \
  --candidate-variant geo_history_actual_event_replay_x0 \
  --baseline-label "Historical-spatial baseline" \
  --candidate-label "Independent event replay + x0"

echo "FORECAST_EVENT_ATTRIBUTION_START"
"${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
  "${REFERENCE_HISTORY_RESULT}" "${B1_RESULT}" \
  --event-records "${TIMING_ROOT}/event_records.csv" --data-path "${DATA}" \
  --output-dir "${ATTRIBUTION_ROOT}"

echo "SUSTAINED_TAIL_AUDIT_START"
"${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
  --baseline "${REFERENCE_HISTORY_RESULT}" --candidate "${B1_RESULT}" \
  --data-path "${DATA}" --output-dir "${TAIL_ROOT}" --top-issues 5 \
  --baseline-label "Historical-spatial baseline" \
  --candidate-label "Independent event replay + x0"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_event_replay_x0_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "REFERENCE_HISTORY_RUN=${REFERENCE_HISTORY_RUN}"
  echo "REFERENCE_HISTORY_RESULT=${REFERENCE_HISTORY_RESULT}"
  echo "B1_RUN=${B1_RUN}"
  echo "B1_RESULT=${B1_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "FORECAST_ATTRIBUTION_DIR=${ATTRIBUTION_ROOT}"
  echo "EXTREME_TAIL_DIR=${TAIL_ROOT}"
  echo "EVENT_REPLAY_AUDIT_DIR=${REPLAY_AUDIT_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_EVENT_REPLAY_X0_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
