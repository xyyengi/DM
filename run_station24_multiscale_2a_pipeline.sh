#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-wind-solar-168h}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
BASELINE_RESULT=${BASELINE_RESULT:-}
SEED=${SEED:-2027}
NSAMPLES=${NSAMPLES:-80}
GEN_SEED=${GEN_SEED:-424242}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}

die() { echo "ERROR: $*" >&2; exit 1; }

record_exit() {
  local code=$?
  trap - EXIT
  if [[ -n "${STATUS_FILE:-}" ]]; then
    local state=completed
    [[ ${code} -eq 0 ]] || state=failed
    printf 'state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n' \
      "${state}" "${BASHPID}" "$(date --iso-8601=seconds)" "${code}" \
      > "${STATUS_FILE}"
  fi
  exit "${code}"
}

launch_background() {
  cd "${REPO_ROOT}"
  mkdir -p "${LOG_ROOT}"
  local stamp log_file pid_file status_file
  stamp=$(date +%Y%m%d_%H%M%S)
  log_file="${LOG_ROOT}/station24_multiscale_2a_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_multiscale_2a_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_multiscale_2a_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    STATION24_MULTISCALE_2A_INTERNAL_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    BASELINE_RESULT="${BASELINE_RESULT}" \
    SEED="${SEED}" NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started Station24 Experiment 2A multiscale pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_MULTISCALE_2A_INTERNAL_WORKER:-0}" != "1" ]]; then
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

CONFIG_BASELINE="configs/station24_state_v1_fixed_graph_168h.yaml"
CONFIG_CANDIDATE="configs/station24_state_v1_multiscale_graph_168h.yaml"
for config in "${CONFIG_BASELINE}" "${CONFIG_CANDIDATE}"; do
  [[ -f "${config}" ]] || die "missing config ${config}"
done

"${PYTHON_BIN}" - "${CONFIG_BASELINE}" "${CONFIG_CANDIDATE}" "${DATA}" <<'PY'
import copy
import sys
import yaml

from src.models.station_conditioned_diffusion import Station24DiffusionModel
from station_dataset import load_station_static_data

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    baseline = yaml.safe_load(handle)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    candidate = yaml.safe_load(handle)

baseline_core = copy.deepcopy(baseline)
candidate_core = copy.deepcopy(candidate)
baseline_core.pop("experiment", None)
candidate_core.pop("experiment", None)
baseline_core["model"].pop("spatial_mix_levels", None)
candidate_core["model"].pop("spatial_mix_levels", None)
if baseline_core != candidate_core:
    raise SystemExit("Experiment 2A configs differ outside spatial_mix_levels")
if baseline["model"].get("spatial_mix_levels", ["bottleneck"]) != ["bottleneck"]:
    raise SystemExit("baseline must use bottleneck-only graph mixing")
expected = ["encoder_0", "encoder_1", "bottleneck"]
if candidate["model"].get("spatial_mix_levels") != expected:
    raise SystemExit("candidate must use 168h/84h/42h multiscale graph mixing")
if candidate["model"].get("state_feature_dim") != 4:
    raise SystemExit("Experiment 2A must keep the four State V1 indicators")

static = load_station_static_data(sys.argv[3])
models = [
    Station24DiffusionModel(
        config["model"],
        static["station_features"],
        static["station_adjacency"],
        static["station_capacities"],
    )
    for config in (baseline, candidate)
]
counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
channels = candidate["model"]["channel_multipliers"]
base = int(candidate["model"]["base_channels"])
expected_delta = sum(
    (base * value) ** 2 + 3 * (base * value) + 1
    for value in channels[:-1]
)
if counts[1] - counts[0] != expected_delta:
    raise SystemExit(
        f"unexpected parameter delta: actual={counts[1] - counts[0]} expected={expected_delta}"
    )
print(
    "MULTISCALE_2A_PARITY_OK "
    f"baseline_params={counts[0]} candidate_params={counts[1]} delta={expected_delta}"
)
PY

if [[ -z "${BASELINE_RESULT}" ]]; then
  while IFS= read -r -d '' candidate; do
    if [[ -z "${BASELINE_RESULT}" || "${candidate}" -nt "${BASELINE_RESULT}" ]]; then
      BASELINE_RESULT=${candidate}
    fi
  done < <(
    find "${OUTPUT_ROOT}" -type d \
      -name "state_v1_fixed_graph_val_n${NSAMPLES}_seed${GEN_SEED}" -print0
  )
fi
[[ -n "${BASELINE_RESULT}" && -d "${BASELINE_RESULT}" ]] \
  || die "State V1 baseline result not found; set BASELINE_RESULT=/path/to/result"

"${PYTHON_BIN}" - "${BASELINE_RESULT}" "${NSAMPLES}" "${GEN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = json.loads((path / "generation_metadata.json").read_text(encoding="utf-8"))
expected = {
    "condition_variant": "state_v1_fixed_graph",
    "spatial_mode": "fixed_graph",
    "split": "val",
    "n_samples": int(sys.argv[2]),
    "generation_seed": int(sys.argv[3]),
    "test_used": False,
}
for key, value in expected.items():
    if metadata.get(key) != value:
        raise SystemExit(f"baseline metadata mismatch {key}: {metadata.get(key)!r} != {value!r}")
if metadata.get("spatial_mix_levels", ["bottleneck"]) != ["bottleneck"]:
    raise SystemExit("baseline is not bottleneck-only graph mixing")
print(f"BASELINE_RESULT_OK path={path}")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/multiscale_2a_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparison"
TIMING_ROOT="${PIPELINE_ROOT}/wind_event_timing"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"
printf '%s\n' "${BASELINE_RESULT}" > "${PIPELINE_ROOT}/baseline_result_reference.txt"

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "baseline_result=${BASELINE_RESULT}"
  echo "seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "experiment_family=station24_multiscale_2a"
  echo "test_used=false"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

EXPERIMENT_NAME="station24_state_v1_multiscale_graph_${JOB_STAMP}"
echo "TRAIN_START variant=state_v1_multiscale_graph levels=168h,84h,42h"
"${PYTHON_BIN}" train_station24.py \
  --config "${CONFIG_CANDIDATE}" --data-path "${DATA}" \
  --output-root "${TRAIN_ROOT}" --exp-name "${EXPERIMENT_NAME}" \
  --seed "${SEED}"

shopt -s nullglob
RUN_MATCHES=("${TRAIN_ROOT}"/*_"${EXPERIMENT_NAME}"_seed"${SEED}")
shopt -u nullglob
[[ ${#RUN_MATCHES[@]} -eq 1 ]] \
  || die "expected one candidate run, found ${#RUN_MATCHES[@]}"
CANDIDATE_RUN=${RUN_MATCHES[0]}
cp "${ENVIRONMENT_FILE}" "${CANDIDATE_RUN}/logs/server_environment.txt"

CANDIDATE_RESULT="${RESULT_ROOT}/state_v1_multiscale_graph_val_n${NSAMPLES}_seed${GEN_SEED}"
echo "GENERATION_START variant=state_v1_multiscale_graph members=${NSAMPLES}"
"${PYTHON_BIN}" generate_station24.py \
  --run-dir "${CANDIDATE_RUN}" --data-path "${DATA}" \
  --output-dir "${CANDIDATE_RESULT}" --split val \
  --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
  --issue-batch-size "${ISSUE_BATCH}" \
  --member-chunk-size "${MEMBER_CHUNK}"

echo "COMPARISON_START baseline=${BASELINE_RESULT} candidate=${CANDIDATE_RESULT}"
"${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
  "${BASELINE_RESULT}" "${CANDIDATE_RESULT}" \
  --data-path "${DATA}" --output-dir "${COMPARISON_ROOT}"

echo "TIMING_DIAGNOSTIC_START"
"${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
  "${BASELINE_RESULT}" "${CANDIDATE_RESULT}" \
  --data-path "${DATA}" --output-dir "${TIMING_ROOT}" \
  --baseline-variant state_v1_fixed_graph \
  --candidate-variant state_v1_multiscale_graph \
  --baseline-label "State V1 bottleneck graph" \
  --candidate-label "Experiment 2A multiscale graph"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_multiscale_2a_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "BASELINE_RESULT=${BASELINE_RESULT}"
  echo "MULTISCALE_RUN=${CANDIDATE_RUN}"
  echo "MULTISCALE_RESULT=${CANDIDATE_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "TIMING_DIAGNOSTIC_DIR=${TIMING_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_STATION24_MULTISCALE_2A_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
