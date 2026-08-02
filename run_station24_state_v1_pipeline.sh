#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-wind-solar-168h}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_shandong/station24}
LOG_ROOT=${LOG_ROOT:-logs/station24}
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
  log_file="${LOG_ROOT}/station24_state_v1_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_state_v1_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_state_v1_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    OMP_NUM_THREADS=1 \
    STATION24_STATE_V1_INTERNAL_WORKER=1 \
    JOB_STAMP="${stamp}" LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" OUTPUT_ROOT="${OUTPUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
    SEED="${SEED}" NSAMPLES="${NSAMPLES}" GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started station24 State V1 pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_STATE_V1_INTERNAL_WORKER:-0}" != "1" ]]; then
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

CONFIG_CONTROL="configs/station24_state_v1_ramp36_control_168h.yaml"
CONFIG_STATE="configs/station24_state_v1_fixed_graph_168h.yaml"
for config in "${CONFIG_CONTROL}" "${CONFIG_STATE}"; do
  [[ -f "${config}" ]] || die "missing config ${config}"
done

"${PYTHON_BIN}" - "${CONFIG_CONTROL}" "${CONFIG_STATE}" <<'PY'
import copy
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    control = yaml.safe_load(handle)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    state = yaml.safe_load(handle)
for config in (control, state):
    config.pop("experiment", None)
allowed = {
    "use_forecast_ramps", "forecast_ramp_lags", "use_state_encoder",
    "state_feature_dim", "state_feature_names", "state_channels",
    "state_ramp_lags", "state_low_quantile", "state_high_quantile",
    "state_ramp_quantile", "state_clip", "state_global_gate_init",
    "state_film_gate_init",
}
control_base = copy.deepcopy(control)
state_base = copy.deepcopy(state)
for key in allowed:
    control_base["model"].pop(key, None)
    state_base["model"].pop(key, None)
if control_base != state_base:
    raise SystemExit("State V1 configs differ outside the approved state/ramp fields")
if not control["model"].get("use_forecast_ramps"):
    raise SystemExit("control must use raw ramps")
if control["model"].get("forecast_ramp_lags") != [3, 6]:
    raise SystemExit("control ramp lags must be [3, 6]")
if state["model"].get("use_forecast_ramps"):
    raise SystemExit("State V1 must not duplicate the raw ramp branch")
if not state["model"].get("use_state_encoder"):
    raise SystemExit("State V1 encoder is disabled")
print("STATE_V1_PARITY_OK")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/state_v1_${JOB_STAMP}"
TRAIN_ROOT="${PIPELINE_ROOT}/training"
RESULT_ROOT="${PIPELINE_ROOT}/validation_results"
COMPARISON_ROOT="${PIPELINE_ROOT}/comparison"
mkdir -p "${TRAIN_ROOT}" "${RESULT_ROOT}"

ENVIRONMENT_FILE="${LOG_FILE%.log}.environment.txt"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "branch=$(git branch --show-current)"
  echo "commit=$(git rev-parse HEAD)"
  echo "data=${DATA}"
  echo "seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "experiment_family=station24_state_v1"
  echo "test_used=false"
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

run_one() {
  local variant=$1
  local config=$2
  local experiment_name="station24_${variant}_${JOB_STAMP}"
  echo "TRAIN_START state_variant=${variant}"
  "${PYTHON_BIN}" train_station24.py \
    --config "${config}" --data-path "${DATA}" \
    --output-root "${TRAIN_ROOT}" --exp-name "${experiment_name}" \
    --seed "${SEED}"
  shopt -s nullglob
  local matches=("${TRAIN_ROOT}"/*_"${experiment_name}"_seed"${SEED}")
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] \
    || die "expected one run for ${variant}, found ${#matches[@]}"
  local run_dir=${matches[0]}
  cp "${ENVIRONMENT_FILE}" "${run_dir}/logs/server_environment.txt"
  local result_dir="${RESULT_ROOT}/${variant}_val_n${NSAMPLES}_seed${GEN_SEED}"
  echo "GENERATION_START state_variant=${variant}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${run_dir}" --data-path "${DATA}" \
    --output-dir "${result_dir}" --split val \
    --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
    --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}"
  echo "EXPERIMENT_COMPLETE state_variant=${variant} run_dir=${run_dir} result_dir=${result_dir}"
  printf '%s\t%s\n' "${run_dir}" "${result_dir}"
}

CONTROL_RECORD=$(run_one "ramp36_control" "${CONFIG_CONTROL}" | tee /dev/stderr | tail -n 1)
STATE_RECORD=$(run_one "state_v1_fixed_graph" "${CONFIG_STATE}" | tee /dev/stderr | tail -n 1)
IFS=$'\t' read -r CONTROL_RUN CONTROL_RESULT <<< "${CONTROL_RECORD}"
IFS=$'\t' read -r STATE_RUN STATE_RESULT <<< "${STATE_RECORD}"

"${PYTHON_BIN}" tools/compare_station24_state_v1.py \
  "${CONTROL_RESULT}" "${STATE_RESULT}" \
  --data-path "${DATA}" --output-dir "${COMPARISON_ROOT}"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_state_v1_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "CONTROL_RUN=${CONTROL_RUN}"
  echo "CONTROL_RESULT=${CONTROL_RESULT}"
  echo "STATE_V1_RUN=${STATE_RUN}"
  echo "STATE_V1_RESULT=${STATE_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_STATION24_STATE_V1_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
