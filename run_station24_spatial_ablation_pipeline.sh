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

die() {
  echo "ERROR: $*" >&2
  exit 1
}

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
  log_file="${LOG_ROOT}/station24_three_experiments_${stamp}.log"
  pid_file="${LOG_ROOT}/station24_three_experiments_${stamp}.pid"
  status_file="${LOG_ROOT}/station24_three_experiments_${stamp}.status"
  nohup setsid env \
    PYTHONUNBUFFERED=1 \
    STATION24_INTERNAL_WORKER=1 \
    JOB_STAMP="${stamp}" \
    LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" \
    STATUS_FILE="${status_file}" \
    EXPECTED_BRANCH="${EXPECTED_BRANCH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATA="${DATA}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    SEED="${SEED}" \
    NSAMPLES="${NSAMPLES}" \
    GEN_SEED="${GEN_SEED}" \
    ISSUE_BATCH="${ISSUE_BATCH}" \
    MEMBER_CHUNK="${MEMBER_CHUNK}" \
    bash "$0" > "${log_file}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_file}"
  echo "Started station24 three-experiment pipeline"
  echo "PID: ${pid}"
  echo "Log: ${log_file}"
  echo "Status: ${status_file}"
  echo "Monitor: tail -f '${log_file}'"
  echo "Stop entire pipeline: kill -- -\$(cat '${pid_file}')"
}

if [[ "${STATION24_INTERNAL_WORKER:-0}" != "1" ]]; then
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
  train_lead_mark.npy train_fill_mask.npy val_forecast.npy val_actual.npy \
  val_residual.npy val_time_mark.npy val_lead_mark.npy val_fill_mask.npy \
  station_features.npy station_adjacency.npy station_order.csv export_metadata.json; do
  [[ -f "${DATA}/${required}" ]] || die "missing data artifact ${DATA}/${required}"
done

"${PYTHON_BIN}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the full station24 pipeline")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

CONFIG_NONE="configs/station24_no_spatial_168h.yaml"
CONFIG_FIXED="configs/station24_fixed_graph_168h.yaml"
CONFIG_GATED="configs/station24_type_gated_graph_168h.yaml"
for config in "${CONFIG_NONE}" "${CONFIG_FIXED}" "${CONFIG_GATED}"; do
  [[ -f "${config}" ]] || die "missing config ${config}"
done

# Enforce a strict ablation: configs may differ only in experiment text and
# the spatial mode/gate initialization.
"${PYTHON_BIN}" - "${CONFIG_NONE}" "${CONFIG_FIXED}" "${CONFIG_GATED}" <<'PY'
import copy
import sys

import yaml

configs = []
for path in sys.argv[1:]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    config.pop("experiment", None)
    config["model"].pop("spatial_mode", None)
    config["model"].pop("spatial_gate_init", None)
    configs.append(config)
if not all(config == configs[0] for config in configs[1:]):
    raise SystemExit("spatial ablation configs differ outside allowed fields")
print("CONFIG_ABLATION_PARITY_OK")
PY

PIPELINE_ROOT="${OUTPUT_ROOT}/three_experiments_${JOB_STAMP}"
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
  nvidia-smi
} > "${ENVIRONMENT_FILE}"

run_one() {
  local mode=$1
  local config=$2
  local experiment_name="station24_${mode}_${JOB_STAMP}"
  echo "TRAIN_START spatial_mode=${mode}"
  "${PYTHON_BIN}" train_station24.py \
    --config "${config}" \
    --data-path "${DATA}" \
    --output-root "${TRAIN_ROOT}" \
    --exp-name "${experiment_name}" \
    --seed "${SEED}"

  shopt -s nullglob
  local matches=("${TRAIN_ROOT}"/*_"${experiment_name}"_seed"${SEED}")
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] \
    || die "expected one run for ${mode}, found ${#matches[@]}"
  local run_dir=${matches[0]}
  cp "${ENVIRONMENT_FILE}" "${run_dir}/logs/server_environment.txt"

  local result_dir="${RESULT_ROOT}/${mode}_val_n${NSAMPLES}_seed${GEN_SEED}"
  echo "GENERATION_START spatial_mode=${mode}"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${run_dir}" \
    --data-path "${DATA}" \
    --output-dir "${result_dir}" \
    --split val \
    --n-samples "${NSAMPLES}" \
    --seed "${GEN_SEED}" \
    --issue-batch-size "${ISSUE_BATCH}" \
    --member-chunk-size "${MEMBER_CHUNK}"
  echo "EXPERIMENT_COMPLETE spatial_mode=${mode} run_dir=${run_dir} result_dir=${result_dir}"
  printf '%s\t%s\n' "${run_dir}" "${result_dir}"
}

NONE_RECORD=$(run_one "no_spatial" "${CONFIG_NONE}" | tee /dev/stderr | tail -n 1)
FIXED_RECORD=$(run_one "fixed_graph" "${CONFIG_FIXED}" | tee /dev/stderr | tail -n 1)
GATED_RECORD=$(run_one "type_gated_graph" "${CONFIG_GATED}" | tee /dev/stderr | tail -n 1)
IFS=$'\t' read -r NONE_RUN NONE_RESULT <<< "${NONE_RECORD}"
IFS=$'\t' read -r FIXED_RUN FIXED_RESULT <<< "${FIXED_RECORD}"
IFS=$'\t' read -r GATED_RUN GATED_RESULT <<< "${GATED_RECORD}"

"${PYTHON_BIN}" tools/compare_station24_spatial_ablation.py \
  "${NONE_RESULT}" "${FIXED_RESULT}" "${GATED_RESULT}" \
  --data-path "${DATA}" \
  --output-dir "${COMPARISON_ROOT}"

RESULT_FILE="${LOG_FILE%.log}.results.env"
ARCHIVE="${OUTPUT_ROOT}/station24_three_experiments_${JOB_STAMP}.tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "NO_SPATIAL_RUN=${NONE_RUN}"
  echo "NO_SPATIAL_RESULT=${NONE_RESULT}"
  echo "FIXED_GRAPH_RUN=${FIXED_RUN}"
  echo "FIXED_GRAPH_RESULT=${FIXED_RESULT}"
  echo "TYPE_GATED_GRAPH_RUN=${GATED_RUN}"
  echo "TYPE_GATED_GRAPH_RESULT=${GATED_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_ROOT}"
  echo "ARCHIVE=${ARCHIVE}"
  echo "ALL_STATION24_SPATIAL_EXPERIMENTS_COMPLETED"
} > "${RESULT_FILE}"
cat "${RESULT_FILE}"
