#!/usr/bin/env bash
set -euo pipefail

STAGE1_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE1_LOG_DIR=${STAGE1_LOG_DIR:-logs/v5_stage1}
EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/v5-risk-conditioned-film}
PYTHON_BIN=${PYTHON_BIN:-python}

stage1_die() {
  echo "ERROR: $*" >&2
  exit 1
}

stage1_launch_background() {
  local script_path=$1
  local job_name=$2
  shift 2

  cd "${STAGE1_REPO_ROOT}"
  mkdir -p "${STAGE1_LOG_DIR}"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local log_path="${STAGE1_LOG_DIR}/${job_name}_${stamp}.log"
  local pid_path="${STAGE1_LOG_DIR}/${job_name}_${stamp}.pid"
  local status_path="${STAGE1_LOG_DIR}/${job_name}_${stamp}.status"
  [[ ! -e "${log_path}" && ! -e "${pid_path}" && ! -e "${status_path}" ]] \
    || stage1_die "launcher files already exist for ${job_name}_${stamp}"

  nohup env \
    PYTHONUNBUFFERED=1 \
    STAGE1_INTERNAL_WORKER=1 \
    STAGE1_JOB_STAMP="${stamp}" \
    STAGE1_LOG="${log_path}" \
    STAGE1_PID_FILE="${pid_path}" \
    STAGE1_STATUS_FILE="${status_path}" \
    bash "${script_path}" "$@" > "${log_path}" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_path}"
  printf 'state=running\npid=%s\nstarted_at=%s\n' \
    "${pid}" "$(date --iso-8601=seconds)" > "${status_path}"

  echo "Started ${job_name}"
  echo "PID: ${pid}"
  echo "PID file: ${pid_path}"
  echo "Status file: ${status_path}"
  echo "Log: ${log_path}"
  echo "Monitor: tail -f '${log_path}'"
  echo "Stop only this job: kill \$(cat '${pid_path}')"
}

_stage1_record_exit() {
  local code=$?
  trap - EXIT
  if [[ -n "${STAGE1_STATUS_FILE:-}" ]]; then
    local state=completed
    [[ ${code} -eq 0 ]] || state=failed
    printf 'state=%s\npid=%s\nfinished_at=%s\nexit_code=%s\n' \
      "${state}" "${BASHPID}" "$(date --iso-8601=seconds)" "${code}" \
      > "${STAGE1_STATUS_FILE}"
  fi
  exit "${code}"
}

stage1_install_status_trap() {
  trap _stage1_record_exit EXIT
}

stage1_preflight() {
  local job_name=$1
  cd "${STAGE1_REPO_ROOT}"
  command -v git >/dev/null || stage1_die "git is not available"
  command -v nvidia-smi >/dev/null || stage1_die "nvidia-smi is not available"
  command -v "${PYTHON_BIN}" >/dev/null || stage1_die "${PYTHON_BIN} is not available"

  local branch
  branch=$(git branch --show-current)
  [[ "${branch}" == "${EXPECTED_BRANCH}" ]] \
    || stage1_die "expected branch ${EXPECTED_BRANCH}, got ${branch}"
  git diff --quiet || stage1_die "tracked working-tree changes are present"
  git diff --cached --quiet || stage1_die "staged changes are present"

  STAGE1_ENV_RECORD="${STAGE1_LOG%.log}.environment.txt"
  export STAGE1_ENV_RECORD
  {
    echo "job=${job_name}"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "repo=${STAGE1_REPO_ROOT}"
    echo "branch=${branch}"
    echo "commit=$(git rev-parse HEAD)"
    echo "git_status_begin"
    git status --short --branch
    echo "git_status_end"
    echo "nvidia_smi_begin"
    nvidia-smi
    echo "nvidia_smi_end"
  } > "${STAGE1_ENV_RECORD}"

  "${PYTHON_BIN}" - <<'PY' >> "${STAGE1_ENV_RECORD}"
import platform
import sys
import torch

print(f"python={sys.version.replace(chr(10), ' ')}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required; refusing to run full training on CPU")
print(f"cuda_device_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"gpu_{index}_name={props.name}")
    print(f"gpu_{index}_memory_bytes={props.total_memory}")
PY
  cat "${STAGE1_ENV_RECORD}"
}

stage1_require_data() {
  local data_path=$1
  [[ -d "${data_path}" ]] || stage1_die "data directory not found: ${data_path}"
  local required
  for required in train_pred.npy train_res.npy val_pred.npy val_res.npy normalization_params.json; do
    [[ -f "${data_path}/${required}" ]] \
      || stage1_die "required data artifact missing: ${data_path}/${required}"
  done
}

stage1_assert_training_protocol() {
  local config_path=$1
  local architecture=$2
  local seed=$3
  local batch=$4
  local epochs=$5
  local patience=$6
  "${PYTHON_BIN}" - "${config_path}" "${architecture}" "${seed}" \
    "${batch}" "${epochs}" "${patience}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path, expected_arch, seed, batch, epochs, patience = sys.argv[1:]
with Path(config_path).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
model = config["model"]
target = config["target"]
sampling = config["sampling"]
train = config["train"]
checks = {
    "architecture": model.get("architecture", "v4_legacy") == expected_arch,
    "length_168": int(config["data"]["length"]) == 168,
    "residual_target": target["type"] == "residual",
    "residual_standardization": bool(target["residual_standardization"]["enabled"]),
    "num_steps_500": int(model["num_steps"]) == 500,
    "linear_schedule": model["schedule"] == "linear",
    "posterior_variance": sampling["reverse_variance_type"] == "posterior",
    "validation_seed": int(train["validation_seed"]) == 314159,
    "top_k_three": int(train["top_k_checkpoints"]) == 3,
    "config_seed": int(train["seed"]) == int(seed),
    "runtime_batch_positive": int(batch) > 0,
    "runtime_epochs_positive": int(epochs) > 0,
    "runtime_patience_positive": int(patience) > 0,
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit(f"paired protocol check failed for {config_path}: {failed}")
print(f"PAIRED_PROTOCOL_OK config={config_path} architecture={expected_arch}")
PY
}

stage1_run_training() {
  local config_path=$1
  local architecture=$2
  local experiment_base=$3
  local data_path=${DATA:-diffusion_npy_normalized}
  local outputs_dir=${OUTPUTS_DIR:-outputs_shandong/v5_stage1}
  local seed=${SEED:-2026}
  local epochs=${EPOCHS:-150}
  local patience=${PATIENCE:-15}
  local batch=${BATCH:-64}
  local save_every=${SAVE_EVERY:-50}

  [[ "${seed}" == "2026" ]] \
    || stage1_die "stage-1 paired experiments are locked to SEED=2026"
  [[ -f "${config_path}" ]] || stage1_die "config not found: ${config_path}"
  stage1_require_data "${data_path}"
  stage1_assert_training_protocol \
    "${config_path}" "${architecture}" "${seed}" "${batch}" "${epochs}" "${patience}"

  mkdir -p "${outputs_dir}"
  local experiment_name="${experiment_base}_seed${seed}_${STAGE1_JOB_STAMP}"
  local train_started train_finished training_seconds
  train_started=$(date +%s)
  echo "TRAIN_START architecture=${architecture} experiment=${experiment_name}"
  "${PYTHON_BIN}" train.py \
    --config "${config_path}" \
    --data_path "${data_path}" \
    --save_path "${outputs_dir}" \
    --epochs "${epochs}" \
    --patience "${patience}" \
    --save_every "${save_every}" \
    --batch_size "${batch}" \
    --seed "${seed}" \
    --exp_name "${experiment_name}"
  train_finished=$(date +%s)
  training_seconds=$((train_finished - train_started))

  shopt -s nullglob
  local matches=("${outputs_dir}"/*_"${experiment_name}")
  shopt -u nullglob
  [[ ${#matches[@]} -eq 1 ]] \
    || stage1_die "expected one run directory for ${experiment_name}, found ${#matches[@]}"
  local run_dir=${matches[0]}
  [[ -f "${run_dir}/config_used.yaml" ]] || stage1_die "config_used.yaml missing"
  [[ -f "${run_dir}/checkpoints/model_best.pt" ]] || stage1_die "model_best.pt missing"
  [[ -f "${run_dir}/checkpoints/top_checkpoints.json" ]] \
    || stage1_die "top_checkpoints.json missing"

  cp "${STAGE1_ENV_RECORD}" "${run_dir}/logs/server_environment.txt"
  {
    echo "architecture=${architecture}"
    echo "experiment_name=${experiment_name}"
    echo "run_dir=${run_dir}"
    echo "branch=$(git branch --show-current)"
    echo "commit=$(git rev-parse HEAD)"
    echo "training_seed=${seed}"
    echo "validation_seed=314159"
    echo "batch_size=${batch}"
    echo "epochs=${epochs}"
    echo "patience=${patience}"
    echo "training_seconds=${training_seconds}"
    echo "completed_at=$(date --iso-8601=seconds)"
  } > "${run_dir}/logs/server_run_record.env"

  echo "TRAIN_COMPLETE architecture=${architecture}"
  echo "RUN_DIR=${run_dir}"
  echo "TRAINING_SECONDS=${training_seconds}"
  echo "Next: bash run_v5_stage1_validation.sh '${run_dir}'"
}
