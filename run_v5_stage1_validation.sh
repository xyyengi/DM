#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
source "${REPO_ROOT}/server_stage1_common.sh"

if [[ "${STAGE1_INTERNAL_WORKER:-0}" != "1" ]]; then
  [[ $# -ge 1 ]] || stage1_die \
    "usage: bash run_v5_stage1_validation.sh RUN_DIR [RUN_DIR ...]"
  stage1_launch_background "$0" "v5_stage1_validation" "$@"
  exit 0
fi

stage1_install_status_trap
stage1_preflight "v5_stage1_validation"

[[ $# -ge 1 ]] || stage1_die "at least one training run directory is required"
DATA=${DATA:-diffusion_npy_normalized}
GEN_BATCH=${GEN_BATCH:-4}
NSAMPLES=${NSAMPLES:-20}
GEN_SEED=${GEN_SEED:-424242}
RUN_ABLATIONS=${RUN_ABLATIONS:-1}
RESUME_VALIDATION=${RESUME_VALIDATION:-0}
COMPARISON_ROOT=${COMPARISON_ROOT:-outputs_shandong/v5_stage1/comparisons}
[[ "${NSAMPLES}" == "20" ]] || stage1_die "paired validation is locked to NSAMPLES=20"
[[ "${GEN_SEED}" == "424242" ]] \
  || stage1_die "paired validation is locked to GEN_SEED=424242"
[[ "${RUN_ABLATIONS}" == "0" || "${RUN_ABLATIONS}" == "1" ]] \
  || stage1_die "RUN_ABLATIONS must be 0 or 1"
[[ "${RESUME_VALIDATION}" == "0" || "${RESUME_VALIDATION}" == "1" ]] \
  || stage1_die "RESUME_VALIDATION must be 0 or 1"
stage1_require_data "${DATA}"

declare -a RESULT_DIRS=()

write_validation_metadata() {
  local result_dir=$1
  local run_dir=$2
  local architecture=$3
  local rank=$4
  local epoch=$5
  local val_loss=$6
  local ablation=$7
  local generation_seconds=$8
  local checkpoint_path=$9
  "${PYTHON_BIN}" - \
    "${result_dir}" "${run_dir}" "${architecture}" "${rank}" "${epoch}" \
    "${val_loss}" "${ablation}" "${generation_seconds}" "${checkpoint_path}" \
    "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path
import yaml

(
    result_dir, run_dir, architecture, rank, epoch, val_loss, ablation,
    generation_seconds, checkpoint_path, commit,
) = sys.argv[1:]
with (Path(run_dir) / "config_used.yaml").open("r", encoding="utf-8") as handle:
    training_config = yaml.safe_load(handle)
record = {
    "training_run_dir": str(Path(run_dir).resolve()),
    "training_run_name": Path(run_dir).name,
    "result_dir": str(Path(result_dir).resolve()),
    "architecture": architecture,
    "training_seed": int(training_config["train"]["seed"]),
    "checkpoint_rank": int(rank),
    "checkpoint_epoch": int(epoch),
    "validation_epsilon_mse": float(val_loss),
    "checkpoint_path": str(Path(checkpoint_path).resolve()),
    "data_split": "val",
    "reverse_variance_type": "posterior",
    "n_samples": 20,
    "generation_seed": 424242,
    "condition_ablation": ablation,
    "generation_seconds": int(generation_seconds),
    "commit": commit,
}
with (Path(result_dir) / "validation_metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(record, handle, ensure_ascii=False, indent=2)
PY
}

generate_one() {
  local run_dir=$1
  local run_parent=$2
  local run_name=$3
  local architecture=$4
  local rank=$5
  local epoch=$6
  local val_loss=$7
  local ablation=$8
  local checkpoint_path="${run_dir}/checkpoints/model_epoch_${epoch}.pt"
  [[ -f "${checkpoint_path}" ]] \
    || stage1_die "top-${rank} checkpoint missing: ${checkpoint_path}"

  local suffix="val_rank${rank}_epoch${epoch}_posterior_n${NSAMPLES}_seed${GEN_SEED}"
  if [[ "${ablation}" != "none" ]]; then
    suffix="${suffix}_ablate_${ablation}"
  fi
  local result_dir="${run_parent}/${run_name}_${suffix}"
  if [[ -e "${result_dir}" ]]; then
    if [[ "${RESUME_VALIDATION}" == "1" \
          && -f "${result_dir}/metrics.json" \
          && -f "${result_dir}/generation_config_used.yaml" \
          && -f "${result_dir}/validation_metadata.json" ]]; then
      RESULT_DIRS+=("${result_dir}")
      echo "VALIDATION_REUSE_COMPLETE result=${result_dir}"
      return
    fi
    stage1_die "refusing to overwrite incomplete/existing validation result: ${result_dir}"
  fi

  local started finished generation_seconds
  started=$(date +%s)
  "${PYTHON_BIN}" generate.py \
    --save_path "${run_parent}" \
    --exp_name "${run_name}" \
    --data_path "${DATA}" \
    --split val \
    --ckpt_epoch "${epoch}" \
    --n_samples "${NSAMPLES}" \
    --batch_size "${GEN_BATCH}" \
    --seed "${GEN_SEED}" \
    --reverse_variance_type posterior \
    --condition_ablation "${ablation}" \
    --output_dir "${result_dir}"
  finished=$(date +%s)
  generation_seconds=$((finished - started))
  [[ -f "${result_dir}/metrics.json" ]] || stage1_die "metrics.json missing"
  [[ -f "${result_dir}/generation_config_used.yaml" ]] \
    || stage1_die "generation_config_used.yaml missing"
  write_validation_metadata \
    "${result_dir}" "${run_dir}" "${architecture}" "${rank}" "${epoch}" \
    "${val_loss}" "${ablation}" "${generation_seconds}" "${checkpoint_path}"
  RESULT_DIRS+=("${result_dir}")
  echo "VALIDATION_COMPLETE result=${result_dir} seconds=${generation_seconds}"
}

for run_argument in "$@"; do
  run_argument=${run_argument%/}
  [[ -d "${run_argument}" ]] || stage1_die "run directory not found: ${run_argument}"
  run_parent=$(cd "$(dirname "${run_argument}")" && pwd)
  run_name=$(basename "${run_argument}")
  run_dir="${run_parent}/${run_name}"
  config_path="${run_dir}/config_used.yaml"
  manifest_path="${run_dir}/checkpoints/top_checkpoints.json"
  [[ -f "${config_path}" ]] || stage1_die "config_used.yaml missing: ${run_dir}"
  [[ -f "${manifest_path}" ]] || stage1_die "top checkpoint manifest missing: ${run_dir}"

  architecture=$("${PYTHON_BIN}" - "${config_path}" <<'PY'
import sys
from pathlib import Path
import yaml
from src.eval.stage1_protocol import failed_checks, validation_protocol_checks

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
architecture = config["model"].get("architecture", "v4_legacy")
if architecture not in {"v4_legacy", "v5_t", "v5_tf"}:
    raise SystemExit(f"unsupported architecture: {architecture}")
checks = validation_protocol_checks(config)
failed = failed_checks(checks)
if failed:
    raise SystemExit(f"validation protocol mismatch: {failed}")
print(architecture)
PY
  )

  mapfile -t checkpoint_rows < <("${PYTHON_BIN}" - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path
with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
checkpoints = manifest.get("checkpoints", [])
if len(checkpoints) < 3:
    raise SystemExit(f"expected top-3 checkpoints, found {len(checkpoints)}")
for expected_rank, item in enumerate(checkpoints[:3], start=1):
    rank = int(item.get("rank", expected_rank))
    if rank != expected_rank:
        raise SystemExit(f"non-contiguous checkpoint rank: {rank}")
    print(f"{rank}\t{int(item['epoch'])}\t{float(item['val_loss']):.17g}")
PY
  )
  [[ ${#checkpoint_rows[@]} -eq 3 ]] \
    || stage1_die "failed to resolve three checkpoints for ${run_dir}"

  for row in "${checkpoint_rows[@]}"; do
    IFS=$'\t' read -r rank epoch val_loss <<< "${row}"
    generate_one \
      "${run_dir}" "${run_parent}" "${run_name}" "${architecture}" \
      "${rank}" "${epoch}" "${val_loss}" "none"
    if [[ "${architecture}" == "v5_tf" && "${rank}" == "1" \
          && "${RUN_ABLATIONS}" == "1" ]]; then
      generate_one \
        "${run_dir}" "${run_parent}" "${run_name}" "${architecture}" \
        "${rank}" "${epoch}" "${val_loss}" "forecast"
      generate_one \
        "${run_dir}" "${run_parent}" "${run_name}" "${architecture}" \
        "${rank}" "${epoch}" "${val_loss}" "calendar"
    fi
  done
done

mkdir -p "${COMPARISON_ROOT}"
comparison_dir="${COMPARISON_ROOT}/${STAGE1_JOB_STAMP}"
[[ ! -e "${comparison_dir}" ]] \
  || stage1_die "comparison directory already exists: ${comparison_dir}"
"${PYTHON_BIN}" tools/compare_v5_stage1_results.py \
  --output-dir "${comparison_dir}" \
  "${RESULT_DIRS[@]}"
echo "COMPARISON_DIR=${comparison_dir}"
