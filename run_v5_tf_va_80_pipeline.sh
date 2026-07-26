#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
STAGE1_LOG_DIR=${STAGE1_LOG_DIR:-logs/v5_stage2}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong/v5_stage2}
export STAGE1_LOG_DIR OUTPUTS_DIR
source "${REPO_ROOT}/server_stage1_common.sh"

if [[ "${STAGE1_INTERNAL_WORKER:-0}" != "1" ]]; then
  stage1_launch_background "$0" "v5_tf_va_80_pipeline" "$@"
  exit 0
fi

stage1_install_status_trap
stage1_preflight "v5_tf_va_80_pipeline"

DATA=${DATA:-diffusion_npy_normalized}
SEED=${SEED:-2027}
NSAMPLES=80
GEN_SEED=424242
GEN_BATCH=${GEN_BATCH:-1}
BASELINE_RUN=${BASELINE_RUN:-outputs_shandong/v5_stage1/20260723_232043_v5_tf_stage1_seed2027_20260723_232038}
PIPELINE_ROOT=${PIPELINE_ROOT:-${OUTPUTS_DIR}/comparison80_${STAGE1_JOB_STAMP}}
export DATA SEED

[[ "${SEED}" == "2027" ]] \
  || stage1_die "the paired 80-member experiment is locked to SEED=2027"
[[ "${GEN_BATCH}" =~ ^[1-9][0-9]*$ ]] \
  || stage1_die "GEN_BATCH must be a positive integer"
[[ -d "${BASELINE_RUN}" ]] \
  || stage1_die "baseline V5-TF run not found: ${BASELINE_RUN}"
[[ ! -e "${PIPELINE_ROOT}" ]] \
  || stage1_die "refusing to overwrite pipeline output: ${PIPELINE_ROOT}"
stage1_require_data "${DATA}"
mkdir -p "${PIPELINE_ROOT}"

PIPELINE_RESULT_FILE="${STAGE1_LOG%.log}.results.env"
export PIPELINE_RESULT_FILE

resolve_rank1() {
  local run_dir=$1
  local expected_architecture=$2
  "${PYTHON_BIN}" - "${run_dir}" "${expected_architecture}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

run_dir = Path(sys.argv[1])
expected_architecture = sys.argv[2]
expected_seed = int(sys.argv[3])
with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
architecture = config["model"].get("architecture", "v4_legacy")
training_seed = int(config["train"]["seed"])
if architecture != expected_architecture:
    raise SystemExit(
        f"expected architecture={expected_architecture}, got {architecture}"
    )
if training_seed != expected_seed:
    raise SystemExit(
        f"expected training seed={expected_seed}, got {training_seed}"
    )
with (run_dir / "checkpoints/top_checkpoints.json").open(
    "r", encoding="utf-8"
) as handle:
    manifest = json.load(handle)
rank1 = manifest["checkpoints"][0]
if int(rank1.get("rank", 1)) != 1:
    raise SystemExit("first checkpoint entry is not rank 1")
epoch = int(rank1["epoch"])
checkpoint = run_dir / "checkpoints" / f"model_epoch_{epoch}.pt"
if not checkpoint.is_file():
    raise SystemExit(f"rank-1 checkpoint missing: {checkpoint}")
print(f"{epoch}\t{float(rank1['val_loss']):.17g}")
PY
}

write_metadata() {
  local result_dir=$1
  local run_dir=$2
  local architecture=$3
  local epoch=$4
  local val_loss=$5
  local generation_seconds=$6
  "${PYTHON_BIN}" - \
    "${result_dir}" "${run_dir}" "${architecture}" "${epoch}" \
    "${val_loss}" "${generation_seconds}" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

import yaml

(
    result_dir, run_dir, architecture, epoch, val_loss,
    generation_seconds, commit,
) = sys.argv[1:]
run_dir = Path(run_dir)
with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
record = {
    "training_run_dir": str(run_dir.resolve()),
    "training_run_name": run_dir.name,
    "result_dir": str(Path(result_dir).resolve()),
    "architecture": architecture,
    "training_seed": int(config["train"]["seed"]),
    "checkpoint_rank": 1,
    "checkpoint_epoch": int(epoch),
    "validation_epsilon_mse": float(val_loss),
    "checkpoint_path": str(
        (run_dir / "checkpoints" / f"model_epoch_{epoch}.pt").resolve()
    ),
    "data_split": "val",
    "reverse_variance_type": "posterior",
    "n_samples": 80,
    "generation_seed": 424242,
    "condition_ablation": "none",
    "generation_seconds": int(generation_seconds),
    "commit": commit,
}
with (Path(result_dir) / "validation_metadata.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(record, handle, ensure_ascii=False, indent=2)
PY
}

generate_rank1_80() {
  local run_dir=$1
  local architecture=$2
  local result_dir=$3
  local run_parent run_name rank1 epoch val_loss started finished seconds
  run_parent=$(cd "$(dirname "${run_dir}")" && pwd)
  run_name=$(basename "${run_dir}")
  rank1=$(resolve_rank1 "${run_dir}" "${architecture}")
  IFS=$'\t' read -r epoch val_loss <<< "${rank1}"
  [[ ! -e "${result_dir}" ]] \
    || stage1_die "refusing to overwrite generation output: ${result_dir}"

  echo "GENERATION_START architecture=${architecture} members=${NSAMPLES}"
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
    --condition_ablation none \
    --output_dir "${result_dir}"
  finished=$(date +%s)
  seconds=$((finished - started))

  [[ -f "${result_dir}/metrics.json" ]] \
    || stage1_die "metrics.json missing: ${result_dir}"
  [[ -f "${result_dir}/actual_scenarios.npy" ]] \
    || stage1_die "actual_scenarios.npy missing: ${result_dir}"
  write_metadata \
    "${result_dir}" "${run_dir}" "${architecture}" \
    "${epoch}" "${val_loss}" "${seconds}"
  echo "GENERATION_COMPLETE architecture=${architecture} result=${result_dir}"
}

V5TF_RESULT="${PIPELINE_ROOT}/v5_tf_seed2027_val_rank1_posterior_n80_seed424242"
generate_rank1_80 "${BASELINE_RUN}" "v5_tf" "${V5TF_RESULT}"

stage1_run_training \
  "configs/v5_tf_va_stage2_168h.yaml" \
  "v5_tf_va" \
  "v5_tf_va_stage2"

shopt -s nullglob
va_runs=("${OUTPUTS_DIR}"/*_"v5_tf_va_stage2_seed${SEED}_${STAGE1_JOB_STAMP}")
shopt -u nullglob
[[ ${#va_runs[@]} -eq 1 ]] \
  || stage1_die "expected exactly one V5-TF-VA run, found ${#va_runs[@]}"
V5TFVA_RUN=${va_runs[0]}
V5TFVA_RESULT="${PIPELINE_ROOT}/v5_tf_va_seed2027_val_rank1_posterior_n80_seed424242"
generate_rank1_80 "${V5TFVA_RUN}" "v5_tf_va" "${V5TFVA_RESULT}"

COMPARISON_DIR="${PIPELINE_ROOT}/comparison"
"${PYTHON_BIN}" tools/compare_v5_stage1_results.py \
  "${V5TF_RESULT}" "${V5TFVA_RESULT}" \
  --expected-n-samples 80 \
  --output-dir "${COMPARISON_DIR}"

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "training_seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "V5TF_RUN=${BASELINE_RUN}"
  echo "V5TF_RESULT=${V5TF_RESULT}"
  echo "V5TFVA_RUN=${V5TFVA_RUN}"
  echo "V5TFVA_RESULT=${V5TFVA_RESULT}"
  echo "COMPARISON_DIR=${COMPARISON_DIR}"
  echo "ALL_V5_TF_VA_80_PIPELINE_COMPLETED"
} > "${PIPELINE_RESULT_FILE}"
cat "${PIPELINE_RESULT_FILE}"
