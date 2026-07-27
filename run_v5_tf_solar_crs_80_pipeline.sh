#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
STAGE1_LOG_DIR=${STAGE1_LOG_DIR:-logs/v5_stage2}
OUTPUTS_DIR=${OUTPUTS_DIR:-outputs_shandong/v5_stage2}
export STAGE1_LOG_DIR OUTPUTS_DIR
source "${REPO_ROOT}/server_stage1_common.sh"

if [[ "${STAGE1_INTERNAL_WORKER:-0}" != "1" ]]; then
  stage1_launch_background "$0" "v5_tf_solar_crs_80_pipeline" "$@"
  exit 0
fi

stage1_install_status_trap
stage1_preflight "v5_tf_solar_crs_80_pipeline"

DATA=${DATA:-diffusion_npy_normalized}
SEED=${SEED:-2027}
NSAMPLES=80
GEN_SEED=424242
GEN_BATCH=${GEN_BATCH:-1}
PIPELINE_ROOT=${PIPELINE_ROOT:-${OUTPUTS_DIR}/solar_crs80_${STAGE1_JOB_STAMP}}
export DATA SEED

[[ "${SEED}" == "2027" ]] \
  || stage1_die "the paired solar-CRS 80-member experiment is locked to SEED=2027"
[[ "${GEN_BATCH}" =~ ^[1-9][0-9]*$ ]] \
  || stage1_die "GEN_BATCH must be a positive integer"
[[ ! -e "${PIPELINE_ROOT}" ]] \
  || stage1_die "refusing to overwrite pipeline output: ${PIPELINE_ROOT}"
stage1_require_data "${DATA}"
mkdir -p "${PIPELINE_ROOT}"

PIPELINE_RESULT_FILE="${STAGE1_LOG%.log}.results.env"
export PIPELINE_RESULT_FILE

resolve_rank1() {
  local run_dir=$1
  "${PYTHON_BIN}" - "${run_dir}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

run_dir = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
if config["model"].get("architecture") != "v5_tf":
    raise SystemExit("solar-CRS experiment must use architecture=v5_tf")
if int(config["train"]["seed"]) != expected_seed:
    raise SystemExit("training seed mismatch")
standardization = config["target"]["residual_standardization"]
if standardization.get("mode") != "solar_forecast_conditional":
    raise SystemExit("solar conditional residual standardization is not active")
if "fitted_stats" not in standardization:
    raise SystemExit("train-only fitted residual statistics are missing")
with (run_dir / "checkpoints/top_checkpoints.json").open(
    "r", encoding="utf-8"
) as handle:
    rank1 = json.load(handle)["checkpoints"][0]
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
  local epoch=$3
  local val_loss=$4
  local generation_seconds=$5
  "${PYTHON_BIN}" - \
    "${result_dir}" "${run_dir}" "${epoch}" "${val_loss}" \
    "${generation_seconds}" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path

import yaml

result_dir, run_dir, epoch, val_loss, generation_seconds, commit = sys.argv[1:]
run_dir = Path(run_dir)
with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
record = {
    "training_run_dir": str(run_dir.resolve()),
    "training_run_name": run_dir.name,
    "result_dir": str(Path(result_dir).resolve()),
    "architecture": "v5_tf",
    "experiment_variant": "solar_forecast_conditional_residual_standardization",
    "residual_standardization_mode": config["target"][
        "residual_standardization"
    ]["mode"],
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
    "test_used": False,
}
with (Path(result_dir) / "validation_metadata.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(record, handle, ensure_ascii=False, indent=2)
PY
}

stage1_run_training \
  "configs/v5_tf_solar_crs_stage2_168h.yaml" \
  "v5_tf" \
  "v5_tf_solar_crs_stage2"

shopt -s nullglob
runs=("${OUTPUTS_DIR}"/*_"v5_tf_solar_crs_stage2_seed${SEED}_${STAGE1_JOB_STAMP}")
shopt -u nullglob
[[ ${#runs[@]} -eq 1 ]] \
  || stage1_die "expected exactly one solar-CRS run, found ${#runs[@]}"
RUN_DIR=${runs[0]}

rank1=$(resolve_rank1 "${RUN_DIR}")
IFS=$'\t' read -r epoch val_loss <<< "${rank1}"
RESULT_DIR="${PIPELINE_ROOT}/v5_tf_solar_crs_seed2027_val_rank1_posterior_n80_seed424242"
[[ ! -e "${RESULT_DIR}" ]] \
  || stage1_die "refusing to overwrite generation output: ${RESULT_DIR}"

run_parent=$(cd "$(dirname "${RUN_DIR}")" && pwd)
run_name=$(basename "${RUN_DIR}")
echo "GENERATION_START variant=solar_crs members=${NSAMPLES}"
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
  --output_dir "${RESULT_DIR}"
finished=$(date +%s)
generation_seconds=$((finished - started))

[[ -f "${RESULT_DIR}/metrics.json" ]] \
  || stage1_die "metrics.json missing: ${RESULT_DIR}"
[[ -f "${RESULT_DIR}/actual_scenarios_constrained.npy" ]] \
  || stage1_die "projected scenarios missing: ${RESULT_DIR}"
[[ -f "${RESULT_DIR}/physical_projection.json" ]] \
  || stage1_die "physical projection audit missing: ${RESULT_DIR}"
write_metadata \
  "${RESULT_DIR}" "${RUN_DIR}" "${epoch}" "${val_loss}" \
  "${generation_seconds}"

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "training_seed=${SEED}"
  echo "generation_seed=${GEN_SEED}"
  echo "ensemble_members=${NSAMPLES}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "RESULT_DIR=${RESULT_DIR}"
  echo "ALL_V5_TF_SOLAR_CRS_80_PIPELINE_COMPLETED"
} > "${PIPELINE_RESULT_FILE}"
cat "${PIPELINE_RESULT_FILE}"
