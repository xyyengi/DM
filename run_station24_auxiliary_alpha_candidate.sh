#!/usr/bin/env bash
# One frozen Stage-1A alpha candidate: CUDA gate -> train. Finalization is separate.
set -Eeuo pipefail
CONFIG=${1:?'usage: run_station24_auxiliary_alpha_candidate.sh CONFIG CANDIDATE_ROOT ALPHA'}
ROOT=${2:?}
ALPHA=${3:?}
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RAW_CHECKPOINT="outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/checkpoints/model_best.pt"
SECONDARY_GRAPH="outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/graphs/secondary_adjacency.npy"
BODY_RESULTS="outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242"
TRAIN_ROOT="$ROOT/training"
mkdir -p "$ROOT" "$TRAIN_ROOT"
for required in "$CONFIG" "$RAW_CHECKPOINT" "$SECONDARY_GRAPH" "$BODY_RESULTS/metrics.json"; do
  test -f "$required" || { echo "missing required input: $required" >&2; false; }
done
if [[ "$ALPHA" != "0.65" && "$ALPHA" != "1.35" ]]; then
  echo "formal Stage 1A permits only new alpha 0.65 or 1.35" >&2
  exit 2
fi
python -m tools.audit_station24_lightweight_joint_tail \
  --config "$CONFIG" --device cuda --output "$ROOT/preflight_cuda"
test -f "$ROOT/preflight_cuda/preflight.json"
CONFIG_AUDIT="${CONFIG%.yaml}_audit.json"
test -f "$CONFIG_AUDIT"
python tools/check_station24_auxiliary_alpha_launch.py \
  --preflight "$ROOT/preflight_cuda/preflight.json" \
  --config-audit "$CONFIG_AUDIT" --alpha "$ALPHA" \
  --output "$ROOT/formal_launch_manifest.json"
python train_station24.py \
  --config "$CONFIG" \
  --data-path diffusion_input_station \
  --output-root "$TRAIN_ROOT" \
  --exp-name "auxiliary_alpha_${ALPHA/./p}_seed2027" \
  --initialize-checkpoint "$RAW_CHECKPOINT" \
  --secondary-adjacency "$SECONDARY_GRAPH"
RUN_DIR=$(find "$TRAIN_ROOT" -type f -path '*/checkpoints/model_best.pt' -printf '%h\n' | sed 's#/checkpoints$##' | sort)
test "$(printf '%s\n' "$RUN_DIR" | sed '/^$/d' | wc -l)" -eq 1
test -f "$RUN_DIR/checkpoints/model_best.pt"
printf '%s\n' "$RUN_DIR" > "$ROOT/train_run.txt"
python tools/audit_station24_auxiliary_alpha_run.py \
  --alpha "$ALPHA" --run-dir "$RUN_DIR" --raw-checkpoint "$RAW_CHECKPOINT" \
  --preflight "$ROOT/preflight_cuda/preflight.json" \
  --output "$ROOT/alpha_run_integrity.json"
echo "AUXILIARY_ALPHA_TRAINING_COMPLETE alpha=$ALPHA run=$RUN_DIR"

