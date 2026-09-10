#!/usr/bin/env bash
# Train and screen the independent joint tail before any formal 500-member run.
set -euo pipefail

PIPELINE_ROOT=${1:?"usage: $0 PIPELINE_ROOT"}
RAW_CHECKPOINT=${2:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/checkpoints/model_best.pt}
SECONDARY_GRAPH=${3:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/graphs/secondary_adjacency.npy}
RAW_BODY_RESULT=${4:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
cd /root/autodl-tmp/DM
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

CONFIG=configs/station24_independent_joint_tail_v1_168h.yaml
test ! -e "$PIPELINE_ROOT"
mkdir -p "$PIPELINE_ROOT"

python tools/audit_station24_independent_joint_tail_preflight.py \
  --config "$CONFIG" --data-path diffusion_input_station \
  --checkpoint "$RAW_CHECKPOINT" --secondary-adjacency "$SECONDARY_GRAPH" \
  --output "$PIPELINE_ROOT/cuda_preflight.json"

python tools/check_station24_independent_tail_gate.py --pipeline-root "$PIPELINE_ROOT" --pretrain

TRAIN_ROOT="$PIPELINE_ROOT/training"
EXP_NAME="independent_joint_tail_v1_seed2027"
python train_station24.py \
  --config "$CONFIG" --data-path diffusion_input_station \
  --output-root "$TRAIN_ROOT" --exp-name "$EXP_NAME" \
  --secondary-adjacency "$SECONDARY_GRAPH" \
  --initialize-checkpoint "$RAW_CHECKPOINT"

mapfile -t RUNS < <(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d)
[[ ${#RUNS[@]} -eq 1 ]]
RUN_DIR="${RUNS[0]}"
printf '%s\n' "$RUN_DIR" > "$PIPELINE_ROOT/train_run.txt"
CHECKPOINT="$RUN_DIR/checkpoints/model_best.pt"
test -f "$CHECKPOINT"

# Development gate: only 20 newly sampled tail members.  The 80 body members
# are reused by slicing the locked 500-member Raw result in the merge utility.
TAIL_DEV="$PIPELINE_ROOT/development_tail_n20"
python generate_station24.py \
  --run-dir "$RUN_DIR" --data-path diffusion_input_station \
  --output-dir "$TAIL_DEV" --split val --n-samples 20 --seed 424242 \
  --checkpoint-state raw --result-variant independent_joint_tail_v1_dev

python tools/merge_station24_independent_tail_members.py \
  --body-results "$RAW_BODY_RESULT" --tail-results "$TAIL_DEV" \
  --output-dir "$PIPELINE_ROOT/development_mixture_body80_tail20_n100" \
  --data-path diffusion_input_station --body-member-limit 80 \
  --energy-score-member-limit 80

echo "INDEPENDENT_JOINT_TAIL_DEVELOPMENT_COMPLETE root=$PIPELINE_ROOT"
echo "TRAIN_RUN=$RUN_DIR"
echo "TAIL_DEV=$TAIL_DEV"
echo "MIXTURE_DEV=$PIPELINE_ROOT/development_mixture_body80_tail20_n100"
