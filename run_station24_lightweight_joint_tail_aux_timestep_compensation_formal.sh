#!/usr/bin/env bash
# One command: CUDA/AMP mechanism gate -> train -> generate/merge/evaluate/plot -> archive.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOB=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT="outputs_shandong/station24/lightweight_aux_timestep_compensation_${JOB}"
TRAIN_ROOT="$ROOT/training"
LOG="logs/station24/lightweight_aux_timestep_compensation_${JOB}.log"
STATUS="logs/station24/lightweight_aux_timestep_compensation_${JOB}.status"
CONFIG="configs/station24_lightweight_joint_tail_aux_timestep_compensation_168h.yaml"
RAW_RUN="outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027"
RAW_CHECKPOINT="$RAW_RUN/checkpoints/model_best.pt"
SECONDARY_GRAPH="$RAW_RUN/graphs/secondary_adjacency.npy"
BODY_RESULTS="outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242"
PHASE=initializing
mkdir -p logs/station24 "$ROOT" "$TRAIN_ROOT"
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() {
  code=$?
  write_status failed
  echo "AUX_TIMESTEP_COMPENSATION_FAILED phase=$PHASE exit_code=$code root=$ROOT log=$LOG" >&2
  exit "$code"
}
trap on_error ERR
write_status running
for required in "$CONFIG" "$RAW_CHECKPOINT" "$SECONDARY_GRAPH" "$BODY_RESULTS/metrics.json" "$BODY_RESULTS/generation_metadata.json" run_station24_lightweight_joint_tail_finalize.sh; do
  test -f "$required" || { echo "missing required input: $required" >&2; false; }
done
PHASE=cuda_amp_preflight
write_status running
python -m tools.audit_station24_aux_timestep_compensation \
  --config "$CONFIG" --checkpoint "$RAW_CHECKPOINT" \
  --secondary-adjacency "$SECONDARY_GRAPH" --data-path diffusion_input_station \
  --device cuda --output "$ROOT/preflight_cuda"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["checks"]["CUDA_AMP_forward_backward"]["status"]=="PASS" and r["launch_eligible"]' "$ROOT/preflight_cuda/preflight.json"
PHASE=training
write_status running
python train_station24.py \
  --config "$CONFIG" --data-path diffusion_input_station \
  --output-root "$TRAIN_ROOT" \
  --exp-name "lightweight_aux_timestep_compensation_${JOB}_seed2027" \
  --initialize-checkpoint "$RAW_CHECKPOINT" \
  --secondary-adjacency "$SECONDARY_GRAPH"
RUN_DIR=$(find "$TRAIN_ROOT" -type f -path '*/checkpoints/model_best.pt' -printf '%h\n' | sed 's#/checkpoints$##' | sort | tail -n 1)
test -n "$RUN_DIR"
test -f "$RUN_DIR/checkpoints/model_best.pt"
printf '%s\n' "$RUN_DIR" > "$ROOT/train_run.txt"
PHASE=finalizing
write_status running
bash run_station24_lightweight_joint_tail_finalize.sh \
  "$ROOT" "$BODY_RESULTS" "" \
  "lightweight_aux_timestep_compensation" \
  "Lightweight Tail Aux Timestep Compensation"
PHASE=complete
write_status complete
echo "AUX_TIMESTEP_COMPENSATION_COMPLETE root=$ROOT"
