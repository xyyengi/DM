#!/usr/bin/env bash
# Start the complete V2 pipeline in the background with durable log/status files.
set -euo pipefail

cd /root/autodl-tmp/DM
JOB=${JOB:-$(date +%Y%m%d_%H%M%S)}
ROOT=${ROOT:-outputs_shandong/station24/independent_joint_tail_v2_event_balanced_${JOB}}
LOG=${LOG:-logs/station24/station24_independent_joint_tail_v2_${JOB}.log}
STATUS=${STATUS:-logs/station24/station24_independent_joint_tail_v2_${JOB}.status}
RAW_CHECKPOINT=${1:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/checkpoints/model_best.pt}
SECONDARY_GRAPH=${2:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/graphs/secondary_adjacency.npy}
RAW_BODY_RESULT=${3:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}
mkdir -p logs/station24
printf "state=starting\nroot=%s\nlog=%s\n" "$ROOT" "$LOG" > "$STATUS"

nohup bash -c '
  root=$1; log=$2; status=$3; raw_checkpoint=$4; secondary_graph=$5; raw_result=$6
  bash run_station24_independent_joint_tail_v2_pipeline.sh "$root" "$raw_checkpoint" "$secondary_graph" "$raw_result" > "$log" 2>&1
  code=$?
  if [[ $code -eq 0 ]]; then state=completed; else state=failed; fi
  printf "state=%s\npid=%s\nroot=%s\nlog=%s\nfinished_at=%s\nexit_code=%s\n" "$state" "$$" "$root" "$log" "$(date --iso-8601=seconds)" "$code" > "$status"
  exit "$code"
' bash "$ROOT" "$LOG" "$STATUS" "$RAW_CHECKPOINT" "$SECONDARY_GRAPH" "$RAW_BODY_RESULT" </dev/null >/dev/null 2>&1 &
PID=$!
printf "state=running\npid=%s\nroot=%s\nlog=%s\nstarted_at=%s\n" "$PID" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
echo "PID=$PID"
echo "ROOT=$ROOT"
echo "LOG=$LOG"
echo "STATUS=$STATUS"
echo "Monitor: tail -f '$LOG'"
