#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/DM
mkdir -p logs/station24
JOB=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT=${2:-outputs_shandong/station24/generation_seed_stability_${JOB}}
LOG="logs/station24/generation_seed_stability_${JOB}.log"
STATUS="logs/station24/generation_seed_stability_${JOB}.status"
nohup setsid env STABILITY_LOG="$LOG" STABILITY_STATUS="$STATUS" STABILITY_LOG_ATTACHED=1 \
  bash run_station24_generation_seed_stability.sh "$ROOT" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf 'state=starting\nphase=launcher\nroot=%s\nlog=%s\npid=%s\n' "$ROOT" "$LOG" "$PID" > "$STATUS"
echo "PID=$PID"
echo "ROOT=$ROOT"
echo "LOG=$LOG"
echo "STATUS=$STATUS"
echo "Monitor: tail -f '$LOG'"
