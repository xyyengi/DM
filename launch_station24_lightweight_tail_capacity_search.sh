#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/DM
JOB=${1:-$(date +%Y%m%d_%H%M%S)}
mkdir -p logs/station24
LOG="logs/station24/lightweight_tail_capacity_search_${JOB}.launcher.log"
STATUS="logs/station24/lightweight_tail_capacity_search_${JOB}.launcher.status"
nohup setsid bash run_station24_lightweight_tail_capacity_search.sh "$JOB" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf 'state=running\npid=%s\njob=%s\nlog=%s\n' "$PID" "$JOB" "$LOG" > "$STATUS"
echo "PID=$PID"
echo "LOG=$LOG"
echo "STATUS=$STATUS"
echo "ROOT=outputs_shandong/station24/lightweight_tail_capacity_search_${JOB}"
echo "MONITOR=tail -f $LOG"
