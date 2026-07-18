#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
stamp=$(date +%Y%m%d_%H%M%S)
log="logs/shandong_v4s_${stamp}.log"
pid_file="${log%.log}.pid"

nohup env PYTHONUNBUFFERED=1 bash run_shandong_v4s.sh \
  > "${log}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${pid_file}"

echo "V4-s started in background"
echo "PID: ${pid}"
echo "Log: ${log}"
echo "PID file: ${pid_file}"
echo "Follow log: tail -f ${log}"
