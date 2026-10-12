#!/usr/bin/env bash
# Sequential, restart-safe 24-candidate orchestrator. It never launches 64ch automatically.
set -Eeuo pipefail
if [[ "${ALLOW_FULL24_SEARCH:-0}" != "1" ]];then
  echo 'Full 24-candidate execution is disabled by the revised staged protocol.' >&2
  echo 'Use launch_station24_lightweight_tail_capacity_search.sh for the six-candidate screen.' >&2
  exit 2
fi
cd /root/autodl-tmp/DM
JOB=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT="outputs_shandong/station24/lightweight_tail_capacity_search_${JOB}"
mkdir -p "$ROOT" logs/station24
python tools/build_station24_tail_capacity_search.py
cp SEARCH_PROTOCOL.md station24_tail_capacity_search_manifest.csv "$ROOT/"
LOG="logs/station24/lightweight_tail_capacity_search_${JOB}.log"
exec > >(tee -a "$LOG") 2>&1
echo "SEARCH_ROOT=$ROOT"
for c in 16 24 32 48;do
 for d in 1 2 3;do
  for h in 2 3;do
   tag="c${c}_d${d}_h${h}"
   if grep -q '^state=complete$' "$ROOT/$tag/status.env" 2>/dev/null;then echo "REUSED $tag";continue;fi
   bash run_station24_lightweight_tail_capacity_candidate.sh "$tag" "$ROOT"
  done
 done
done
echo "FIRST_ROUND_24_COMPLETE root=$ROOT boundary_expansion=NOT_RUN repeats=NOT_RUN"
