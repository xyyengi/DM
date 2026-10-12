#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/DM
JOB=${1:-$(date +%Y%m%d_%H%M%S)};ROOT="outputs_shandong/station24/lightweight_tail_capacity_stage1_$JOB";mkdir -p "$ROOT" logs/station24
STATUS="$ROOT/STAGE_STATUS.env";PHASE=initializing;CURRENT=none;COMPLETED=0
write_status(){ printf 'state=%s\nphase=%s\ncurrent=%s\ncompleted=%s\ntotal=6\nupdated_at=%s\n' "$1" "$PHASE" "$CURRENT" "$COMPLETED" "$(date --iso-8601=seconds)" > "$STATUS"; }
on_error(){ c=$?;write_status failed;echo "STAGE1_SEARCH_FAILED phase=$PHASE current=$CURRENT code=$c";exit "$c";};trap on_error ERR;write_status running
python tools/build_station24_tail_capacity_search.py;cp SEARCH_PROTOCOL.md station24_tail_capacity_search_manifest.csv "$ROOT/"
RAW500=outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242
python -c 'from pathlib import Path;import numpy as np,sys;from tools.merge_station24_independent_tail_members import MEMBER_ARRAYS;p=Path(sys.argv[1]);[(lambda a,n:(_ for _ in ()).throw(ValueError(n)) if a.ndim!=4 or a.shape[1]!=500 else None)(np.load(p/n,mmap_mode="r"),n) for n in MEMBER_ARRAYS];print("RAW500_ARTIFACT_PASS")' "$RAW500"
RAW150="$ROOT/raw_prefix_n150_seed424242";if [[ ! -f "$RAW150/metrics.json" ]];then python tools/subset_station24_members.py --source "$RAW500" --output "$RAW150" --members 150 --data-path diffusion_input_station;fi
for tag in c16_d2_h3 c24_d2_h3 c32_d2_h3 c48_d2_h3 c24_d3_h3 c24_d2_h2;do
 CURRENT=$tag;PHASE=candidate;write_status running
 if grep -q '^state=complete$' "$ROOT/$tag/status.env" 2>/dev/null;then echo "REUSED $tag";else bash run_station24_lightweight_tail_capacity_stage1_candidate.sh "$tag" "$ROOT";fi
 COMPLETED=$((COMPLETED+1));write_status running
done
CURRENT=none;PHASE=summarizing;write_status running
python tools/summarize_station24_tail_capacity_stage1.py --root "$ROOT"
PHASE=review_required;write_status complete
echo "STAGE1_SIX_COMPLETE root=$ROOT promotion=NOT_RUN full_training=NOT_RUN"
