#!/usr/bin/env bash
# One Stage-1 candidate: CUDA gate -> 1520-update pause -> 120+30 numeric evaluation.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh;conda activate dm_env
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
TAG=${1:?'usage: stage1_candidate.sh TAG SEARCH_ROOT'};ROOT=${2:?'missing SEARCH_ROOT'}
OUT="$ROOT/$TAG";CONFIG="configs/station24_lightweight_tail_capacity_search/$TAG.yaml";mkdir -p "$OUT" "$OUT/training"
RAW_RUN=outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027
RAW_CKPT="$RAW_RUN/checkpoints/model_best.pt";GRAPH="$RAW_RUN/graphs/secondary_adjacency.npy"
ORIGINAL=outputs_shandong/station24/lightweight_joint_tail_v2_fair_20260923_194256/mixture_body400_tail100_n500
RAW500=$(cat "$ROOT/raw500_path.txt");RAW150="$ROOT/raw_prefix_n150_seed424242";STATUS="$OUT/status.env";LOG="$OUT/stage1.log";PHASE=init
exec > >(tee -a "$LOG") 2>&1
status(){ printf 'state=%s\nphase=%s\ntag=%s\nupdated_at=%s\n' "$1" "$PHASE" "$TAG" "$(date --iso-8601=seconds)" > "$STATUS"; }
fail(){ c=$?;status failed;echo "STAGE1_FAILED tag=$TAG phase=$PHASE code=$c";exit "$c";};trap fail ERR;status running
for f in "$CONFIG" "$RAW_CKPT" "$GRAPH" "$RAW500/metrics.json" "$RAW150/metrics.json";do test -f "$f";done
if pgrep -af 'train_shandong91|generate_shandong91|run_shandong91'|grep -v grep;then echo 'Station-91 task detected; no Station-24 launch';false;fi
PHASE=cuda_amp_gate;status running
if ! python -c 'import json,sys;assert json.load(open(sys.argv[1]))["launch_eligible"]' "$OUT/preflight_cuda/preflight.json" 2>/dev/null;then
 if [[ -e "$OUT/preflight_cuda" ]];then mv "$OUT/preflight_cuda" "$OUT/preflight_cuda_failed_$(date +%Y%m%d_%H%M%S)";fi
 python -m tools.audit_station24_lightweight_joint_tail --config "$CONFIG" --checkpoint "$RAW_CKPT" --secondary-adjacency "$GRAPH" --data-path diffusion_input_station --body-results "$RAW500" --device cuda --formal-gate --output "$OUT/preflight_cuda"
fi
python -c 'import json,sys;assert json.load(open(sys.argv[1]))["launch_eligible"]' "$OUT/preflight_cuda/preflight.json"
PHASE=train_to_1520;status running
if [[ ! -f "$OUT/train_run.txt" ]];then
 python train_station24.py --config "$CONFIG" --data-path diffusion_input_station --output-root "$OUT/training" --exp-name "capacity_stage1_${TAG}" --initialize-checkpoint "$RAW_CKPT" --secondary-adjacency "$GRAPH" --max-optimizer-updates 1520 --disable-early-stop
 RUN=$(find "$OUT/training" -type f -name 'model_pause_updates_1520.pt' -printf '%h\n'|sed 's#/checkpoints$##'|sort|tail -1);test -n "$RUN";printf '%s\n' "$RUN">"$OUT/train_run.txt"
fi
RUN=$(cat "$OUT/train_run.txt");PAUSE="$RUN/checkpoints/model_pause_updates_1520.pt";test -f "$PAUSE";test -f "$RUN/checkpoints/model_best.pt"
python -c 'import torch,sys;p=torch.load(sys.argv[1],map_location="cpu",weights_only=False);s=p["training_state"];assert s["optimizer_updates"]==1520 and s["scheduler"]=="none" and s["scaler_state_dict"] is not None' "$PAUSE"
PHASE=generate_tail30;status running
TAIL="$OUT/tail_n30_seed424242"
if [[ ! -f "$TAIL/metrics.json" ]];then python generate_station24.py --run-dir "$RUN" --data-path diffusion_input_station --output-dir "$TAIL" --split val --n-samples 30 --seed 424242 --checkpoint-state raw --result-variant "capacity_stage1_${TAG}";fi
PHASE=merge_120_30;status running
MIX="$OUT/mixture_raw120_tail30_n150"
if [[ ! -f "$MIX/metrics.json" ]];then python tools/merge_station24_independent_tail_members.py --body-results "$RAW500" --body-member-limit 120 --tail-results "$TAIL" --tail-member-limit 30 --output-dir "$MIX" --data-path diffusion_input_station --energy-score-member-limit 80 --condition-variant "capacity_stage1_${TAG}_mixture" --family staged_capacity_screen_120_30;fi
BASE_VARIANT=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["run"]["condition_variant"])' "$RAW150/metrics.json")
CAND_VARIANT=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["run"]["condition_variant"])' "$MIX/metrics.json")
POST="$OUT/stage1_numeric_evaluation";mkdir -p "$POST";PHASE=numeric_evaluation;status running
if [[ ! -f "$POST/comparisons/comparison_report.md" ]];then python tools/compare_station24_multiscale_2a.py "$RAW150" "$MIX" --output-dir "$POST/comparisons" --baseline-variant "$BASE_VARIANT" --candidate-variant "$CAND_VARIANT" --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 --baseline-label 'Raw prefix n150' --candidate-label "$TAG" --candidate-spatial-levels bottleneck --title "Stage1 Raw150 vs $TAG";fi
if [[ ! -f "$POST/events/report.md" ]];then python tools/evaluate_station24_jstd_events.py --baseline "$RAW150" --candidate "$MIX" --candidate-run "$RUN" --data-path diffusion_input_station --output-dir "$POST/events" --baseline-label 'Raw prefix n150' --candidate-label "$TAG";fi
if [[ ! -f "$POST/joint/ordinary_comparison.csv" ]];then python tools/evaluate_station24_diffusion_ts.py --baseline "$RAW150" --candidate "$MIX" --data diffusion_input_station --output "$POST/joint" --candidate-label "$TAG";fi
RECOVERY_REFERENCE="$ORIGINAL"
if [[ ! -f "$RECOVERY_REFERENCE/actual_scenarios_normalized.npy" ]];then RECOVERY_REFERENCE="$RAW150";echo "ORIGINAL_LIGHTWEIGHT_ARRAYS_MISSING recovery_reference=Raw150";fi
if [[ ! -f "$POST/recovery/drop_recovery_summary.csv" ]];then python -m tools.diagnose_station24_wind_direction_recovery --data-path diffusion_input_station --raw-result "$RAW150" --lightweight-result "$RECOVERY_REFERENCE" --ramp-selection-result "$MIX" --ramp-selection-run "$RUN" --output-dir "$POST/recovery" --candidate-label "$TAG";fi
python -c 'import json,sys,pathlib;pathlib.Path(sys.argv[1]).write_text(json.dumps({"tag":sys.argv[2],"run":sys.argv[3],"pause":sys.argv[4],"tail30":sys.argv[5],"mixture150":sys.argv[6],"evaluation":sys.argv[7]},indent=2),encoding="utf-8")' "$OUT/stage1_artifacts.json" "$TAG" "$RUN" "$PAUSE" "$TAIL" "$MIX" "$POST"
PHASE=complete;status complete;echo "STAGE1_CANDIDATE_COMPLETE tag=$TAG output=$OUT"
