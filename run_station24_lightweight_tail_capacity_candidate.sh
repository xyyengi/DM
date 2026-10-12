#!/usr/bin/env bash
# Resumable candidate: CUDA gate -> train -> generation/merge/evaluation -> archive.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
TAG=${1:?'usage: candidate.sh cCHANNEL_dDEPTH_hHAAR SEARCH_ROOT'}
SEARCH_ROOT=${2:?'missing SEARCH_ROOT'}
CONFIG="configs/station24_lightweight_tail_capacity_search/${TAG}.yaml"
ROOT="$SEARCH_ROOT/$TAG"; TRAIN_ROOT="$ROOT/training"
RAW_RUN=outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027
RAW_CHECKPOINT="$RAW_RUN/checkpoints/model_best.pt"; GRAPH="$RAW_RUN/graphs/secondary_adjacency.npy"
BODY=outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242
ORIGINAL=outputs_shandong/station24/lightweight_joint_tail_v2_fair_20260923_194256/mixture_body400_tail100_n500
mkdir -p "$ROOT" "$TRAIN_ROOT" logs/station24
STATUS="$ROOT/status.env"; LOG="$ROOT/pipeline.log"; PHASE=initializing
exec > >(tee -a "$LOG") 2>&1
status(){ printf 'state=%s\nphase=%s\ntag=%s\nupdated_at=%s\n' "$1" "$PHASE" "$TAG" "$(date --iso-8601=seconds)" > "$STATUS"; }
fail(){ code=$?;status failed;echo "CAPACITY_CANDIDATE_FAILED tag=$TAG phase=$PHASE code=$code";exit "$code"; };trap fail ERR
status running
for x in "$CONFIG" "$RAW_CHECKPOINT" "$GRAPH" "$BODY/metrics.json" "$ORIGINAL/metrics.json";do test -f "$x";done
if pgrep -af 'train_shandong91|generate_shandong91|run_shandong91' | grep -v grep;then echo '91-node GPU task detected; refusing to compete';false;fi
PHASE=cuda_amp_preflight;status running
if [[ ! -f "$ROOT/preflight_cuda/preflight.json" ]];then
  python -m tools.audit_station24_lightweight_joint_tail --config "$CONFIG" --checkpoint "$RAW_CHECKPOINT" --secondary-adjacency "$GRAPH" --data-path diffusion_input_station --device cuda --formal-gate --output "$ROOT/preflight_cuda"
fi
python -c 'import json,sys;p=json.load(open(sys.argv[1]));assert p["launch_eligible"]' "$ROOT/preflight_cuda/preflight.json"
PHASE=training;status running
if [[ ! -f "$ROOT/train_run.txt" ]];then
  python train_station24.py --config "$CONFIG" --data-path diffusion_input_station --output-root "$TRAIN_ROOT" --exp-name "lightweight_tail_capacity_${TAG}_seed2027" --initialize-checkpoint "$RAW_CHECKPOINT" --secondary-adjacency "$GRAPH"
  RUN=$(find "$TRAIN_ROOT" -type f -path '*/checkpoints/model_best.pt' -printf '%h\n'|sed 's#/checkpoints$##'|sort|tail -1);test -n "$RUN";printf '%s\n' "$RUN" > "$ROOT/train_run.txt"
fi
RUN=$(cat "$ROOT/train_run.txt");test -f "$RUN/checkpoints/model_best.pt"
PHASE=finalize;status running
SKIP_ARCHIVE=1 bash run_station24_lightweight_joint_tail_finalize.sh "$ROOT" "$BODY" "" "lightweight_tail_capacity_${TAG}" "Lightweight Tail ${TAG}"
PHASE=drop_recovery;status running
POST=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["postprocess"])' "$ROOT/finalize_artifacts.json")
MIXTURE=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["mixture"])' "$ROOT/finalize_artifacts.json")
if [[ ! -f "$POST/drop_recovery/drop_recovery_summary.csv" ]];then
  python -m tools.diagnose_station24_wind_direction_recovery --data-path diffusion_input_station --raw-result "$BODY" --lightweight-result "$ORIGINAL" --ramp-selection-result "$MIXTURE" --ramp-selection-run "$RUN" --output-dir "$POST/drop_recovery" --candidate-label "Lightweight Tail ${TAG}"
fi
PHASE=archive;status running
tar -czf "${ROOT}_completed.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")";test -s "${ROOT}_completed.tar.gz"
PHASE=complete;status complete
echo "CAPACITY_CANDIDATE_COMPLETE tag=$TAG root=$ROOT"
