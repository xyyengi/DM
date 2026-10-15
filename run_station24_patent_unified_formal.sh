#!/usr/bin/env bash
# Paid-server formal lifecycle. Do not run before explicit user approval.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOB=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT="outputs_shandong/station24/patent_unified_lightweight_${JOB}"
LOG="logs/station24/patent_unified_lightweight_${JOB}.log"
STATUS="$ROOT/STATUS.env"
CONFIG=configs/station24_lightweight_joint_tail_v2_fair_168h.yaml
RAW_RUN=outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027
RAW_CHECKPOINT="$RAW_RUN/checkpoints/model_best.pt"
SECONDARY_GRAPH="$RAW_RUN/graphs/secondary_adjacency.npy"
PHASE=initializing
mkdir -p logs/station24 "$ROOT/training"
exec > >(tee -a "$LOG") 2>&1
write_status() { printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"; }
on_error() { code=$?; write_status failed; echo "PATENT_UNIFIED_FAILED phase=$PHASE code=$code" >&2; exit "$code"; }
trap on_error ERR
write_status running
for f in "$CONFIG" "$RAW_CHECKPOINT" "$SECONDARY_GRAPH" run_station24_patent_unified_finalize.sh; do test -f "$f" || { echo "missing: $f" >&2; false; }; done
test "$(git branch --show-current)" = experiment/24site-patent-unified
git diff --quiet
git diff --cached --quiet

PHASE=cuda_amp_preflight
write_status running
python -m tools.audit_station24_lightweight_joint_tail --config "$CONFIG" --checkpoint "$RAW_CHECKPOINT" \
  --secondary-adjacency "$SECONDARY_GRAPH" --data-path diffusion_input_station --device cuda --output "$ROOT/preflight_cuda"
python -c 'import json,sys; r=json.load(open(sys.argv[1])); bad={k:v for k,v in r["checks"].items() if v["status"]!="PASS"}; assert not bad,bad' "$ROOT/preflight_cuda/preflight.json"

PHASE=training
write_status running
python train_station24.py --config "$CONFIG" --data-path diffusion_input_station --output-root "$ROOT/training" \
  --exp-name "patent_unified_lightweight_${JOB}_seed2027" --initialize-checkpoint "$RAW_CHECKPOINT" --secondary-adjacency "$SECONDARY_GRAPH"
RUN_DIR=$(find "$ROOT/training" -type f -path '*/checkpoints/model_best.pt' -printf '%h\n' | sed 's#/checkpoints$##' | sort | tail -n 1)
test -f "$RUN_DIR/checkpoints/model_best.pt"
printf '%s\n' "$RUN_DIR" > "$ROOT/train_run.txt"

PHASE=posttrain_freeze_audit
write_status running
python -m tools.verify_station24_lightweight_freeze --raw-checkpoint "$RAW_CHECKPOINT" \
  --trained-checkpoint "$RUN_DIR/checkpoints/model_best.pt" --output "$ROOT/posttrain_freeze_audit.json"

PHASE=finalizing
write_status running
bash run_station24_patent_unified_finalize.sh "$ROOT" "$RAW_RUN"
PHASE=complete
write_status complete
echo "PATENT_UNIFIED_FORMAL_COMPLETE root=$ROOT"
