#!/usr/bin/env bash
# Idempotent post-training Stage-1A continuation. Never starts training.
set -Eeuo pipefail
ROOT=${1:?'usage: run_station24_auxiliary_alpha_stage1a_finalize.sh STAGE_ROOT'}
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOB=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/auxiliary_alpha_stage1a_finalize_${JOB}.log"
STATUS="logs/station24/auxiliary_alpha_stage1a_finalize_${JOB}.status"
RAW_CHECKPOINT="outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/checkpoints/model_best.pt"
BODY_RESULTS="outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242"
CONTROL_ROOT=${CONTROL_ROOT:-outputs_shandong/station24/lightweight_joint_tail_v2_fair_20260923_194256}
CONTROL_RESULT=${CONTROL_RESULT:-$CONTROL_ROOT/mixture_body400_tail100_n500}
CONTROL_RUN=${CONTROL_RUN:-$CONTROL_ROOT/20260923_194315_lightweight_joint_tail_v2_fair_20260923_194256_seed2027}
CONTROL_POST=${CONTROL_POST:-$CONTROL_ROOT/postprocess_20260924_152622}
PHASE=initializing
mkdir -p logs/station24
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() {
  code=$?; write_status failed
  echo "AUXILIARY_ALPHA_STAGE1A_FINALIZE_FAILED phase=$PHASE exit_code=$code root=$ROOT" >&2
  exit "$code"
}
trap on_error ERR
write_status running
for required in "$CONTROL_RESULT/metrics.json" "$CONTROL_RUN/checkpoints/model_best.pt" "$CONTROL_POST/continuous_event_evaluation/continuous_event_per_event.csv"; do
  test -f "$required" || { echo "missing alpha=1 control artifact: $required" >&2; false; }
done

for spec in "0.65:alpha_065:auxiliary_alpha_0p65:Auxiliary alpha 0.65" "1.35:alpha_135:auxiliary_alpha_1p35:Auxiliary alpha 1.35"; do
  IFS=: read -r ALPHA TAG VARIANT LABEL <<< "$spec"
  CANDIDATE_ROOT="$ROOT/$TAG"
  test -f "$CANDIDATE_ROOT/train_run.txt"
  RUN_DIR=$(cat "$CANDIDATE_ROOT/train_run.txt")
  test -f "$RUN_DIR/checkpoints/model_best.pt"
  PHASE="integrity_${TAG}"; write_status running
  python tools/audit_station24_auxiliary_alpha_run.py \
    --alpha "$ALPHA" --run-dir "$RUN_DIR" --raw-checkpoint "$RAW_CHECKPOINT" \
    --preflight "$CANDIDATE_ROOT/preflight_cuda/preflight.json" \
    --output "$CANDIDATE_ROOT/alpha_run_integrity.json"
  PHASE="finalize_${TAG}"; write_status running
  SKIP_ARCHIVE=1 bash run_station24_lightweight_joint_tail_finalize.sh \
    "$CANDIDATE_ROOT" "$BODY_RESULTS" "$CANDIDATE_ROOT/postprocess" "$VARIANT" "$LABEL"
  test -f "$CANDIDATE_ROOT/finalize_artifacts.json"
  MIXTURE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["mixture"])' "$CANDIDATE_ROOT/finalize_artifacts.json")
  EVENT_DIR=$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["event_dir"])' "$CANDIDATE_ROOT/finalize_artifacts.json")
  test -f "$MIXTURE/metrics.json"
  test -f "$EVENT_DIR/continuous_event_per_event.csv"
  PHASE="recovery_${TAG}"; write_status running
  RECOVERY="$CANDIDATE_ROOT/postprocess/drop_recovery_diagnostic"
  if [[ ! -f "$RECOVERY/drop_recovery_summary.csv" ]]; then
    if [[ -e "$RECOVERY" ]]; then
      RECOVERY="${RECOVERY}_retry_${JOB}"
      echo "INCOMPLETE_RECOVERY_PRESERVED new_output=$RECOVERY"
    fi
    python tools/diagnose_station24_wind_direction_recovery.py \
      --data-path diffusion_input_station --raw-result "$BODY_RESULTS" \
      --lightweight-result "$CONTROL_RESULT" \
      --ramp-selection-result "$MIXTURE" \
      --ramp-selection-run "$RUN_DIR" --output-dir "$RECOVERY" \
      --lightweight-label "Auxiliary alpha 1.00" --candidate-label "$LABEL"
  fi
  printf '%s\n' "$RECOVERY" > "$CANDIDATE_ROOT/drop_recovery_dir.txt"
done

PHASE=selection_evaluation; write_status running
SELECTION="$ROOT/selection"
if [[ -s "$SELECTION/selected_alpha.json" && -s "$SELECTION/ALPHA_SENSITIVITY_REPORT.md" ]]; then
  echo "ALPHA_SELECTION_REUSED output=$SELECTION"
else
  if [[ -e "$SELECTION" ]]; then
    SELECTION="${SELECTION}_retry_${JOB}"
    echo "INCOMPLETE_SELECTION_PRESERVED new_output=$SELECTION"
  fi
  python tools/evaluate_station24_auxiliary_alpha_stage1a.py \
    --data-path diffusion_input_station \
    --raw-result "$BODY_RESULTS" \
    --control-result "$CONTROL_RESULT" --control-run "$CONTROL_RUN" --control-post "$CONTROL_POST" \
    --alpha065-root "$ROOT/alpha_065" --alpha135-root "$ROOT/alpha_135" \
    --output-dir "$SELECTION"
fi
printf '%s\n' "$SELECTION" > "$ROOT/selection_dir.txt"
for required in ALPHA_SENSITIVITY_REPORT.md alpha_summary.csv alpha_body_metrics.csv alpha_extreme_metrics.csv alpha_bootstrap_ci.csv alpha_lead_day_metrics.csv alpha_eventwise_metrics.csv alpha_integrity_audit.json selected_alpha.json; do
  test -s "$SELECTION/$required" || { echo "missing selection artifact: $required" >&2; false; }
done
PHASE=archive; write_status running
ARCHIVE="${ROOT}_completed_${JOB}.tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
test -s "$ARCHIVE"
PHASE=complete; write_status complete
echo "AUXILIARY_ALPHA_STAGE1A_COMPLETE root=$ROOT archive=$ARCHIVE"
echo "STOPPED_AFTER_STAGE1A Stage1B=NOT_RUN test=NOT_RUN multi_seed=NOT_RUN"

