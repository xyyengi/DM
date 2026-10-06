#!/usr/bin/env bash
# Evaluation-only Stage-1A decision-state re-audit. Never trains or generates.
set -Eeuo pipefail
ROOT=${1:?'usage: run_station24_auxiliary_alpha_stage1a_decision_reaudit.sh STAGE_ROOT'}
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

JOB=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/auxiliary_alpha_stage1a_decision_reaudit_${JOB}.log"
STATUS="logs/station24/auxiliary_alpha_stage1a_decision_reaudit_${JOB}.status"
BODY_RESULTS="outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242"
CONTROL_ROOT=${CONTROL_ROOT:-outputs_shandong/station24/lightweight_joint_tail_v2_fair_20260923_194256}
CONTROL_RESULT=${CONTROL_RESULT:-$CONTROL_ROOT/mixture_body400_tail100_n500}
CONTROL_RUN=${CONTROL_RUN:-$CONTROL_ROOT/20260923_194315_lightweight_joint_tail_v2_fair_20260923_194256_seed2027}
CONTROL_POST=${CONTROL_POST:-$CONTROL_ROOT/postprocess_20260924_152622}
OUTPUT="$ROOT/selection_decision_state_v3_${JOB}"
PHASE=initializing

mkdir -p logs/station24
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\noutput=%s\nlog=%s\nupdated_at=%s\n' \
    "$1" "$PHASE" "$ROOT" "$OUTPUT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() {
  code=$?
  write_status failed
  echo "AUXILIARY_ALPHA_DECISION_REAUDIT_FAILED phase=$PHASE exit_code=$code root=$ROOT" >&2
  exit "$code"
}
trap on_error ERR
write_status running

PHASE=input_integrity
write_status running
for required in \
  "$BODY_RESULTS/metrics.json" \
  "$CONTROL_RESULT/metrics.json" \
  "$CONTROL_RUN/checkpoints/model_best.pt" \
  "$CONTROL_POST/continuous_event_evaluation/continuous_event_per_event.csv" \
  "$ROOT/frozen_protocol_hashes.json" \
  "$ROOT/alpha_065/finalize_artifacts.json" \
  "$ROOT/alpha_065/alpha_run_integrity.json" \
  "$ROOT/alpha_135/finalize_artifacts.json" \
  "$ROOT/alpha_135/alpha_run_integrity.json"; do
  test -s "$required" || { echo "missing required existing artifact: $required" >&2; false; }
done

PHASE=decision_reaudit
write_status running
python tools/evaluate_station24_auxiliary_alpha_stage1a.py \
  --data-path diffusion_input_station \
  --raw-result "$BODY_RESULTS" \
  --control-result "$CONTROL_RESULT" \
  --control-run "$CONTROL_RUN" \
  --control-post "$CONTROL_POST" \
  --alpha065-root "$ROOT/alpha_065" \
  --alpha135-root "$ROOT/alpha_135" \
  --output-dir "$OUTPUT"

for required in \
  ALPHA_SENSITIVITY_REPORT.md alpha_summary.csv alpha_body_metrics.csv \
  alpha_extreme_metrics.csv alpha_bootstrap_ci.csv alpha_lead_day_metrics.csv \
  alpha_eventwise_metrics.csv alpha_integrity_audit.json selected_alpha.json; do
  test -s "$OUTPUT/$required" || { echo "missing re-audit artifact: $required" >&2; false; }
done
python - "$OUTPUT/alpha_integrity_audit.json" "$OUTPUT/selected_alpha.json" <<'PY'
import json
import sys

integrity = json.load(open(sys.argv[1], encoding="utf-8"))
decision = json.load(open(sys.argv[2], encoding="utf-8"))
assert integrity["status"] == "PASS", integrity
assert integrity["evaluation_semantics_version"] == "stage1a_decision_state_v3", integrity
assert integrity["spatial_bootstrap_all_finite"] is True, integrity
assert decision["stage1b_started"] is False, decision
assert decision["test_used"] is False, decision
assert decision["multi_seed_started"] is False, decision
if decision["selection_status"] != "SELECTED_ALPHA_STAR":
    assert decision["stage1b_launch_eligible"] is False, decision
print("STAGE1A_DECISION_STATE_V3_INTEGRITY_PASS", decision["selection_status"])
PY

printf '%s\n' "$OUTPUT" > "$ROOT/selection_decision_state_v3_dir.txt"
PHASE=reports_archive
write_status running
ARCHIVE="${ROOT}_selection_decision_state_v3_${JOB}.tar.gz"
tar -czf "$ARCHIVE" -C "$ROOT" "$(basename "$OUTPUT")" selection_decision_state_v3_dir.txt
test -s "$ARCHIVE"

PHASE=complete
write_status complete
echo "AUXILIARY_ALPHA_DECISION_REAUDIT_COMPLETE output=$OUTPUT archive=$ARCHIVE"
echo "NO_TRAINING NO_GENERATION Stage1B=NOT_RUN test=NOT_RUN multi_seed=NOT_RUN"
