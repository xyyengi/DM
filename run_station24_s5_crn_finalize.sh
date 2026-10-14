#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONUNBUFFERED=1

ROOT=${1:?usage: run_station24_s5_crn_finalize.sh PIPELINE_ROOT}
DATA=${DATA_PATH:-diffusion_input_station}
RAW_RUN=${RAW_RUN:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027}
TAIL_RUN=${TAIL_RUN:-outputs_shandong/station24/independent_joint_tail_v2_event_balanced_20260910_200653/training/20260910_200730_independent_joint_tail_v2_event_balanced_seed2027_seed2027}
SEEDS=(271828 314159 161803)
STATUS="$ROOT/STATUS.env"
LOG=${CRN_LOG:-logs/station24/s5_crn_finalize_$(date +%Y%m%d_%H%M%S).log}
PHASE=initializing
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

write_status() {
  cat > "$STATUS" <<EOF
state=$1
phase=$PHASE
root=$ROOT
log=$LOG
updated_at=$(date --iso-8601=seconds)
EOF
}
trap 'code=$?; write_status failed; echo "S5_CRN_FINALIZE_FAILED phase=$PHASE code=$code"; exit $code' ERR
write_status running

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="$ROOT/seed_$SEED"
  RAW="$SEED_ROOT/raw_n500"
  TAIL="$SEED_ROOT/tail_n100"
  COUPLINGS="$SEED_ROOT/couplings"
  PHASE="seed_${SEED}_input_validation"
  for path in \
    "$RAW/actual_scenarios_normalized.npy" \
    "$TAIL/actual_scenarios_normalized.npy" \
    "$RAW/generation_metadata.json" \
    "$TAIL/generation_metadata.json"; do
    test -s "$path"
  done

  PHASE="seed_${SEED}_materialization"
  if [[ ! -s "$COUPLINGS/coupling_manifest.json" ]]; then
    if [[ -d "$COUPLINGS" ]]; then
      mv "$COUPLINGS" "${COUPLINGS}_incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
    python -m tools.materialize_station24_s5_crn_couplings \
      --raw-results "$RAW" --tail-results "$TAIL" --output-root "$COUPLINGS" \
      --data-path "$DATA" --generation-seed "$SEED" \
      --duration-seed 424242 --pairing-seed 424242 \
      --body-members 400 --tail-members 100 --edge-taper-hours 3 \
      --energy-score-member-limit 80
  else
    echo "COUPLINGS_REUSED seed=$SEED root=$COUPLINGS"
  fi

  PHASE="seed_${SEED}_event_evaluation"
  declare -A LABELS=(
    [full_v2_mixture]="Full V2 CRN"
    [crn_identity]="Original S5 CRN"
    [independent_permutation]="Original S5 independent permutation"
  )
  for VARIANT in full_v2_mixture crn_identity independent_permutation; do
    RESULT="$COUPLINGS/$VARIANT"
    EVENT="$SEED_ROOT/evaluation_$VARIANT/continuous_event_evaluation"
    if [[ ! -s "$EVENT/continuous_event_member_matches.csv" ]]; then
      if [[ -d "$EVENT" ]]; then
        mv "$EVENT" "${EVENT}_incomplete_$(date +%Y%m%d_%H%M%S)"
      fi
      python -m tools.evaluate_station24_jstd_events \
        --baseline "$RAW" --candidate "$RESULT" --candidate-run "$TAIL_RUN" \
        --data-path "$DATA" --output-dir "$EVENT" \
        --baseline-label "Raw CRN" --candidate-label "${LABELS[$VARIANT]}"
    else
      echo "EVENT_EVALUATION_REUSED seed=$SEED variant=$VARIANT"
    fi
    JOINT="$SEED_ROOT/evaluation_$VARIANT/joint_wind_solar_evaluation"
    if [[ ! -s "$JOINT/ordinary_comparison.csv" ]]; then
      if [[ -d "$JOINT" ]]; then
        mv "$JOINT" "${JOINT}_incomplete_$(date +%Y%m%d_%H%M%S)"
      fi
      python -m tools.evaluate_station24_diffusion_ts \
        --baseline "$RAW" --candidate "$RESULT" --data "$DATA" \
        --output "$JOINT" --candidate-label "${LABELS[$VARIANT]}"
    else
      echo "JOINT_EVALUATION_REUSED seed=$SEED variant=$VARIANT"
    fi
  done

  PHASE="seed_${SEED}_member_attribution"
  for ARM in crn_identity independent_permutation; do
    ATTR="$SEED_ROOT/attribution_$ARM"
    if [[ ! -s "$ATTR/s5_member_attribution_summary.json" ]]; then
      if [[ -d "$ATTR" ]]; then
        mv "$ATTR" "${ATTR}_incomplete_$(date +%Y%m%d_%H%M%S)"
      fi
      if [[ "$ARM" == crn_identity ]]; then
        LOCAL_LABEL="Original S5 CRN"
      else
        LOCAL_LABEL="Original S5 independent permutation"
      fi
      python -m tools.audit_station24_s5_member_attribution \
        --full-mixture "$COUPLINGS/full_v2_mixture" \
        --tail100 "$TAIL" \
        --full-event-evaluation "$SEED_ROOT/evaluation_full_v2_mixture/continuous_event_evaluation" \
        --localized "$COUPLINGS/$ARM" \
        --localized-event-evaluation "$SEED_ROOT/evaluation_$ARM/continuous_event_evaluation" \
        --event-targets "$TAIL_RUN/jstd_event_targets.json" \
        --full-label "Full V2 CRN" --localized-label "$LOCAL_LABEL" \
        --output-dir "$ATTR"
    else
      echo "ATTRIBUTION_REUSED seed=$SEED arm=$ARM"
    fi
  done
done

PHASE=cross_seed_summary
if [[ ! -s "$ROOT/cross_seed_summary/CRN_ATTRIBUTION_REPORT.md" ]]; then
  if [[ -d "$ROOT/cross_seed_summary" ]]; then
    mv "$ROOT/cross_seed_summary" "$ROOT/cross_seed_summary_incomplete_$(date +%Y%m%d_%H%M%S)"
  fi
  python -m tools.summarize_station24_s5_crn --root "$ROOT" --data-path "$DATA"
else
  echo "CROSS_SEED_SUMMARY_REUSED root=$ROOT/cross_seed_summary"
fi

PHASE=archive
write_status complete
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT_ARCHIVE="${ROOT}_reports_${STAMP}.tar.gz"
tar -czf "$REPORT_ARCHIVE" \
  "$ROOT/STATUS.env" "$ROOT/preflight_cuda" "$ROOT/cross_seed_summary" \
  "$ROOT"/seed_*/attribution_* "$ROOT"/seed_*/evaluation_*
test -s "$REPORT_ARCHIVE"
COMPLETE_ARCHIVE="${ROOT}_completed_${STAMP}.tar.gz"
tar -czf "$COMPLETE_ARCHIVE" "$ROOT"
test -s "$COMPLETE_ARCHIVE"

PHASE=complete
write_status complete
echo "S5_CRN_ATTRIBUTION_COMPLETE root=$ROOT reports_archive=$REPORT_ARCHIVE complete_archive=$COMPLETE_ARCHIVE"
echo "STOPPED_AFTER_CRN training=NOT_RUN test=NOT_RUN S5_V3=NOT_DEVELOPED"
