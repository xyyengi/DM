#!/usr/bin/env bash
# Idempotent materialization, evaluation, summary and archive. Never generates models.
set -Eeuo pipefail

CONFIG=${1:?"usage: $0 CONFIG OUTPUT_ROOT"}
ROOT=${2:?"usage: $0 CONFIG OUTPUT_ROOT"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
PY=${PYTHON_BIN:-python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/s5_crn_single_seed_finalize_${STAMP}.log"
STATUS="$ROOT/STATUS.env"
PHASE=initializing
test -s "$ROOT/EXPERIMENT_MANIFEST.json"
mkdir -p logs/station24
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' \
    "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
trap 'code=$?; write_status failed; echo "S5_CRN_FINALIZE_FAILED phase=$PHASE code=$code"; exit $code' ERR
write_status running

PHASE=input_validation
"$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --output-root "$ROOT"
DATA=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.data_path)
TAIL_RUN=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field paths.tail_run)
SEED=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.generation_seed)
DURATION_SEED=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.duration_seed)
EDGE_TAPER=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.edge_taper_hours)
ENERGY_LIMIT=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.energy_score_member_limit)
PERM_JSON=$("$PY" -m tools.validate_station24_s5_crn_stage2 --config "$CONFIG" --field experiment.permutation_seeds)
mapfile -t PERM_SEEDS < <("$PY" -c 'import json,sys; print(*json.loads(sys.argv[1]),sep="\n")' "$PERM_JSON")
[[ ${#PERM_SEEDS[@]} -eq 3 ]]
for path in "$ROOT/raw_n500/actual_scenarios_normalized.npy" "$ROOT/tail_n100/actual_scenarios_normalized.npy"; do test -s "$path"; done

PHASE=coupling_materialization
COUPLINGS="$ROOT/couplings"
COUPLINGS_READY=false
if [[ -s "$COUPLINGS/coupling_manifest.json" ]]; then
  if "$PY" - "$COUPLINGS" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
required = {
    "spatial_mode", "physical_projection", "spatial_mix_levels",
    "parallel_spatial_fusion_levels", "parallel_spatial_adjacency_mode",
    "parameter_count", "checkpoint_validation_objective",
}
for arm in ("full_v2_mixture", "crn_identity", "permutation_1", "permutation_2", "permutation_3"):
    run = json.loads((root / arm / "metrics.json").read_text(encoding="utf-8"))["run"]
    missing = sorted(required - set(run))
    if missing:
        raise SystemExit(f"{arm} missing evaluation metadata: {missing}")
print("COUPLING_METADATA_SCHEMA_PASS")
PY
  then
    COUPLINGS_READY=true
  fi
fi
if [[ "$COUPLINGS_READY" != true ]]; then
  [[ ! -e "$COUPLINGS" ]] || mv "$COUPLINGS" "${COUPLINGS}_incomplete_${STAMP}"
  "$PY" -m tools.materialize_station24_s5_crn_single_seed \
    --raw-results "$ROOT/raw_n500" --tail-results "$ROOT/tail_n100" \
    --output-root "$COUPLINGS" --data-path "$DATA" \
    --generation-seed "$SEED" --duration-seed "$DURATION_SEED" \
    --permutation-seeds "${PERM_SEEDS[@]}" \
    --body-members 400 --background-members 100 --tail-members 100 \
    --edge-taper-hours "$EDGE_TAPER" --energy-score-member-limit "$ENERGY_LIMIT"
else
  echo "COUPLINGS_REUSED root=$COUPLINGS"
fi

declare -A LABELS=(
  [full_v2_mixture]="Full V2 CRN mixture"
  [crn_identity]="Original S5 full-path CRN"
  [permutation_1]="Original S5 permutation 1"
  [permutation_2]="Original S5 permutation 2"
  [permutation_3]="Original S5 permutation 3"
)
BASELINE="$ROOT/raw_n500"
BASELINE_VARIANT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["run"]["condition_variant"])' "$BASELINE/metrics.json")

PHASE=complete_evaluation
for ARM in full_v2_mixture crn_identity permutation_1 permutation_2 permutation_3; do
  RESULT="$COUPLINGS/$ARM"
  POST="$ROOT/evaluation_$ARM"
  mkdir -p "$POST"
  CANDIDATE_VARIANT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["run"]["condition_variant"])' "$RESULT/metrics.json")
  if [[ ! -s "$POST/comparisons/comparison_report.md" ]]; then
    [[ ! -e "$POST/comparisons" ]] || mv "$POST/comparisons" "$POST/comparisons_incomplete_${STAMP}"
    "$PY" tools/compare_station24_multiscale_2a.py "$BASELINE" "$RESULT" \
      --output-dir "$POST/comparisons" --baseline-variant "$BASELINE_VARIANT" \
      --candidate-variant "$CANDIDATE_VARIANT" --baseline-parallel-levels encoder_0 \
      --candidate-parallel-levels encoder_0 --candidate-spatial-levels bottleneck \
      --baseline-label "Raw CRN" --candidate-label "${LABELS[$ARM]}" \
      --title "Raw CRN vs ${LABELS[$ARM]}"
  fi
  if [[ ! -s "$POST/continuous_event_evaluation/continuous_event_member_matches.csv" ]]; then
    [[ ! -e "$POST/continuous_event_evaluation" ]] || mv "$POST/continuous_event_evaluation" "$POST/continuous_event_evaluation_incomplete_${STAMP}"
    "$PY" tools/evaluate_station24_jstd_events.py \
      --baseline "$BASELINE" --candidate "$RESULT" --candidate-run "$TAIL_RUN" \
      --data-path "$DATA" --output-dir "$POST/continuous_event_evaluation" \
      --baseline-label "Raw CRN" --candidate-label "${LABELS[$ARM]}"
  fi
  if [[ ! -s "$POST/joint_wind_solar_evaluation/ordinary_comparison.csv" ]]; then
    [[ ! -e "$POST/joint_wind_solar_evaluation" ]] || mv "$POST/joint_wind_solar_evaluation" "$POST/joint_wind_solar_evaluation_incomplete_${STAMP}"
    "$PY" tools/evaluate_station24_diffusion_ts.py \
      --baseline "$BASELINE" --candidate "$RESULT" --data "$DATA" \
      --output "$POST/joint_wind_solar_evaluation" --candidate-label "${LABELS[$ARM]}"
  fi
  if [[ ! -s "$POST/extreme_wind_tail/extreme_wind_tail_summary.json" ]]; then
    [[ ! -e "$POST/extreme_wind_tail" ]] || mv "$POST/extreme_wind_tail" "$POST/extreme_wind_tail_incomplete_${STAMP}"
    "$PY" tools/plot_station24_extreme_tail.py \
      --baseline "$BASELINE" --candidate "$RESULT" --data-path "$DATA" \
      --output-dir "$POST/extreme_wind_tail" --top-issues 5 \
      --baseline-label "Raw CRN" --candidate-label "${LABELS[$ARM]}"
  fi
  if [[ ! -s "$POST/wind_event_timing/timing_diagnostics.md" ]]; then
    [[ ! -e "$POST/wind_event_timing" ]] || mv "$POST/wind_event_timing" "$POST/wind_event_timing_incomplete_${STAMP}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
      "$PY" tools/diagnose_station24_wind_event_timing.py "$BASELINE" "$RESULT" \
        --data-path "$DATA" --output-dir "$POST/wind_event_timing" \
        --baseline-variant "$BASELINE_VARIANT" --candidate-variant "$CANDIDATE_VARIANT" \
        --baseline-label "Raw CRN" --candidate-label "${LABELS[$ARM]}"
  fi
  if ! compgen -G "$POST/representative_joint_plots/*.png" >/dev/null; then
    "$PY" tools/plot_station24_independent_tail_mixture.py \
      --result "$RESULT" --data-path "$DATA" \
      --output-dir "$POST/representative_joint_plots" --top-issues 5
  fi
  "$PY" tools/summarize_station24_independent_tail_v2.py \
    --baseline "$BASELINE" --candidate "$RESULT" \
    --event-dir "$POST/continuous_event_evaluation" \
    --joint-dir "$POST/joint_wind_solar_evaluation" \
    --output "$POST/RESULT_SUMMARY.md" --candidate-label "${LABELS[$ARM]}"
done

PHASE=member_attribution
for ARM in crn_identity permutation_1 permutation_2 permutation_3; do
  OUT="$ROOT/attribution_$ARM"
  if [[ ! -s "$OUT/s5_member_attribution_summary.json" ]]; then
    [[ ! -e "$OUT" ]] || mv "$OUT" "${OUT}_incomplete_${STAMP}"
    "$PY" -m tools.audit_station24_s5_member_attribution \
      --full-mixture "$COUPLINGS/full_v2_mixture" --tail100 "$ROOT/tail_n100" \
      --full-event-evaluation "$ROOT/evaluation_full_v2_mixture/continuous_event_evaluation" \
      --localized "$COUPLINGS/$ARM" \
      --localized-event-evaluation "$ROOT/evaluation_$ARM/continuous_event_evaluation" \
      --event-targets "$TAIL_RUN/jstd_event_targets.json" \
      --full-label "Full V2 CRN mixture" --localized-label "${LABELS[$ARM]}" \
      --output-dir "$OUT"
  fi
done

PHASE=unified_summary
if [[ ! -s "$ROOT/summary/CRN_SINGLE_SEED_REPORT.md" ]]; then
  [[ ! -e "$ROOT/summary" ]] || mv "$ROOT/summary" "$ROOT/summary_incomplete_${STAMP}"
  "$PY" -m tools.summarize_station24_s5_crn_single_seed \
    --root "$ROOT" --data-path "$DATA" --generation-seed "$SEED"
fi

PHASE=archive
write_status complete
REPORT_ARCHIVE="${ROOT}_reports_${STAMP}.tar.gz"
tar -czf "$REPORT_ARCHIVE" "$ROOT/STATUS.env" "$ROOT/EXPERIMENT_MANIFEST.json" \
  "$ROOT/config_used.yaml" "$ROOT/summary" "$ROOT"/attribution_* "$ROOT"/evaluation_*
test -s "$REPORT_ARCHIVE"
COMPLETE_ARCHIVE="${ROOT}_completed_${STAMP}.tar.gz"
tar -czf "$COMPLETE_ARCHIVE" "$ROOT"
test -s "$COMPLETE_ARCHIVE"
PHASE=complete
write_status complete
echo "S5_CRN_SINGLE_SEED_COMPLETE root=$ROOT"
echo "REPORT_ARCHIVE=$REPORT_ARCHIVE"
echo "COMPLETE_ARCHIVE=$COMPLETE_ARCHIVE"
echo "STOPPED_AFTER_SINGLE_SEED training=NOT_RUN test=NOT_RUN additional_seed=NOT_RUN"
