#!/usr/bin/env bash
# Idempotent post-training continuation: generation -> merge -> full evaluation -> archive.
set -Eeuo pipefail
ROOT=${1:?'usage: run_station24_lightweight_joint_tail_finalize.sh PIPELINE_ROOT [RAW_BODY_RESULT]'}
BODY_RESULTS=${2:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}
POST_OVERRIDE=${3:-}
CANDIDATE_VARIANT_REQUESTED=${4:-lightweight_joint_tail_v2_fair}
CANDIDATE_LABEL=${5:-Lightweight Joint Tail V2}
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOB=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/lightweight_joint_tail_finalize_${JOB}.log"
STATUS="logs/station24/lightweight_joint_tail_finalize_${JOB}.status"
PHASE=locating_checkpoint
mkdir -p logs/station24
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() {
  code=$?
  write_status failed
  echo "LIGHTWEIGHT_FINALIZE_FAILED phase=$PHASE exit_code=$code root=$ROOT log=$LOG" >&2
  exit "$code"
}
trap on_error ERR
write_status running
test -d "$ROOT"
test -f "$BODY_RESULTS/metrics.json"
if [[ -f "$ROOT/train_run.txt" ]]; then
  RUN_DIR=$(cat "$ROOT/train_run.txt")
else
  RUN_DIR=$(find "$ROOT" -type f -path '*/checkpoints/model_best.pt' -printf '%h\n' | sed 's#/checkpoints$##' | sort | tail -n 1)
  test -n "$RUN_DIR"
  printf '%s\n' "$RUN_DIR" > "$ROOT/train_run.txt"
fi
test -f "$RUN_DIR/checkpoints/model_best.pt"
echo "TRAIN_RUN=$RUN_DIR"
TAIL_RESULTS="$ROOT/independent_tail_n100"
PHASE=generation
write_status running
if [[ -f "$TAIL_RESULTS/metrics.json" && -f "$TAIL_RESULTS/actual_scenarios_normalized.npy" ]]; then
  echo "GENERATION_REUSED result=$TAIL_RESULTS"
else
  if [[ -e "$TAIL_RESULTS" ]]; then
    TAIL_RESULTS="$ROOT/independent_tail_n100_retry_${JOB}"
    echo "INCOMPLETE_GENERATION_PRESERVED new_output=$TAIL_RESULTS"
  fi
  python generate_station24.py --run-dir "$RUN_DIR" --data-path diffusion_input_station --output-dir "$TAIL_RESULTS" --split val --n-samples 100 --seed 424242 --checkpoint-state raw --result-variant "$CANDIDATE_VARIANT_REQUESTED"
fi
MIXTURE="$ROOT/mixture_body400_tail100_n500"
PHASE=merge_body400_tail100
write_status running
if [[ -f "$MIXTURE/metrics.json" && -f "$MIXTURE/actual_scenarios_normalized.npy" ]]; then
  echo "MERGE_REUSED result=$MIXTURE"
else
  if [[ -e "$MIXTURE" ]]; then
    MIXTURE="$ROOT/mixture_body400_tail100_n500_retry_${JOB}"
    echo "INCOMPLETE_MIXTURE_PRESERVED new_output=$MIXTURE"
  fi
  python tools/merge_station24_independent_tail_members.py \
    --body-results "$BODY_RESULTS" --body-member-limit 400 \
    --tail-results "$TAIL_RESULTS" --tail-member-limit 100 \
    --output-dir "$MIXTURE" --data-path diffusion_input_station \
    --energy-score-member-limit 80 \
    --condition-variant "${CANDIDATE_VARIANT_REQUESTED}_mixture" \
    --family fixed_quota_lightweight_joint_tail_v2
fi
BASELINE_VARIANT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1] + "/metrics.json", encoding="utf-8"))["run"]["condition_variant"])' "$BODY_RESULTS")
CANDIDATE_VARIANT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1] + "/metrics.json", encoding="utf-8"))["run"]["condition_variant"])' "$MIXTURE")
test -n "$BASELINE_VARIANT"
test -n "$CANDIDATE_VARIANT"
test "$BASELINE_VARIANT" != "$CANDIDATE_VARIANT"
echo "EVALUATION_VARIANTS baseline=$BASELINE_VARIANT candidate=$CANDIDATE_VARIANT"
if [[ -n "$POST_OVERRIDE" ]]; then
  POST="$POST_OVERRIDE"
else
  POST="$ROOT/postprocess_${JOB}"
fi
PHASE=full_evaluation
write_status running
mkdir -p "$POST"

COMPARISON="$POST/comparisons"
if [[ -f "$COMPARISON/comparison_report.md" ]]; then
  echo "COMPARISON_REUSED output=$COMPARISON"
else
  [[ ! -e "$COMPARISON" ]] || COMPARISON="$POST/comparisons_retry_${JOB}"
  python tools/compare_station24_multiscale_2a.py "$BODY_RESULTS" "$MIXTURE" \
    --output-dir "$COMPARISON" \
    --baseline-variant "$BASELINE_VARIANT" \
    --candidate-variant "$CANDIDATE_VARIANT" \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-label "Raw body-tail" --candidate-label "$CANDIDATE_LABEL" \
    --candidate-spatial-levels bottleneck \
    --title "Raw body-tail vs ${CANDIDATE_LABEL}"
fi

EVENT_DIR="$POST/continuous_event_evaluation"
if [[ -f "$EVENT_DIR/report.md" ]]; then
  echo "EVENT_EVALUATION_REUSED output=$EVENT_DIR"
else
  [[ ! -e "$EVENT_DIR" ]] || EVENT_DIR="$POST/continuous_event_evaluation_retry_${JOB}"
  python tools/evaluate_station24_jstd_events.py \
    --baseline "$BODY_RESULTS" --candidate "$MIXTURE" --candidate-run "$RUN_DIR" \
    --data-path diffusion_input_station --output-dir "$EVENT_DIR" \
    --baseline-label "Raw body-tail" --candidate-label "$CANDIDATE_LABEL"
fi

JOINT_DIR="$POST/joint_wind_solar_evaluation"
if [[ -f "$JOINT_DIR/ordinary_comparison.csv" ]]; then
  echo "JOINT_EVALUATION_REUSED output=$JOINT_DIR"
else
  [[ ! -e "$JOINT_DIR" ]] || JOINT_DIR="$POST/joint_wind_solar_evaluation_retry_${JOB}"
  python tools/evaluate_station24_diffusion_ts.py \
    --baseline "$BODY_RESULTS" --candidate "$MIXTURE" --data diffusion_input_station \
    --output "$JOINT_DIR" --candidate-label "$CANDIDATE_LABEL"
fi

EXTREME_DIR="$POST/extreme_wind_tail"
if [[ -f "$EXTREME_DIR/extreme_wind_tail_summary.json" ]]; then
  echo "EXTREME_AUDIT_REUSED output=$EXTREME_DIR"
else
  [[ ! -e "$EXTREME_DIR" ]] || EXTREME_DIR="$POST/extreme_wind_tail_retry_${JOB}"
  python tools/plot_station24_extreme_tail.py \
    --baseline "$BODY_RESULTS" --candidate "$MIXTURE" --data-path diffusion_input_station \
    --output-dir "$EXTREME_DIR" --top-issues 5 \
    --baseline-label "Raw body-tail" --candidate-label "$CANDIDATE_LABEL"
fi

TIMING_DIR="$POST/wind_event_timing"
if [[ -f "$TIMING_DIR/timing_diagnostics.md" ]]; then
  echo "TIMING_DIAGNOSTIC_REUSED output=$TIMING_DIR"
else
  [[ ! -e "$TIMING_DIR" ]] || TIMING_DIR="$POST/wind_event_timing_retry_${JOB}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    python tools/diagnose_station24_wind_event_timing.py "$BODY_RESULTS" "$MIXTURE" \
      --data-path diffusion_input_station --output-dir "$TIMING_DIR" \
      --baseline-variant "$BASELINE_VARIANT" \
      --candidate-variant "$CANDIDATE_VARIANT" \
      --baseline-label "Raw body-tail" --candidate-label "$CANDIDATE_LABEL"
fi

REPRESENTATIVE_DIR="$POST/representative_joint_plots"
shopt -s nullglob
REPRESENTATIVE_PLOTS=("$REPRESENTATIVE_DIR"/*.png)
shopt -u nullglob
if (( ${#REPRESENTATIVE_PLOTS[@]} > 0 )); then
  echo "REPRESENTATIVE_PLOTS_REUSED output=$REPRESENTATIVE_DIR"
else
  python tools/plot_station24_independent_tail_mixture.py \
    --result "$MIXTURE" --data-path diffusion_input_station \
    --output-dir "$REPRESENTATIVE_DIR" --top-issues 5
fi

python tools/summarize_station24_independent_tail_v2.py \
  --baseline "$BODY_RESULTS" --candidate "$MIXTURE" \
  --event-dir "$EVENT_DIR" --joint-dir "$JOINT_DIR" \
  --output "$POST/RESULT_SUMMARY.md" \
  --candidate-label "$CANDIDATE_LABEL"
test -f "$POST/RESULT_SUMMARY.md"
python -c 'import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); payload={"tail_results":sys.argv[2],"mixture":sys.argv[3],"postprocess":sys.argv[4],"comparison":sys.argv[5],"event_dir":sys.argv[6],"joint_dir":sys.argv[7],"extreme_dir":sys.argv[8],"timing_dir":sys.argv[9],"representative_dir":sys.argv[10],"result_summary":sys.argv[11]}; out.write_text(json.dumps(payload,indent=2),encoding="utf-8")' \
  "$ROOT/finalize_artifacts.json" "$TAIL_RESULTS" "$MIXTURE" "$POST" \
  "$COMPARISON" "$EVENT_DIR" "$JOINT_DIR" "$EXTREME_DIR" "$TIMING_DIR" \
  "$REPRESENTATIVE_DIR" "$POST/RESULT_SUMMARY.md"
if [[ "${SKIP_ARCHIVE:-0}" == "1" ]]; then
  ARCHIVE="SKIPPED_BY_PARENT_PIPELINE"
else
  PHASE=archive
  write_status running
  ARCHIVE="${ROOT}_completed_${JOB}.tar.gz"
  tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  test -s "$ARCHIVE"
fi
PHASE=complete
write_status complete
echo "LIGHTWEIGHT_JOINT_TAIL_FINALIZE_COMPLETE postprocess=$POST"
echo "ARCHIVE=$ARCHIVE"
