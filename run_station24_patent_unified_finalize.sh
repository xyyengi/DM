#!/usr/bin/env bash
# Idempotent post-training path for the patent handoff comparison.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

ROOT=${1:?'usage: run_station24_patent_unified_finalize.sh PIPELINE_ROOT'}
RAW_RUN=${2:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027}
JOB=$(date +%Y%m%d_%H%M%S)
LOG="logs/station24/patent_unified_finalize_${JOB}.log"
STATUS="$ROOT/FINALIZE_STATUS.env"
PHASE=locating_inputs
mkdir -p logs/station24 "$ROOT"
exec > >(tee -a "$LOG") 2>&1
write_status() { printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"; }
on_error() { code=$?; write_status failed; echo "PATENT_UNIFIED_FINALIZE_FAILED phase=$PHASE code=$code" >&2; exit "$code"; }
trap on_error ERR
write_status running

TRAIN_RUN=$(cat "$ROOT/train_run.txt")
test -f "$TRAIN_RUN/checkpoints/model_best.pt"
test -f "$RAW_RUN/checkpoints/model_best.pt"
RAW="$ROOT/raw_n500_seed424242"
TAIL="$ROOT/lightweight_tail_n100_seed424242"
MIX20="$ROOT/mixture_raw400_tail100_n500"
MIX10="$ROOT/mixture_raw450_tail50_n500"

PHASE=raw_generation
write_status running
if [[ ! -f "$RAW/metrics.json" ]]; then
  [[ ! -e "$RAW" ]] || mv "$RAW" "${RAW}_incomplete_${JOB}"
  python generate_station24.py --run-dir "$RAW_RUN" --data-path diffusion_input_station \
    --output-dir "$RAW" --split val --n-samples 500 --seed 424242 \
    --checkpoint-state raw --tail-route-probability 0 \
    --energy-score-member-limit 80 --result-variant patent_unified_pure_raw
fi

PHASE=tail_generation
write_status running
if [[ ! -f "$TAIL/metrics.json" ]]; then
  [[ ! -e "$TAIL" ]] || mv "$TAIL" "${TAIL}_incomplete_${JOB}"
  python generate_station24.py --run-dir "$TRAIN_RUN" --data-path diffusion_input_station \
    --output-dir "$TAIL" --split val --n-samples 100 --seed 424242 \
    --checkpoint-state raw --energy-score-member-limit 80 \
    --result-variant patent_unified_lightweight_tail
fi

merge_once() {
  local body_n=$1 tail_n=$2 output=$3 variant=$4
  if [[ ! -f "$output/metrics.json" ]]; then
    [[ ! -e "$output" ]] || mv "$output" "${output}_incomplete_${JOB}"
    python tools/merge_station24_independent_tail_members.py \
      --body-results "$RAW" --body-member-limit "$body_n" \
      --tail-results "$TAIL" --tail-member-limit "$tail_n" \
      --output-dir "$output" --data-path diffusion_input_station \
      --energy-score-member-limit 80 --condition-variant "$variant" \
      --family patent_unified_fixed_quota_lightweight_tail
  fi
}
PHASE=member_merge
write_status running
merge_once 400 100 "$MIX20" patent_unified_raw400_tail100
merge_once 450 50 "$MIX10" patent_unified_raw450_tail50

evaluate_one() {
  local candidate=$1 tag=$2 label=$3 post="$ROOT/evaluation_${tag}"
  mkdir -p "$post"
  if [[ ! -f "$post/continuous_event_evaluation/report.md" ]]; then
    [[ ! -e "$post/continuous_event_evaluation" ]] || mv "$post/continuous_event_evaluation" "$post/events_incomplete_${JOB}"
    python tools/evaluate_station24_jstd_events.py --baseline "$RAW" --candidate "$candidate" \
      --candidate-run "$TRAIN_RUN" --data-path diffusion_input_station \
      --output-dir "$post/continuous_event_evaluation" --baseline-label "Pure Raw" --candidate-label "$label"
  fi
  if [[ ! -f "$post/joint_wind_solar_evaluation/ordinary_comparison.csv" ]]; then
    [[ ! -e "$post/joint_wind_solar_evaluation" ]] || mv "$post/joint_wind_solar_evaluation" "$post/joint_incomplete_${JOB}"
    python tools/evaluate_station24_diffusion_ts.py --baseline "$RAW" --candidate "$candidate" \
      --data diffusion_input_station --output "$post/joint_wind_solar_evaluation" --candidate-label "$label"
  fi
  if [[ ! -f "$post/comparisons/comparison_report.md" ]]; then
    [[ ! -e "$post/comparisons" ]] || mv "$post/comparisons" "$post/comparisons_incomplete_${JOB}"
    python tools/compare_station24_multiscale_2a.py "$RAW" "$candidate" --output-dir "$post/comparisons" \
      --baseline-variant patent_unified_pure_raw --candidate-variant "patent_unified_${tag}" \
      --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
      --baseline-label "Pure Raw" --candidate-label "$label" --candidate-spatial-levels bottleneck \
      --title "Pure Raw vs ${label}"
  fi
  python tools/summarize_station24_independent_tail_v2.py --baseline "$RAW" --candidate "$candidate" \
    --event-dir "$post/continuous_event_evaluation" --joint-dir "$post/joint_wind_solar_evaluation" \
    --output "$post/RESULT_SUMMARY.md" --candidate-label "$label"
}
PHASE=complete_evaluation
write_status running
evaluate_one "$MIX20" raw400_tail100 "Raw400 + Lightweight Tail100"
evaluate_one "$MIX10" raw450_tail50 "Raw450 + Lightweight Tail50"

PHASE=summary
write_status running
python -m tools.summarize_station24_patent_unified --root "$ROOT" --raw-run "$RAW_RUN" --train-run "$TRAIN_RUN"
test -f "$ROOT/RUN_SUMMARY.md"

PHASE=archive
write_status running
ARCHIVE="${ROOT}_completed_${JOB}.tar.gz"
if [[ ! -s "$ARCHIVE" ]]; then tar -czf "$ARCHIVE" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"; fi
test -s "$ARCHIVE"
PHASE=complete
write_status complete
echo "PATENT_UNIFIED_COMPLETE root=$ROOT archive=$ARCHIVE"
