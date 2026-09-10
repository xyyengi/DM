#!/usr/bin/env bash
# Resume only evaluation/plots/report/archive after training, generation and merge succeeded.
set -euo pipefail

PIPELINE_ROOT=${1:?"usage: $0 PIPELINE_ROOT [RAW_BODY_RESULT]"}
RAW_BODY_RESULT=${2:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}
cd /root/autodl-tmp/DM
PY=/root/miniconda3/envs/dm_env/bin/python
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

RUN_DIR=$(cat "$PIPELINE_ROOT/train_run.txt")
MIXTURE="$PIPELINE_ROOT/mixture_body400_tail100_n500"
test -f "$RUN_DIR/checkpoints/model_best.pt"
test -f "$MIXTURE/metrics.json"
test -f "$MIXTURE/actual_scenarios_normalized.npy"
test -f "$RAW_BODY_RESULT/metrics.json"

JOB=$(date +%Y%m%d_%H%M%S)
POST="$PIPELINE_ROOT/postprocess_resume_$JOB"
mkdir -p "$POST"

"$PY" tools/compare_station24_multiscale_2a.py "$RAW_BODY_RESULT" "$MIXTURE" \
  --output-dir "$POST/comparisons" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant independent_joint_tail_v2_event_balanced_mixture \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-label "Raw body-tail" --candidate-label "Event-balanced independent joint tail V2" \
  --candidate-spatial-levels bottleneck --title "Raw body-tail vs event-balanced independent joint tail V2"
"$PY" tools/evaluate_station24_jstd_events.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" --candidate-run "$RUN_DIR" \
  --output-dir "$POST/continuous_event_evaluation" \
  --baseline-label "Raw body-tail" --candidate-label "Independent joint tail V2"
"$PY" tools/evaluate_station24_diffusion_ts.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" --data diffusion_input_station \
  --output "$POST/joint_wind_solar_evaluation" --candidate-label "Independent joint tail V2"
"$PY" tools/plot_station24_extreme_tail.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" --data-path diffusion_input_station \
  --output-dir "$POST/extreme_wind_tail" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "Independent joint tail V2"
"$PY" tools/diagnose_station24_wind_event_timing.py "$RAW_BODY_RESULT" "$MIXTURE" \
  --output-dir "$POST/wind_event_timing" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant independent_joint_tail_v2_event_balanced_mixture \
  --baseline-label "Raw body-tail" --candidate-label "Independent joint tail V2"
"$PY" tools/plot_station24_independent_tail_mixture.py \
  --result "$MIXTURE" --data-path diffusion_input_station \
  --output-dir "$POST/representative_joint_plots" --top-issues 5
"$PY" tools/summarize_station24_independent_tail_v2.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" \
  --event-dir "$POST/continuous_event_evaluation" \
  --joint-dir "$POST/joint_wind_solar_evaluation" --output "$POST/RESULT_SUMMARY.md"

ARCHIVE="${PIPELINE_ROOT}_resumed_${JOB}.tar.gz"
tar -czf "$ARCHIVE" "$PIPELINE_ROOT"
echo "INDEPENDENT_JOINT_TAIL_V2_RESUME_COMPLETE postprocess=$POST"
echo "ARCHIVE=$ARCHIVE"
