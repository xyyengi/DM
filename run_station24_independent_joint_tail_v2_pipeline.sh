#!/usr/bin/env bash
# One-command paid-server pipeline: gate, train, generate, merge, evaluate, plot, report, archive.
set -euo pipefail

PIPELINE_ROOT=${1:?"usage: $0 PIPELINE_ROOT [RAW_CHECKPOINT] [SECONDARY_GRAPH] [RAW_BODY_RESULT]"}
RAW_CHECKPOINT=${2:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/checkpoints/model_best.pt}
SECONDARY_GRAPH=${3:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/graphs/secondary_adjacency.npy}
RAW_BODY_RESULT=${4:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}

cd /root/autodl-tmp/DM
PY=/root/miniconda3/envs/dm_env/bin/python
CONFIG=configs/station24_independent_joint_tail_v2_event_balanced_168h.yaml
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

test ! -e "$PIPELINE_ROOT"
test -x "$PY"
test -f "$RAW_CHECKPOINT"
test -f "$SECONDARY_GRAPH"
test -f "$RAW_BODY_RESULT/metrics.json"
mkdir -p "$PIPELINE_ROOT"
"$PY" -c 'import pandas, torch; print("ENV_OK", pandas.__version__, torch.__version__, torch.cuda.is_available(), flush=True)'

"$PY" tools/audit_station24_independent_joint_tail_preflight.py \
  --config "$CONFIG" --data-path diffusion_input_station \
  --checkpoint "$RAW_CHECKPOINT" --secondary-adjacency "$SECONDARY_GRAPH" \
  --output "$PIPELINE_ROOT/cuda_preflight.json"
"$PY" tools/check_station24_independent_tail_gate.py --pipeline-root "$PIPELINE_ROOT" --pretrain

TRAIN_ROOT="$PIPELINE_ROOT/training"
"$PY" train_station24.py \
  --config "$CONFIG" --data-path diffusion_input_station \
  --output-root "$TRAIN_ROOT" --exp-name independent_joint_tail_v2_event_balanced_seed2027 \
  --secondary-adjacency "$SECONDARY_GRAPH" --initialize-checkpoint "$RAW_CHECKPOINT"

mapfile -t CHECKPOINTS < <(find "$TRAIN_ROOT" -type f -path '*/checkpoints/model_best.pt')
[[ ${#CHECKPOINTS[@]} -eq 1 ]]
CHECKPOINT="${CHECKPOINTS[0]}"
RUN_DIR="${CHECKPOINT%/checkpoints/model_best.pt}"
printf '%s\n' "$RUN_DIR" > "$PIPELINE_ROOT/train_run.txt"

TAIL100="$PIPELINE_ROOT/independent_tail_n100"
"$PY" generate_station24.py --run-dir "$RUN_DIR" --data-path diffusion_input_station \
  --output-dir "$TAIL100" --split val --n-samples 100 --seed 424242 \
  --checkpoint-state raw --result-variant independent_joint_tail_v2_event_balanced_raw

MIXTURE="$PIPELINE_ROOT/mixture_body400_tail100_n500"
"$PY" tools/merge_station24_independent_tail_members.py \
  --body-results "$RAW_BODY_RESULT" --body-member-limit 400 \
  --tail-results "$TAIL100" --tail-member-limit 100 \
  --output-dir "$MIXTURE" --data-path diffusion_input_station \
  --energy-score-member-limit 80 \
  --condition-variant independent_joint_tail_v2_event_balanced_mixture \
  --family fixed_quota_independent_joint_tail_v2_event_balanced

"$PY" tools/compare_station24_multiscale_2a.py "$RAW_BODY_RESULT" "$MIXTURE" \
  --output-dir "$PIPELINE_ROOT/comparisons/raw_body_vs_independent_joint_tail_v2" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant independent_joint_tail_v2_event_balanced_mixture \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-label "Raw body-tail" --candidate-label "Event-balanced independent joint tail V2" \
  --candidate-spatial-levels bottleneck --title "Raw body-tail vs event-balanced independent joint tail V2"

"$PY" tools/evaluate_station24_jstd_events.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" --candidate-run "$RUN_DIR" \
  --output-dir "$PIPELINE_ROOT/continuous_event_evaluation" \
  --baseline-label "Raw body-tail" --candidate-label "Independent joint tail V2"
"$PY" tools/evaluate_station24_diffusion_ts.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" --data diffusion_input_station \
  --output "$PIPELINE_ROOT/joint_wind_solar_evaluation" \
  --candidate-label "Independent joint tail V2"
"$PY" tools/plot_station24_extreme_tail.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" --data-path diffusion_input_station \
  --output-dir "$PIPELINE_ROOT/extreme_wind_tail" --top-issues 5 \
  --baseline-label "Raw body-tail" --candidate-label "Independent joint tail V2"
"$PY" tools/diagnose_station24_wind_event_timing.py "$RAW_BODY_RESULT" "$MIXTURE" \
  --output-dir "$PIPELINE_ROOT/wind_event_timing" \
  --baseline-variant geo_history_actual_body_tail_moe_raw \
  --candidate-variant independent_joint_tail_v2_event_balanced_mixture \
  --baseline-label "Raw body-tail" --candidate-label "Independent joint tail V2"
"$PY" tools/plot_station24_independent_tail_mixture.py \
  --result "$MIXTURE" --data-path diffusion_input_station \
  --output-dir "$PIPELINE_ROOT/representative_joint_plots" --top-issues 5
"$PY" tools/summarize_station24_independent_tail_v2.py \
  --baseline "$RAW_BODY_RESULT" --candidate "$MIXTURE" \
  --event-dir "$PIPELINE_ROOT/continuous_event_evaluation" \
  --joint-dir "$PIPELINE_ROOT/joint_wind_solar_evaluation" \
  --output "$PIPELINE_ROOT/RESULT_SUMMARY.md"

tar -czf "${PIPELINE_ROOT}_completed.tar.gz" "$PIPELINE_ROOT"
echo "INDEPENDENT_JOINT_TAIL_V2_COMPLETE root=$PIPELINE_ROOT"
echo "ARCHIVE=${PIPELINE_ROOT}_completed.tar.gz"
