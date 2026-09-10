#!/usr/bin/env bash
# Full 400-body + 100-independent-tail evaluation. Run only after preflight.
set -euo pipefail

PIPELINE_ROOT=${1:?"usage: $0 PIPELINE_ROOT"}
RAW_CHECKPOINT=${2:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/checkpoints/model_best.pt}
SECONDARY_GRAPH=${3:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027/graphs/secondary_adjacency.npy}
RAW_BODY_RESULT=${4:-outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
cd /root/autodl-tmp/DM
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
CONFIG=configs/station24_independent_joint_tail_v1_168h.yaml
# Formal continuation requires an explicit reviewed development decision.
RUN_DIR=$(cat "$PIPELINE_ROOT/train_run.txt")
CHECKPOINT="$RUN_DIR/checkpoints/model_best.pt"
test -f "$CHECKPOINT"
python tools/check_station24_independent_tail_gate.py --pipeline-root "$PIPELINE_ROOT"

TAIL100="$PIPELINE_ROOT/independent_tail_n100"
python generate_station24.py --run-dir "$RUN_DIR" \
  --data-path diffusion_input_station --output-dir "$TAIL100" \
  --split val --n-samples 100 --seed 424242 --checkpoint-state raw \
  --result-variant independent_joint_tail_v1_raw

MIXTURE="$PIPELINE_ROOT/mixture_body400_tail100_n500"
python tools/merge_station24_independent_tail_members.py \
  --body-results "$RAW_BODY_RESULT" --body-member-limit 400 --tail-results "$TAIL100" --output-dir "$MIXTURE" \
  --data-path diffusion_input_station --energy-score-member-limit 80

python tools/evaluate_station24_jstd_events.py --candidate "$MIXTURE" --candidate-run "$RUN_DIR" \
  --baseline "$RAW_BODY_RESULT" --output-dir "$PIPELINE_ROOT/continuous_event_evaluation"
python tools/compare_station24_multiscale_2a.py "$RAW_BODY_RESULT" "$MIXTURE" \
  --output-dir "$PIPELINE_ROOT/comparisons/raw_body_vs_independent_joint_tail" \
  --baseline-variant geo_history_actual_body_tail_moe_raw --candidate-variant independent_joint_tail_v1_mixture \
  --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
  --baseline-label "Raw baseline" --candidate-label "Independent joint tail mixture"
python tools/diagnose_station24_wind_event_timing.py "$RAW_BODY_RESULT" "$MIXTURE" \
  --output-dir "$PIPELINE_ROOT/wind_event_timing/raw_body_vs_independent_joint_tail" \
  --baseline-variant geo_history_actual_body_tail_moe_raw --candidate-variant independent_joint_tail_v1_mixture \
  --baseline-label "Raw baseline" --candidate-label "Independent joint tail mixture"

tar -czf "${PIPELINE_ROOT}_completed.tar.gz" "$PIPELINE_ROOT"
echo "INDEPENDENT_JOINT_TAIL_FORMAL_COMPLETE root=$PIPELINE_ROOT"
echo "ARCHIVE=${PIPELINE_ROOT}_completed.tar.gz"
