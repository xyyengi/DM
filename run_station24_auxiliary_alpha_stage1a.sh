#!/usr/bin/env bash
# Frozen Stage 1A only: two new alphas, reuse alpha=1, then stop.
set -Eeuo pipefail
cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
JOB=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT="outputs_shandong/station24/auxiliary_alpha_sensitivity_${JOB}"
LOG="logs/station24/auxiliary_alpha_stage1a_${JOB}.log"
STATUS="logs/station24/auxiliary_alpha_stage1a_${JOB}.status"
PHASE=initializing
test ! -e "$ROOT" || { echo "refusing to overwrite $ROOT" >&2; exit 2; }
mkdir -p "$ROOT/configs" logs/station24
exec > >(tee -a "$LOG") 2>&1
write_status() {
  printf 'state=%s\nphase=%s\nroot=%s\nlog=%s\nupdated_at=%s\n' "$1" "$PHASE" "$ROOT" "$LOG" "$(date --iso-8601=seconds)" > "$STATUS"
}
on_error() {
  code=$?; write_status failed
  echo "AUXILIARY_ALPHA_STAGE1A_FAILED phase=$PHASE exit_code=$code root=$ROOT log=$LOG" >&2
  exit "$code"
}
trap on_error ERR
write_status running
PHASE=immutable_input_and_disk_gate; write_status running
RAW_RUN="outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027"
RAW_RESULT="outputs_shandong/station24/body_tail_moe_raw_inference_20260824_224151/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242"
CONTROL_ROOT=${CONTROL_ROOT:-outputs_shandong/station24/lightweight_joint_tail_v2_fair_20260923_194256}
CONTROL_RUN=${CONTROL_RUN:-$CONTROL_ROOT/20260923_194315_lightweight_joint_tail_v2_fair_20260923_194256_seed2027}
CONTROL_POST=${CONTROL_POST:-$CONTROL_ROOT/postprocess_20260924_152622}
V2_MIXTURE="outputs_shandong/station24/independent_joint_tail_v2_event_balanced_20260910_200653/mixture_body400_tail100_n500"
for required in \
  "$RAW_RUN/checkpoints/model_best.pt" "$RAW_RUN/graphs/secondary_adjacency.npy" \
  "$RAW_RESULT/metrics.json" "$CONTROL_RUN/checkpoints/model_best.pt" \
  "$CONTROL_ROOT/mixture_body400_tail100_n500/metrics.json" \
  "$CONTROL_POST/continuous_event_evaluation/continuous_event_per_event.csv" \
  "$V2_MIXTURE/metrics.json" diffusion_input_station/station_order.csv \
  docs/auxiliary_hyperparameter_optimization_20261006/FROZEN_HYPERPARAMETER_SELECTION_PROTOCOL.md \
  docs/auxiliary_hyperparameter_optimization_20261006/validation_objectives.md \
  docs/auxiliary_hyperparameter_optimization_20261006/pareto_selection_protocol.md; do
  test -f "$required" || { echo "missing immutable Stage 1A input: $required" >&2; false; }
done
AVAILABLE_KB=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
REQUIRED_KB=$((12 * 1024 * 1024))
echo "DISK_PREFLIGHT available_kb=$AVAILABLE_KB required_kb=$REQUIRED_KB"
test "$AVAILABLE_KB" -ge "$REQUIRED_KB" || { echo "Stage 1A requires at least 12 GiB free" >&2; false; }
git rev-parse HEAD > "$ROOT/git_commit.txt"
git status --short > "$ROOT/git_status_at_launch.txt"
PHASE=server_static_cli_gate; write_status running
bash -n run_station24_auxiliary_alpha_stage1a.sh \
  run_station24_auxiliary_alpha_candidate.sh \
  run_station24_auxiliary_alpha_stage1a_finalize.sh \
  run_station24_lightweight_joint_tail_finalize.sh
python tools/materialize_station24_auxiliary_alpha_config.py --help >/dev/null
python tools/audit_station24_auxiliary_alpha_inputs.py --help >/dev/null
python -m tools.audit_station24_lightweight_joint_tail --help >/dev/null
python tools/check_station24_auxiliary_alpha_launch.py --help >/dev/null
python tools/audit_station24_auxiliary_alpha_run.py --help >/dev/null
python tools/evaluate_station24_auxiliary_alpha_stage1a.py --help >/dev/null
python generate_station24.py --help >/dev/null
python tools/merge_station24_independent_tail_members.py --help >/dev/null
python tools/audit_station24_auxiliary_alpha_inputs.py \
  --raw-result "$RAW_RESULT" \
  --control-result "$CONTROL_ROOT/mixture_body400_tail100_n500" \
  --control-run "$CONTROL_RUN" --control-post "$CONTROL_POST" \
  --raw-checkpoint "$RAW_RUN/checkpoints/model_best.pt" \
  --v2-mixture "$V2_MIXTURE" --output "$ROOT/immutable_input_audit.json"
PHASE=materialize_frozen_configs; write_status running
python tools/materialize_station24_auxiliary_alpha_config.py --alpha 0.65 \
  --output "$ROOT/configs/alpha_065.yaml" --audit "$ROOT/configs/alpha_065_audit.json"
python tools/materialize_station24_auxiliary_alpha_config.py --alpha 1.35 \
  --output "$ROOT/configs/alpha_135.yaml" --audit "$ROOT/configs/alpha_135_audit.json"
python -c 'import hashlib,json,pathlib,sys; files=sys.argv[1:]; print(json.dumps({str(p):hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in files},indent=2))' \
  docs/auxiliary_hyperparameter_optimization_20261006/FROZEN_HYPERPARAMETER_SELECTION_PROTOCOL.md \
  docs/auxiliary_hyperparameter_optimization_20261006/validation_objectives.md \
  docs/auxiliary_hyperparameter_optimization_20261006/pareto_selection_protocol.md \
  docs/auxiliary_hyperparameter_optimization_20261006/AUXILIARY_HYPERPARAMETER_OPTIMIZATION_PLAN.md \
  > "$ROOT/frozen_protocol_hashes.json"
PHASE=train_alpha_065; write_status running
bash run_station24_auxiliary_alpha_candidate.sh "$ROOT/configs/alpha_065.yaml" "$ROOT/alpha_065" 0.65
PHASE=train_alpha_135; write_status running
bash run_station24_auxiliary_alpha_candidate.sh "$ROOT/configs/alpha_135.yaml" "$ROOT/alpha_135" 1.35
PHASE=post_training_finalize; write_status running
bash run_station24_auxiliary_alpha_stage1a_finalize.sh "$ROOT"
PHASE=complete; write_status complete
echo "AUXILIARY_ALPHA_STAGE1A_FORMAL_COMPLETE root=$ROOT"

