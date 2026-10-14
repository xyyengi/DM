#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/DM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dm_env
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT=${1:-outputs_shandong/station24/s5_full_path_crn_attribution_${STAMP}}
DATA=${DATA_PATH:-diffusion_input_station}
RAW_RUN=${RAW_RUN:-outputs_shandong/station24/body_tail_moe_20260824_191036/training/20260824_191044_station24_body_tail_moe_20260824_191036_seed2027}
TAIL_RUN=${TAIL_RUN:-outputs_shandong/station24/independent_joint_tail_v2_event_balanced_20260910_200653/training/20260910_200730_independent_joint_tail_v2_event_balanced_seed2027_seed2027}
SEEDS=(271828 314159 161803)
LOG=logs/station24/s5_crn_attribution_${STAMP}.log
STATUS="$ROOT/STATUS.env"
PHASE=initializing

if [[ -e "$ROOT" && ! -s "$ROOT/EXPERIMENT_MANIFEST.json" ]]; then
  echo "existing root lacks the frozen CRN manifest; refusing unsafe resume: $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT" "$(dirname "$LOG")"
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
trap 'code=$?; write_status failed; echo "S5_CRN_ATTRIBUTION_FAILED phase=$PHASE code=$code"; exit $code' ERR
write_status running

PHASE=immutable_inputs
for path in \
  "$RAW_RUN/config_used.yaml" "$RAW_RUN/checkpoints/model_best.pt" \
  "$TAIL_RUN/config_used.yaml" "$TAIL_RUN/checkpoints/model_best.pt" \
  "$TAIL_RUN/jstd_event_targets.json" \
  "$DATA/val_forecast.npy" "$DATA/val_actual.npy" "$DATA/station_order.csv"; do
  test -s "$path"
done
AVAILABLE_KB=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
REQUIRED_KB=$((14 * 1024 * 1024))
echo "DISK_PREFLIGHT available_kb=$AVAILABLE_KB required_kb=$REQUIRED_KB"
if (( AVAILABLE_KB < REQUIRED_KB )); then
  echo "at least 14 GiB free space is required for three CRN seeds and both archives" >&2
  exit 1
fi

if [[ ! -s "$ROOT/EXPERIMENT_MANIFEST.json" ]]; then
cat > "$ROOT/EXPERIMENT_MANIFEST.json" <<EOF
{
  "protocol": "docs/station24_s5_crn_experiment_protocol.md",
  "git_commit": "$(git rev-parse HEAD)",
  "generation_seeds": [271828, 314159, 161803],
  "duration_seed": 424242,
  "independent_permutation_seed": 424242,
  "raw_run": "$RAW_RUN",
  "tail_run": "$TAIL_RUN",
  "split": "val",
  "test_locked": true,
  "training": "NOT APPLICABLE",
  "s5_rule_changed": false,
  "launch_eligibility_requires_cuda_preflight": true
}
EOF
else
  echo "CRN_MANIFEST_REUSED root=$ROOT"
fi

PHASE=cpu_regression
bash -n run_station24_s5_crn_attribution.sh
bash -n run_station24_s5_crn_finalize.sh
python generate_station24.py --help >/dev/null
python -m tools.audit_station24_s5_crn_preflight --help >/dev/null
python -m tools.materialize_station24_s5_crn_couplings --help >/dev/null
python -m tools.evaluate_station24_jstd_events --help >/dev/null
python -m tools.evaluate_station24_diffusion_ts --help >/dev/null
python -m tools.audit_station24_s5_member_attribution --help >/dev/null
python -m tools.summarize_station24_s5_crn --help >/dev/null
python -m unittest \
  tests.test_station24_crn_sampling \
  tests.test_station24_s5_espc

PHASE=cuda_preflight
if [[ ! -s "$ROOT/preflight_cuda/preflight.json" ]]; then
  if [[ -d "$ROOT/preflight_cuda" ]]; then
    mv "$ROOT/preflight_cuda" "$ROOT/preflight_cuda_incomplete_$(date +%Y%m%d_%H%M%S)"
  fi
  python -m tools.audit_station24_s5_crn_preflight \
    --raw-run "$RAW_RUN" --tail-run "$TAIL_RUN" --data-path "$DATA" \
    --output-dir "$ROOT/preflight_cuda" --seed 271828
else
  echo "CUDA_PREFLIGHT_REUSED root=$ROOT/preflight_cuda"
fi
python - "$ROOT/preflight_cuda/preflight.json" <<'PY'
import json, sys
report=json.load(open(sys.argv[1], encoding="utf-8"))
assert report["launch_eligible"] is True, report
PY

PHASE=cuda_full_sampler_smoke
for SPEC in "raw:$RAW_RUN" "tail:$TAIL_RUN" "tail_replay:$TAIL_RUN"; do
  NAME=${SPEC%%:*}
  RUN=${SPEC#*:}
  OUT="$ROOT/preflight_cuda/${NAME}_smoke_n2"
  if [[ ! -s "$OUT/generation_metadata.json" ]]; then
    if [[ -d "$OUT" ]]; then
      mv "$OUT" "${OUT}_incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
    python generate_station24.py \
      --run-dir "$RUN" --data-path "$DATA" --output-dir "$OUT" \
      --split val --n-samples 2 --seed 271828 --issue-batch-size 1 \
      --member-chunk-size 2 --no-auto-tune-member-chunk --checkpoint-state raw \
      --result-variant "s5_crn_${NAME}_smoke" --energy-score-member-limit 2 \
      --ancestral-noise-mode keyed_full_path --ancestral-noise-seed 271828 \
      --ancestral-member-offset 400
  fi
done
cmp \
  "$ROOT/preflight_cuda/tail_smoke_n2/generated_stochastic_residual_standardized.npy" \
  "$ROOT/preflight_cuda/tail_replay_smoke_n2/generated_stochastic_residual_standardized.npy"
python - "$ROOT/preflight_cuda/raw_smoke_n2/generation_metadata.json" \
  "$ROOT/preflight_cuda/tail_smoke_n2/generation_metadata.json" <<'PY'
import json, sys
raw=json.load(open(sys.argv[1], encoding="utf-8"))
tail=json.load(open(sys.argv[2], encoding="utf-8"))
for obj in (raw, tail):
    assert obj["ancestral_noise_mode"] == "keyed_full_path"
    assert obj["ancestral_noise_seed"] == 271828
    assert obj["ancestral_member_offset"] == 400
    assert obj["future_actual_used_as_generation_condition"] is False
print("CUDA_FULL_SAMPLER_CRN_SMOKE_PASS")
PY

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="$ROOT/seed_$SEED"
  mkdir -p "$SEED_ROOT"
  PHASE="seed_${SEED}_raw_generation"
  if [[ ! -s "$SEED_ROOT/raw_n500/generation_metadata.json" ]]; then
    if [[ -d "$SEED_ROOT/raw_n500" ]]; then
      mv "$SEED_ROOT/raw_n500" "$SEED_ROOT/raw_n500_incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
  python generate_station24.py \
    --run-dir "$RAW_RUN" --data-path "$DATA" \
    --output-dir "$SEED_ROOT/raw_n500" --split val --n-samples 500 \
    --seed "$SEED" --issue-batch-size 1 --member-chunk-size 100 \
    --no-auto-tune-member-chunk --checkpoint-state raw \
    --result-variant "raw_full_path_crn_seed_${SEED}" \
    --energy-score-member-limit 80 \
    --ancestral-noise-mode keyed_full_path \
    --ancestral-noise-seed "$SEED" --ancestral-member-offset 0
  else
    echo "RAW_GENERATION_REUSED seed=$SEED"
  fi

  PHASE="seed_${SEED}_tail_generation"
  if [[ ! -s "$SEED_ROOT/tail_n100/generation_metadata.json" ]]; then
    if [[ -d "$SEED_ROOT/tail_n100" ]]; then
      mv "$SEED_ROOT/tail_n100" "$SEED_ROOT/tail_n100_incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
  python generate_station24.py \
    --run-dir "$TAIL_RUN" --data-path "$DATA" \
    --output-dir "$SEED_ROOT/tail_n100" --split val --n-samples 100 \
    --seed "$SEED" --issue-batch-size 1 --member-chunk-size 100 \
    --no-auto-tune-member-chunk --checkpoint-state raw \
    --result-variant "full_v2_tail_full_path_crn_seed_${SEED}" \
    --energy-score-member-limit 80 \
    --ancestral-noise-mode keyed_full_path \
    --ancestral-noise-seed "$SEED" --ancestral-member-offset 400
  else
    echo "TAIL_GENERATION_REUSED seed=$SEED"
  fi

  python - "$SEED_ROOT/raw_n500/generation_metadata.json" \
    "$SEED_ROOT/tail_n100/generation_metadata.json" "$SEED" <<'PY'
import json, sys
raw=json.load(open(sys.argv[1], encoding="utf-8"))
tail=json.load(open(sys.argv[2], encoding="utf-8"))
seed=int(sys.argv[3])
for obj,n,offset in ((raw,500,0),(tail,100,400)):
    assert obj["split"] == "val" and not obj["test_used"]
    assert obj["generation_seed"] == seed
    assert obj["ancestral_noise_seed"] == seed
    assert obj["ancestral_noise_mode"] == "keyed_full_path"
    assert obj["ancestral_noise_controls_initial_and_all_reverse_steps"] is True
    assert obj["n_samples"] == n and obj["member_chunk_size"] == 100
    assert obj["ancestral_member_offset"] == offset
print("SEED_GENERATION_AUDIT_PASS", seed)
PY
done

PHASE=finalize
CRN_LOG="$LOG" bash run_station24_s5_crn_finalize.sh "$ROOT"
