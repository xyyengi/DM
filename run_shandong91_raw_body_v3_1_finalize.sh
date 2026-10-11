#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/shandong91/raw_body_v3_1_factor_coordinate_loss.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the completed V3.1 training directory}"
GENERATION_DIR="${GENERATION_DIR:-${OUTPUT_ROOT}/validation_raw_n500}"
EVALUATION_DIR="${EVALUATION_DIR:-${OUTPUT_ROOT}/validation_evaluation}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"
ARCHIVE="${ARCHIVE:-${OUTPUT_ROOT}_completed.tar.gz}"
STATUS_FILE="${OUTPUT_ROOT}_status.json"
PHASE="finalize_startup"

record_status() {
  python - "$STATUS_FILE" "$1" "$PHASE" "$OUTPUT_ROOT" <<'PY'
import json,pathlib,sys,time
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "state":sys.argv[2],"phase":sys.argv[3],"root":sys.argv[4],"timestamp":time.time()
},indent=2))
PY
}
trap 'code=$?; record_status "failed"; exit "$code"' ERR

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG"; exit 2; }
[[ -f "${OUTPUT_ROOT}/checkpoints/best.pt" ]] || { echo "Missing best checkpoint"; exit 2; }
MEMBER_CHUNK="$(python - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["generation"]["member_chunk"])
PY
)"

PHASE="generation"; record_status "running"
if [[ ! -f "$GENERATION_DIR/metadata.json" ]]; then
  python -u generate_shandong91_v3.py --run-dir "$OUTPUT_ROOT" --output-dir "$GENERATION_DIR" \
    --config "$CONFIG" --split validation --n-samples 500 --seed 424242 \
    --member-chunk "$MEMBER_CHUNK" --method ddpm --inference-steps 500 \
    --checkpoint-state ema --device cuda --resume-output
fi
python - "$GENERATION_DIR/metadata.json" <<'PY'
import json,math,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
assert m["model_identifier"]=="shandong91_heterogeneous_raw_body_v3_1_factor_coordinate_loss"
assert m["formal_objective"]=="factor_coordinate_plus_projected_local_mean"
assert m["split"]=="validation" and m["n_samples"]==500
assert m["sampler"]=="ddpm" and m["inference_steps"]==500
assert m["load_local_branch"]=="disabled_strict_rank1"
PY

PHASE="evaluation"; record_status "running"
if [[ ! -e "$EVALUATION_DIR" ]]; then
  EXTRA=()
  if [[ -n "${BASELINE_RESULT:-}" ]]; then EXTRA=(--baseline-result "$BASELINE_RESULT"); fi
  python -u tools/evaluate_shandong91_v3.py --result "$GENERATION_DIR" \
    --output-dir "$EVALUATION_DIR" "${EXTRA[@]}"
fi
python - "$EVALUATION_DIR/metrics.json" <<'PY'
import json,math,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); assert math.isclose(m["finite_ratio"],1.0)
for r in ("Wind","Solar","Load"):
    for k in ("coverage90","crps_mw","width90_mw","interval_score90_mw","ensemble_conditional_std_mw"):
        assert math.isfinite(m["channels"][r][k]),(r,k)
    for k in ("coverage90","crps_mw","width90_mw","interval_score90_mw"):
        assert math.isfinite(m["system_aggregate"][r][k]),(r,k)
    assert math.isfinite(m["system_aggregate"][r]["covariance_amplification"]["mean"])
    assert all(math.isfinite(m["channels"][r]["ramp"][lag]["crps_mw"]) for lag in ("1","3","6"))
assert math.isfinite(m["joint"]["system_resource_energy_score_mw"])
PY

PHASE="plots_summary"; record_status "running"
if [[ ! -e "$PLOT_DIR" ]]; then
  python -u tools/plot_shandong91_v3.py --run-dir "$OUTPUT_ROOT" \
    --result "$GENERATION_DIR" --output-dir "$PLOT_DIR"
fi
[[ -s "$EVALUATION_DIR/RESULT_SUMMARY.md" ]]
[[ -s "$PLOT_DIR/training_curve.png" && -s "$PLOT_DIR/representative_system_scenarios.png" ]]

PHASE="archive"; record_status "running"
if [[ ! -e "$ARCHIVE" ]]; then
  TEMP_ARCHIVE="${ARCHIVE}.tmp"
  tar -czf "$TEMP_ARCHIVE" "$OUTPUT_ROOT"
  mv "$TEMP_ARCHIVE" "$ARCHIVE"
fi
[[ -s "$ARCHIVE" ]]
PHASE="complete"; record_status "complete"
trap - ERR
echo "V3_1_FINALIZE_COMPLETE=$ARCHIVE"
