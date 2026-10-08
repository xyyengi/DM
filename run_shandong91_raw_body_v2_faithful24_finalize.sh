#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/shandong91/raw_body_v2_faithful24.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the completed V2 training directory}"
GENERATION_DIR="${GENERATION_DIR:-${OUTPUT_ROOT}/validation_raw_n500}"
EVALUATION_DIR="${EVALUATION_DIR:-${OUTPUT_ROOT}/validation_evaluation}"
ARCHIVE="${ARCHIVE:-${OUTPUT_ROOT}_completed.tar.gz}"

[[ -f "${OUTPUT_ROOT}/checkpoints/best.pt" ]] || { echo "Missing best checkpoint"; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG"; exit 2; }

if [[ ! -e "$GENERATION_DIR" ]]; then
  python -u generate_shandong91_v2.py \
    --run-dir "$OUTPUT_ROOT" --output-dir "$GENERATION_DIR" --config "$CONFIG" \
    --split validation --n-samples 500 --seed 424242 --member-chunk 2 \
    --method ddpm --inference-steps 500 --checkpoint-state ema --device cuda
fi

python - "$GENERATION_DIR/metadata.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
assert m["model_identifier"] == "shandong91_heterogeneous_raw_body_v2_faithful24"
assert m["split"] == "validation" and m["n_samples"] == 500
assert m["sampler"] == "ddpm" and m["inference_steps"] == 500
assert m["checkpoint_state"] == "ema"
PY

if [[ ! -e "$EVALUATION_DIR" ]]; then
  args=(--result "$GENERATION_DIR" --output-dir "$EVALUATION_DIR")
  if [[ -n "${V1_BASELINE_RESULT:-}" ]]; then args+=(--baseline-result "$V1_BASELINE_RESULT"); fi
  python -u tools/evaluate_shandong91_v2.py "${args[@]}"
fi

python - "$EVALUATION_DIR/metrics.json" <<'PY'
import json, math, sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
assert m["evaluation_contract"] == "strict elementwise effective_mask"
assert math.isclose(m["finite_ratio"], 1.0)
for resource in ("Wind", "Solar", "Load"):
    for key in ("crps_mw", "coverage90", "width90_mw", "interval_score90_mw"):
        assert math.isfinite(m["channels"][resource][key]), (resource, key)
PY

if [[ ! -e "$ARCHIVE" ]]; then
  tar -czf "$ARCHIVE" "$OUTPUT_ROOT"
fi
echo "FINALIZE_COMPLETE=$ARCHIVE"
