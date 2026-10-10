#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${CONFIG:-configs/shandong91/raw_body_v3_low_rank_common_factor.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to the completed V3 training directory}"
GENERATION_DIR="${GENERATION_DIR:-${OUTPUT_ROOT}/validation_raw_n500}"
EVALUATION_DIR="${EVALUATION_DIR:-${OUTPUT_ROOT}/validation_evaluation}"
ARCHIVE="${ARCHIVE:-${OUTPUT_ROOT}_completed.tar.gz}"
MEMBER_CHUNK="$(python - "$CONFIG" <<'PY'
import sys,yaml
print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["generation"]["member_chunk"])
PY
)"

[[ -f "${OUTPUT_ROOT}/checkpoints/best.pt" ]] || { echo "Missing best checkpoint"; exit 2; }
if [[ ! -e "$GENERATION_DIR" ]]; then
  python -u generate_shandong91_v3.py --run-dir "$OUTPUT_ROOT" --output-dir "$GENERATION_DIR" \
    --config "$CONFIG" --split validation --n-samples 500 --seed 424242 \
    --member-chunk "$MEMBER_CHUNK" --method ddpm --inference-steps 500 \
    --checkpoint-state ema --device cuda
fi
python - "$GENERATION_DIR/metadata.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],encoding="utf-8"))
assert m["model_identifier"]=="shandong91_heterogeneous_raw_body_v3_low_rank_common_factor"
assert m["split"]=="validation" and m["n_samples"]==500
assert m["sampler"]=="ddpm" and m["inference_steps"]==500
assert m["load_local_branch"]=="disabled_strict_rank1"
PY
if [[ ! -e "$EVALUATION_DIR" ]]; then
  python -u tools/evaluate_shandong91_v3.py --result "$GENERATION_DIR" --output-dir "$EVALUATION_DIR"
fi
python - "$EVALUATION_DIR/metrics.json" <<'PY'
import json,math,sys
m=json.load(open(sys.argv[1],encoding="utf-8")); assert math.isclose(m["finite_ratio"],1.0)
for r in ("Wind","Solar","Load"):
    for k in ("coverage90","crps_mw","width90_mw"):
        assert math.isfinite(m["system_aggregate"][r][k]),(r,k)
    assert math.isfinite(m["system_aggregate"][r]["covariance_amplification"]["mean"])
PY
if [[ ! -e "$ARCHIVE" ]]; then tar -czf "$ARCHIVE" "$OUTPUT_ROOT"; fi
echo "FINALIZE_COMPLETE=$ARCHIVE"
