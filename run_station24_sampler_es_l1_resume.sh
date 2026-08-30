#!/usr/bin/env bash
set -euo pipefail

PIPELINE_MODE=${PIPELINE_MODE:-l1}
if [[ "${PIPELINE_MODE}" == "upper_bound" ]]; then
  PIPELINE_TAG=event_score_upper_bound
  VARIANT=geo_history_actual_body_tail_event_score_upper_bound
  CANDIDATE_LABEL="Event-score upper bound"
else
  PIPELINE_TAG=sampler_es_l1
  VARIANT=geo_history_actual_body_tail_sampler_es_l1
  CANDIDATE_LABEL="Sampler Energy Score L1"
fi

[[ "${CONDA_DEFAULT_ENV:-}" == "dm_env" ]] || {
  echo "ERROR: activate dm_env before resuming:" >&2
  echo "  source /root/miniconda3/etc/profile.d/conda.sh" >&2
  echo "  conda activate dm_env" >&2
  exit 1
}

[[ $# -ge 1 ]] || {
  echo "usage: bash $0 <score_pipeline_root> [source_raw_result]" >&2
  exit 2
}

PIPELINE_ROOT=$1
SOURCE_RAW_RESULT=${2:-${SOURCE_RAW_RESULT:-}}
PYTHON_BIN=${PYTHON_BIN:-python}
DATA=${DATA:-diffusion_input_station}
NSAMPLES=${NSAMPLES:-500}
GEN_SEED=${GEN_SEED:-424242}
ENERGY_MEMBERS=${ENERGY_MEMBERS:-80}
ISSUE_BATCH=${ISSUE_BATCH:-1}
MEMBER_CHUNK=${MEMBER_CHUNK:-10}

[[ -d "${PIPELINE_ROOT}" ]] || { echo "missing ${PIPELINE_ROOT}" >&2; exit 1; }
shopt -s nullglob
runs=("${PIPELINE_ROOT}"/training/*_station24_${PIPELINE_TAG}_*_seed2027)
shopt -u nullglob
[[ ${#runs[@]} -eq 1 ]] || {
  echo "expected one completed score training run, found ${#runs[@]}" >&2
  exit 1
}
RUN=${runs[0]}
[[ -f "${RUN}/checkpoints/model_best.pt" ]] || {
  echo "training checkpoint is incomplete" >&2
  exit 1
}

if [[ -z "${SOURCE_RAW_RESULT}" ]]; then
  OUTPUT_ROOT=$(dirname "${PIPELINE_ROOT}")
  while IFS= read -r -d '' candidate; do
    if [[ -z "${SOURCE_RAW_RESULT}" || "${candidate}" -nt "${SOURCE_RAW_RESULT}" ]]; then
      SOURCE_RAW_RESULT=${candidate}
    fi
  done < <(find "${OUTPUT_ROOT}" -type d \
    -path '*/body_tail_moe_raw_inference_*/validation_results/geo_history_actual_body_tail_moe_raw_val_n500_seed424242' \
    -print0)
fi
[[ -n "${SOURCE_RAW_RESULT}" && -d "${SOURCE_RAW_RESULT}" ]] || {
  echo "Raw body-tail result not found" >&2
  exit 1
}

SOURCE_RUN=$("${PYTHON_BIN}" - "${SOURCE_RAW_RESULT}" <<'PY'
import json, sys
from pathlib import Path
meta = json.loads((Path(sys.argv[1]) / "generation_metadata.json").read_text(encoding="utf-8"))
if meta.get("condition_variant") != "geo_history_actual_body_tail_moe_raw":
    raise SystemExit("source variant mismatch")
print(meta["run_dir"])
PY
)
RESULT="${PIPELINE_ROOT}/validation_results/${VARIANT}_val_n${NSAMPLES}_seed${GEN_SEED}"
if [[ ! -f "${RESULT}/metrics.json" ]]; then
  [[ ! -e "${RESULT}" ]] || {
    echo "incomplete result directory exists; move it aside before resume: ${RESULT}" >&2
    exit 1
  }
  echo "RESUME_GENERATION_START"
  "${PYTHON_BIN}" generate_station24.py \
    --run-dir "${RUN}" --data-path "${DATA}" --output-dir "${RESULT}" \
    --split val --n-samples "${NSAMPLES}" --seed "${GEN_SEED}" \
    --issue-batch-size "${ISSUE_BATCH}" --member-chunk-size "${MEMBER_CHUNK}" \
    --energy-score-member-limit "${ENERGY_MEMBERS}" --checkpoint-state raw \
    --result-variant "${VARIANT}"
fi

AUDIT="${PIPELINE_ROOT}/${PIPELINE_TAG}_audit"
AUDIT_FILE="${AUDIT}/sampler_es_l1_audit.json"
if [[ "${PIPELINE_MODE}" == "upper_bound" ]]; then
  AUDIT_FILE="${AUDIT}/event_score_upper_bound_audit.json"
fi
if [[ ! -f "${AUDIT_FILE}" ]]; then
  [[ ! -e "${AUDIT}" ]] || { echo "incomplete audit exists: ${AUDIT}" >&2; exit 1; }
  if [[ "${PIPELINE_MODE}" == "upper_bound" ]]; then
    "${PYTHON_BIN}" tools/audit_station24_event_score_upper_bound.py \
      --source-run "${SOURCE_RUN}" --candidate-run "${RUN}" \
      --source-result "${SOURCE_RAW_RESULT}" --candidate-result "${RESULT}" \
      --output-dir "${AUDIT}"
  else
    "${PYTHON_BIN}" tools/audit_station24_sampler_es_l1.py \
      --source-run "${SOURCE_RUN}" --candidate-run "${RUN}" \
      --source-result "${SOURCE_RAW_RESULT}" --candidate-result "${RESULT}" \
      --output-dir "${AUDIT}"
  fi
fi

COMPARISON="${PIPELINE_ROOT}/comparisons/raw_body_tail_vs_${PIPELINE_TAG}"
if [[ ! -f "${COMPARISON}/comparison_summary.csv" ]]; then
  [[ ! -e "${COMPARISON}" ]] || { echo "incomplete comparison exists: ${COMPARISON}" >&2; exit 1; }
  "${PYTHON_BIN}" tools/compare_station24_multiscale_2a.py \
    "${SOURCE_RAW_RESULT}" "${RESULT}" --data-path "${DATA}" \
    --output-dir "${COMPARISON}" \
    --baseline-variant geo_history_actual_body_tail_moe_raw \
    --candidate-variant "${VARIANT}" \
    --baseline-label "Raw body-tail" --candidate-label "${CANDIDATE_LABEL}" \
    --baseline-spatial-levels bottleneck --candidate-spatial-levels bottleneck \
    --baseline-parallel-levels encoder_0 --candidate-parallel-levels encoder_0 \
    --baseline-parallel-adjacency fixed --candidate-parallel-adjacency fixed \
    --title "Raw body-tail versus ${CANDIDATE_LABEL}" \
    --figure-prefix "raw_body_tail_vs_${PIPELINE_TAG}"
fi

TIMING="${PIPELINE_ROOT}/wind_event_timing/raw_body_tail_vs_${PIPELINE_TAG}"
if [[ ! -f "${TIMING}/event_records.csv" ]]; then
  [[ ! -e "${TIMING}" ]] || { echo "incomplete timing output exists: ${TIMING}" >&2; exit 1; }
  "${PYTHON_BIN}" tools/diagnose_station24_wind_event_timing.py \
    "${SOURCE_RAW_RESULT}" "${RESULT}" --data-path "${DATA}" \
    --output-dir "${TIMING}" \
    --baseline-variant geo_history_actual_body_tail_moe_raw \
    --candidate-variant "${VARIANT}" \
    --baseline-label "Raw body-tail" --candidate-label "${CANDIDATE_LABEL}"
fi

ATTRIBUTION="${PIPELINE_ROOT}/forecast_event_attribution/raw_body_tail_vs_${PIPELINE_TAG}"
if [[ ! -f "${ATTRIBUTION}/diagnostic_metadata.json" ]]; then
  [[ ! -e "${ATTRIBUTION}" ]] || { echo "incomplete attribution exists: ${ATTRIBUTION}" >&2; exit 1; }
  "${PYTHON_BIN}" tools/diagnose_station24_forecast_event_attribution.py \
    "${SOURCE_RAW_RESULT}" "${RESULT}" \
    --event-records "${TIMING}/event_records.csv" --data-path "${DATA}" \
    --output-dir "${ATTRIBUTION}"
fi

TAIL="${PIPELINE_ROOT}/extreme_wind_tail/raw_body_tail_vs_${PIPELINE_TAG}"
if [[ ! -f "${TAIL}/extreme_wind_tail_summary.json" ]]; then
  [[ ! -e "${TAIL}" ]] || { echo "incomplete tail output exists: ${TAIL}" >&2; exit 1; }
  "${PYTHON_BIN}" tools/plot_station24_extreme_tail.py \
    --baseline "${SOURCE_RAW_RESULT}" --candidate "${RESULT}" \
    --data-path "${DATA}" --output-dir "${TAIL}" --top-issues 5 \
    --baseline-label "Raw body-tail" --candidate-label "${CANDIDATE_LABEL}"
fi

ARCHIVE="$(dirname "${PIPELINE_ROOT}")/station24_$(basename "${PIPELINE_ROOT}").tar.gz"
tar -czf "${ARCHIVE}" -C "$(dirname "${PIPELINE_ROOT}")" "$(basename "${PIPELINE_ROOT}")"
echo "STATION24_SCORE_RESUME_COMPLETE mode=${PIPELINE_MODE}"
echo "PIPELINE_ROOT=${PIPELINE_ROOT}"
echo "RESULT=${RESULT}"
echo "ARCHIVE=${ARCHIVE}"
