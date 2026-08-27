#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 outputs_shandong/station24/retrieval_dual_tail_JOB" >&2
  exit 2
fi

PIPELINE_ROOT=$1
[[ -d "${PIPELINE_ROOT}" ]] || {
  echo "ERROR: pipeline root not found: ${PIPELINE_ROOT}" >&2
  exit 1
}

RESUME_PIPELINE_ROOT="${PIPELINE_ROOT}" \
  bash run_station24_retrieval_dual_tail_pipeline.sh
