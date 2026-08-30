#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${REPO_ROOT}"

export PIPELINE_MODE=upper_bound
export EXPECTED_BRANCH=${EXPECTED_BRANCH:-experiment/24site-event-score-upper-bound}
exec bash run_station24_sampler_es_l1_pipeline.sh
