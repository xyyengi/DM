#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${REPO_ROOT}"

export PIPELINE_MODE=upper_bound
exec bash run_station24_sampler_es_l1_resume.sh "$@"
