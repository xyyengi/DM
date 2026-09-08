#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "${REPO_ROOT}"

export MSEP_RUN_FAMILY=jstd_msep_v2_prior_only
export MSEP_RESULT_VARIANT=geo_history_actual_jstd_msep_v2_prior_only_raw
export MSEP_RESULT_LABEL="JSTD-MSEP V2 causal prior-only"

exec bash run_station24_jstd_msep_finalize.sh "$@"
