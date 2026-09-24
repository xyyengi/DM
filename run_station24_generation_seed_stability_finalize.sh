#!/usr/bin/env bash
# Idempotent continuation after interruption; completed stages are reused.
set -Eeuo pipefail
ROOT=${1:?'usage: run_station24_generation_seed_stability_finalize.sh PIPELINE_ROOT'}
cd /root/autodl-tmp/DM
exec bash run_station24_generation_seed_stability.sh "$ROOT"
