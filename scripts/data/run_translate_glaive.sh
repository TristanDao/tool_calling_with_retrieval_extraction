#!/bin/bash
# Run translation pipeline for Glaive (EN -> VI)
# Usage:
#   bash scripts/data/run_translate_glaive.sh              # full dataset
#   bash scripts/data/run_translate_glaive.sh 0 100        # mini pilot 0..100
#   bash scripts/data/run_translate_glaive.sh 0 1000       # 1k pilot
#   bash scripts/data/run_translate_glaive.sh 175 176 --smoke-test
#   MODEL=qwen3.7-max bash scripts/data/run_translate_glaive.sh 0 1000
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -n "${MODEL:-}" ]; then
    export ALIBABA_MODEL="$MODEL"
fi

START=${1:-0}
END=${2:-}
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -gt 0 ]; then shift; fi

EXTRA_ARGS=""
if [ -n "$END" ]; then
    EXTRA_ARGS="--end $END"
fi

python -m src.data.translate \
    --config configs/data/translate.yaml \
    --input data/processed/glaive_single_turn_raw.jsonl \
    --output data/translations/glaive_vi.jsonl \
    --failed-output data/translations/failed/glaive_filtered_failed.jsonl \
    --checkpoint data/translations/.checkpoint/glaive_filtered.json \
    --log-file data/translations/logs/translate_glaive_filtered.log \
    --start $START $EXTRA_ARGS "$@"
