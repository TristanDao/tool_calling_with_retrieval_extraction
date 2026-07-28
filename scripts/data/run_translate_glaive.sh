#!/bin/bash
# Run translation pipeline for Glaive (EN -> VI)
# Usage:
#   bash scripts/data/run_translate_glaive.sh              # full dataset
#   bash scripts/data/run_translate_glaive.sh 0 100        # mini pilot 0..100
#   bash scripts/data/run_translate_glaive.sh 0 1000       # 1k pilot
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

START=${1:-0}
END=${2:-}

EXTRA_ARGS=""
if [ -n "$END" ]; then
    EXTRA_ARGS="--end $END"
fi

TRANSLATE_INPUT=data/raw/glaive_raw.jsonl \
TRANSLATE_OUTPUT=data/translations/glaive_vi.jsonl \
TRANSLATE_FAILED_OUTPUT=data/translations/failed/glaive_failed.jsonl \
TRANSLATE_CHECKPOINT=data/translations/.checkpoint/glaive.json \
TRANSLATE_LOG=data/translations/logs/translate_glaive.log \
TRANSLATE_START=$START \
python -m src.data.translate \
    --config configs/data/translate.yaml \
    --start $START $EXTRA_ARGS
