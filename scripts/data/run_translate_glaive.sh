#!/bin/bash
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
    --input data/normalized_en/glaive_normalized.jsonl \
    --output data/translations/glaive_normalized_vi.jsonl \
    --failed-output data/translations/failed/glaive_normalized_failed.jsonl \
    --checkpoint data/translations/.checkpoint/glaive_normalized.json \
    --log-file data/translations/logs/translate_glaive_normalized.log \
    --dataset glaive_normalized \
    --start "$START" $EXTRA_ARGS "$@"
