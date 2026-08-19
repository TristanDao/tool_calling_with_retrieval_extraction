#!/bin/bash
# Run QA pipeline (rule check + LLM judge)
# Usage:
#   bash scripts/data/run_qa.sh glaive
#   bash scripts/data/run_qa.sh xlam
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

DATASET=${1:-glaive}

if [ "$DATASET" = "glaive" ]; then
    CONFIG=configs/data/qa_glaive_normalized.yaml
elif [ "$DATASET" = "xlam" ]; then
    CONFIG=configs/data/qa_xlam_normalized.yaml
else
    echo "Usage: $0 [glaive|xlam]"
    exit 1
fi

python -m src.data.qa_translation --config "$CONFIG"
