#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==> Collecting raw datasets from HuggingFace..."
python -m src.data.collect
