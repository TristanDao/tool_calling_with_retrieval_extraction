#!/bin/bash
# Standalone feature_group classifier (for pre-classification)
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

TOOLS=${1:-data/benchmark_vi/tool_pool.json}

if [ ! -f "$TOOLS" ]; then
    echo "ERROR: tools file not found: $TOOLS"
    echo "Run run_benchmark.sh first to build tool_pool.json"
    exit 1
fi

python -m src.data.feature_group_classify \
    --config configs/data/feature_group.yaml \
    --tools "$TOOLS"
