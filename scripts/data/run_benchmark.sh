#!/bin/bash
# Build Bộ 2 benchmark from Bộ 1 translations
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

python -m src.data.build_benchmark --config configs/data/benchmark.yaml "$@"
