#!/bin/bash
# Build raw-preserving Glaive single-turn input for translation.
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.data.filter_single_turn \
    --input data/raw/glaive_raw.jsonl \
    --output data/processed/glaive_single_turn_raw.jsonl \
    --index-map data/processed/glaive_single_turn_index.jsonl
