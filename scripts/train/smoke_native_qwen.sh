#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
python scripts/train/smoke_native_qwen.py "$@"
