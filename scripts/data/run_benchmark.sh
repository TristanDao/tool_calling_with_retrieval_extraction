#!/bin/bash
# Build and freeze a paired core benchmark revision
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m src.data.rebuild_benchmark "$@"
