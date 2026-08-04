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
    INPUT=data/translations/glaive_vi.jsonl
    OUTPUT=data/translations/qa_samples/glaive_qa.jsonl
    REPORT=data/translations/qa_samples/glaive_qa_report.json
    SOURCE_INPUT=data/processed/glaive_single_turn_raw.jsonl
    FAILED_INPUT=data/translations/failed/glaive_filtered_failed.jsonl
elif [ "$DATASET" = "xlam" ]; then
    INPUT=data/translations/xlam_vi.jsonl
    OUTPUT=data/translations/qa_samples/xlam_qa.jsonl
    REPORT=data/translations/qa_samples/xlam_qa_report.json
    SOURCE_INPUT=data/raw/xlam_raw.jsonl
    FAILED_INPUT=data/translations/failed/xlam_failed.jsonl
else
    echo "Usage: $0 [glaive|xlam]"
    exit 1
fi

INPUT=$INPUT OUTPUT=$OUTPUT REPORT=$REPORT \
    python -c "
import sys
sys.path.insert(0, '.')
import yaml, asyncio
from src.data.qa_translation import QAConfig, run_qa

with open('configs/data/qa.yaml') as f:
    raw = yaml.safe_load(f)
raw['input'] = '$INPUT'
raw['output'] = '$OUTPUT'
raw['report'] = '$REPORT'
raw['dataset'] = '$DATASET'
raw['source_input'] = '$SOURCE_INPUT'
raw['failed_input'] = '$FAILED_INPUT'
cfg = QAConfig.from_dict(raw)
asyncio.run(run_qa(cfg))
"
