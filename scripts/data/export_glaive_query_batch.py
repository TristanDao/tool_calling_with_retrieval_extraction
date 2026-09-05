"""Export flagged Glaive queries with their English source for interactive review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("quality_report", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    flagged = json.loads(args.quality_report.read_text(encoding="utf-8"))["examples"]
    source_by_id: dict[str, dict[str, Any]] = {}
    with args.source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            record = json.loads(line)
            source_by_id[record["id"]] = record
    rows = [
        {
            "id": item["id"],
            "source": source_by_id[item["id"]]["query"],
            "current": item["query"],
        }
        for item in flagged[args.offset : args.offset + args.limit]
    ]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
