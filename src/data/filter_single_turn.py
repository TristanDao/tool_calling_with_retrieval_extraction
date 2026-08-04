"""Filter Glaive raw records while preserving the Bộ 1 format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.data.normalize_en import _parse_glaive


def filter_glaive_single_turn(
    input_path: Path,
    output_path: Path,
    index_map_path: Path,
) -> dict[str, int]:
    """Keep positive first-turn Glaive records without changing their JSON shape."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_map_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    kept = 0
    skipped = 0

    with output_path.open("w", encoding="utf-8") as output, index_map_path.open(
        "w", encoding="utf-8"
    ) as index_map:
        with input_path.open("r", encoding="utf-8") as source:
            for source_index, line in enumerate(source):
                if not line.strip():
                    continue
                total += 1
                try:
                    raw: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if _parse_glaive(raw, source_index) is None:
                    skipped += 1
                    continue

                output.write(json.dumps(raw, ensure_ascii=False) + "\n")
                index_map.write(
                    json.dumps(
                        {"filtered_index": kept, "source_index": source_index},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1

        output.flush()
        os.fsync(output.fileno())
        index_map.flush()
        os.fsync(index_map.fileno())

    return {"total_raw": total, "kept": kept, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Glaive to positive single-turn raw records")
    parser.add_argument("--input", type=Path, default=Path("data/raw/glaive_raw.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/glaive_single_turn_raw.jsonl"),
    )
    parser.add_argument(
        "--index-map",
        type=Path,
        default=Path("data/processed/glaive_single_turn_index.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps(filter_glaive_single_turn(args.input, args.output, args.index_map)))


if __name__ == "__main__":
    main()
