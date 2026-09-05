"""Export untranslated xLAM natural-language fields for an interactive Codex batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("data/translations/xlam_codex_query_overrides.json"),
    )
    args = parser.parse_args()
    reviewed_ids: set[str] = set()
    if args.reviewed.exists():
        reviewed_ids = set(json.loads(args.reviewed.read_text(encoding="utf-8")))
    batch: list[dict[str, str]] = []
    for source_row, translated_row in zip(iter_rows(args.source), iter_rows(args.translated)):
        source_query = source_row.get("query")
        translated_query = translated_row.get("query")
        source_id = str(source_row["id"])
        if (
            source_id not in reviewed_ids
            and isinstance(source_query, str)
            and source_query == translated_query
        ):
            batch.append({"id": source_id, "source": source_query})
            if len(batch) == args.limit:
                break
    print(json.dumps(batch, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
