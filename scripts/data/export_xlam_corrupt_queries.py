"""Export query rows containing signatures from the corrupt translation batch."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD = re.compile(r"ftrong|trtr|chtr|hoặce|perstrên|fhoặc|trtrêng|Chrlà|Spatrong|schoặce|mtrênthly|trênly|trongchomation|whtại|tạieghoặcy|ttrtrêng|ctrên|strtr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for source_line, translated_line in zip(args.source.open(encoding="utf-8"), args.translated.open(encoding="utf-8")):
        source = json.loads(source_line)
        translated = json.loads(translated_line)
        query = translated.get("query")
        if isinstance(query, str) and BAD.search(query):
            rows.append({"id": source.get("id"), "source": source.get("query"), "current": query})
    print(json.dumps(rows[args.offset : args.offset + args.limit], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
