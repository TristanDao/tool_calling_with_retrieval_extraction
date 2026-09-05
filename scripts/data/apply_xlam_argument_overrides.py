"""Apply reviewed natural-language argument value overrides to aligned xLAM JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def walk(source: Any, translated: Any, overrides: dict[str, dict[str, str]]) -> int:
    changed = 0
    if isinstance(source, dict) and isinstance(translated, dict):
        for key, value in source.items():
            if key not in translated:
                continue
            lookup_keys = [key]
            if key in {"q", "query"}:
                lookup_keys.extend(["q", "query"])
            if key in {"text", "caption"}:
                lookup_keys.extend(["text", "q", "query"])
            replacement = None
            if isinstance(value, str) and translated[key] == value:
                for lookup_key in lookup_keys:
                    if lookup_key in overrides and value in overrides[lookup_key]:
                        replacement = overrides[lookup_key][value]
                        break
            if replacement is not None:
                translated[key] = replacement
                changed += 1
            else:
                changed += walk(value, translated[key], overrides)
    elif isinstance(source, list) and isinstance(translated, list):
        for left, right in zip(source, translated):
            changed += walk(left, right, overrides)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("overrides", type=Path)
    args = parser.parse_args()
    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    rows_source = [json.loads(line) for line in args.source.open(encoding="utf-8")]
    rows_translated = [json.loads(line) for line in args.translated.open(encoding="utf-8")]
    if len(rows_source) != len(rows_translated):
        raise RuntimeError("Source/output line count mismatch")
    changed = sum(walk(source, translated, overrides) for source, translated in zip(rows_source, rows_translated))
    temp = args.translated.with_suffix(args.translated.suffix + ".argtmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows_translated:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, args.translated)
    print(json.dumps({"changed": changed, "output": str(args.translated)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
