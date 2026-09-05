"""Scan translated xLAM text fields for common English remnants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ENGLISH = re.compile(r"\b(?:what|is|are|the|from|when|with|and|or|also|check|if|contains|any|language|text|response|service|filling|spaces|adding|additional|parameters|determine|in|for|to|of|on|at|this|that|how|can|please|find|search|latest|news|today|used|use|your|meaning|life)\b", re.I)
FIELDS = {"query", "description", "text", "q", "keyword", "keywords", "caption"}


def walk(value: Any, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key == "query" and not path and isinstance(child, str) and len(ENGLISH.findall(child)) >= 2:
                found.append((child_path, child))
            found.extend(walk(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(walk(child, f"{path}[{index}]"))
    return found


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("translated", type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    total = 0
    examples: list[dict[str, object]] = []
    for line_number, line in enumerate(args.translated.open(encoding="utf-8"), 1):
        row = json.loads(line)
        matches = walk(row)
        total += len(matches)
        for path, text in matches:
            if len(examples) < args.limit:
                examples.append({"line": line_number, "id": row.get("id"), "path": path, "text": text})
    report = {"matches": total, "examples": examples}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matches": total, "examples": examples[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
