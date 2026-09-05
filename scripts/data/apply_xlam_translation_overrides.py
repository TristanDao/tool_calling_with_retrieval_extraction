"""Apply reviewed Codex translations to an aligned xLAM JSONL file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("overrides", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    applied: set[str] = set()
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in iter_rows(args.input):
            identifier = row.get("id")
            if identifier in overrides:
                row["query"] = overrides[identifier]
                applied.add(identifier)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if applied != set(overrides):
        temp.unlink(missing_ok=True)
        missing = sorted(set(overrides) - applied)
        raise RuntimeError(f"Override IDs not found: {missing}")
    os.replace(temp, args.output)
    print(json.dumps({"applied": len(applied), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
