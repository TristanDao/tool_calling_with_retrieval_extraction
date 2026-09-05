"""Fix the final exact-English natural-language argument descriptions in xLAM output."""

from __future__ import annotations

import json
import os
from pathlib import Path


OUTPUT = Path("data/translations/xlam_normalized_vi.jsonl")
FIXES = {
    "xlam_50004": "Giường cho mèo sang trọng",
    "xlam_50238": "Giày chạy bộ màu hồng sáng",
}


def walk(value: object, identifier: str) -> int:
    changed = 0
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "description" and isinstance(item, str) and identifier in FIXES and item in {"Luxury cat beds", "Bright pink running shoes"}:
                value[key] = FIXES[identifier]
                changed += 1
            else:
                changed += walk(item, identifier)
    elif isinstance(value, list):
        for item in value:
            changed += walk(item, identifier)
    return changed


def main() -> None:
    rows = [json.loads(line) for line in OUTPUT.open(encoding="utf-8")]
    changed = sum(walk(row, str(row.get("id"))) for row in rows)
    temp = OUTPUT.with_suffix(".fieldfix.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, OUTPUT)
    print(json.dumps({"changed": changed, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
