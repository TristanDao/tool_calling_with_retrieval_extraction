"""Translate remaining exact-English xLAM natural-language fields with the project's local phrase mapper."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


SOURCE = Path(r"D:\Download\xlam_normalized.jsonl")
OUTPUT = Path("data/translations/xlam_normalized_vi.jsonl")


def load_mapper() -> Any:
    spec = importlib.util.spec_from_file_location("translate_remaining_codex", "scripts/data/translate_remaining_codex.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load local translation mapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def translate_descriptions(source: Any, translated: Any, mapper: Any) -> int:
    changed = 0
    if isinstance(source, dict) and isinstance(translated, dict):
        for key in source:
            if key not in translated:
                continue
            if key == "description" and isinstance(source[key], str) and source[key] == translated[key]:
                translated[key] = mapper.translate_description(source[key])
                changed += 1
            else:
                changed += translate_descriptions(source[key], translated[key], mapper)
    elif isinstance(source, list) and isinstance(translated, list):
        for left, right in zip(source, translated):
            changed += translate_descriptions(left, right, mapper)
    return changed


def main() -> None:
    mapper = load_mapper()
    source_rows = [json.loads(line) for line in SOURCE.open(encoding="utf-8")]
    output_rows = [json.loads(line) for line in OUTPUT.open(encoding="utf-8")]
    if len(source_rows) != len(output_rows):
        raise RuntimeError("Source/output line count mismatch")
    changed_queries = 0
    changed_descriptions = 0
    for source, translated in zip(source_rows, output_rows):
        if source.get("query") == translated.get("query") and isinstance(source.get("query"), str):
            translated["query"] = mapper.translate_query(source["query"])
            changed_queries += 1
        changed_descriptions += translate_descriptions(source, translated, mapper)
    temp = OUTPUT.with_suffix(".auto.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, OUTPUT)
    print(json.dumps({"changed_queries": changed_queries, "changed_descriptions": changed_descriptions, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
