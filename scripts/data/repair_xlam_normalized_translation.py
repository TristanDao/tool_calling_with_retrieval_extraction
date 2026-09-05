"""Repair and realign an xLAM Vietnamese JSONL translation using verified translation memory."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


PROTECTED_VALUE_KEYS = {"id", "source", "name", "type", "required", "default", "has_tool_call"}


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object in {path}")
            yield value


def collect_memory(source: Any, translated: Any, memory: dict[str, Counter[str]], key: str | None = None) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for name, source_value in source.items():
            if name in translated:
                collect_memory(source_value, translated[name], memory, name)
        return
    if isinstance(source, list) and isinstance(translated, list):
        for source_value, translated_value in zip(source, translated):
            collect_memory(source_value, translated_value, memory, key)
        return
    if isinstance(source, str) and isinstance(translated, str) and source != translated and key not in PROTECTED_VALUE_KEYS:
        memory[source][translated] += 1


def repair_value(source: Any, translated: Any, memory: dict[str, str], key: str | None = None) -> Any:
    if isinstance(source, dict):
        translated_dict = translated if isinstance(translated, dict) else {}
        return {name: repair_value(value, translated_dict.get(name), memory, name) for name, value in source.items()}
    if isinstance(source, list):
        translated_list = translated if isinstance(translated, list) else []
        return [repair_value(value, translated_list[index] if index < len(translated_list) else None, memory, key) for index, value in enumerate(source)]
    if key in PROTECTED_VALUE_KEYS or not isinstance(source, str):
        return copy.deepcopy(source)
    if isinstance(translated, str) and translated != source:
        return translated
    return memory.get(source, source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()

    translated_by_id = {row["id"]: row for row in iter_rows(args.translated)}
    memory_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_count = 0
    for source_row in iter_rows(args.source):
        source_count += 1
        translated_row = translated_by_id.get(source_row["id"])
        if translated_row is not None:
            collect_memory(source_row, translated_row, memory_counts)
    memory = {source: candidates.most_common(1)[0][0] for source, candidates in memory_counts.items()}

    missing_before = 0
    memory_replacements = 0
    remaining_equal = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for source_row in iter_rows(args.source):
            translated_row = translated_by_id.get(source_row["id"])
            if translated_row is None:
                missing_before += 1
            repaired = repair_value(source_row, translated_row, memory)
            handle.write(json.dumps(repaired, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, args.output)

    for source_row, repaired_row in zip(iter_rows(args.source), iter_rows(args.output)):
        def count_values(source: Any, repaired: Any, key: str | None = None) -> None:
            nonlocal memory_replacements, remaining_equal
            if isinstance(source, dict) and isinstance(repaired, dict):
                for name, value in source.items():
                    count_values(value, repaired[name], name)
            elif isinstance(source, list) and isinstance(repaired, list):
                for left, right in zip(source, repaired):
                    count_values(left, right, key)
            elif isinstance(source, str) and isinstance(repaired, str) and key not in PROTECTED_VALUE_KEYS:
                if source != repaired and source in memory:
                    memory_replacements += 1
                elif source == repaired:
                    remaining_equal += 1
        count_values(source_row, repaired_row)

    stats = {
        "source_records": source_count,
        "translated_records_before": len(translated_by_id),
        "missing_records_reinserted": missing_before,
        "translation_memory_entries": len(memory),
        "translated_occurrences_covered_by_memory": memory_replacements,
        "remaining_equal_string_occurrences": remaining_equal,
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
