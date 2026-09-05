"""Repair and realign the normalized Vietnamese Glaive translation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repair_xlam_translation_integrity import CORRUPTION_PATTERN, collect_protected_values, is_technical_value
from scan_xlam_query_quality import CORRUPTION as QUERY_CORRUPTION
from scan_xlam_query_quality import ENGLISH, QUOTED, URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--translations-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle]


def query_rank(source: str, translated: str, frequency: int) -> tuple[int, int, int, int, int]:
    unprotected = URL.sub("", QUOTED.sub("", translated))
    return (
        int(bool(QUERY_CORRUPTION.search(translated) or CORRUPTION_PATTERN.search(translated))),
        len(ENGLISH.findall(unprotected)),
        int(source == translated),
        -frequency,
        len(translated),
    )


def description_rank(source: str, translated: str, frequency: int) -> tuple[int, int, int, int]:
    return (
        int(bool(CORRUPTION_PATTERN.search(translated))),
        int(source == translated and bool(source)),
        -frequency,
        len(translated),
    )


def walk_pairs(source: Any, translated: Any, key: str, pairs: dict[str, Counter[str]]) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, source_child in source.items():
            if child_key in translated:
                walk_pairs(source_child, translated[child_key], child_key, pairs)
    elif isinstance(source, list) and isinstance(translated, list):
        for source_child, translated_child in zip(source, translated):
            walk_pairs(source_child, translated_child, key, pairs)
    elif key == "description" and isinstance(source, str) and isinstance(translated, str):
        pairs[source][translated] += 1


def rebuild_tools(source: Any, translated: Any, canonical: dict[str, str], key: str = "") -> Any:
    if isinstance(source, dict):
        translated_dict = translated if isinstance(translated, dict) else {}
        return {
            child_key: rebuild_tools(source_child, translated_dict.get(child_key), canonical, child_key)
            for child_key, source_child in source.items()
        }
    if isinstance(source, list):
        translated_list = translated if isinstance(translated, list) else []
        return [
            rebuild_tools(source_child, translated_list[index] if index < len(translated_list) else None, canonical, key)
            for index, source_child in enumerate(source)
        ]
    if key == "description" and isinstance(source, str):
        return canonical.get(source, source)
    if key == "feature_group" and isinstance(source, str) and isinstance(translated, str):
        return translated
    return source


def repair_arguments(
    source: Any,
    translated: Any,
    key: str,
    protected_values: set[str],
    overrides: dict[str, dict[str, str]],
    literal_overrides: dict[str, str],
    stats: Counter[str],
) -> Any:
    if isinstance(source, dict):
        translated_dict = translated if isinstance(translated, dict) else {}
        return {
            child_key: repair_arguments(
                source_child,
                translated_dict.get(child_key),
                child_key,
                protected_values,
                overrides,
                literal_overrides,
                stats,
            )
            for child_key, source_child in source.items()
        }
    if isinstance(source, list):
        translated_list = translated if isinstance(translated, list) else []
        return [
            repair_arguments(
                source_child,
                translated_list[index] if index < len(translated_list) else None,
                key,
                protected_values,
                overrides,
                literal_overrides,
                stats,
            )
            for index, source_child in enumerate(source)
        ]
    if not isinstance(source, str):
        return source
    if source in literal_overrides:
        reviewed_literal = literal_overrides[source]
        if translated != reviewed_literal:
            stats["query_literals_aligned"] += 1
        return reviewed_literal
    reviewed = overrides.get(key, {}).get(source)
    if reviewed is not None:
        if translated != reviewed:
            stats["argument_overrides_applied"] += 1
        return reviewed
    if not isinstance(translated, str):
        return source
    if is_technical_value(key, source, protected_values) or CORRUPTION_PATTERN.search(translated):
        if translated != source:
            stats["argument_values_restored"] += 1
        return source
    return translated


def build_literal_overrides(source_query: str, translated_query: str) -> dict[str, str]:
    source_literals = [match.group(0)[1:-1] for match in QUOTED.finditer(source_query)]
    translated_literals = [match.group(0)[1:-1] for match in QUOTED.finditer(translated_query)]
    if len(source_literals) != len(translated_literals):
        return {}
    return dict(zip(source_literals, translated_literals))


def load_overrides(directory: Path, pattern: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in sorted(directory.glob(pattern)):
        merged.update(json.loads(path.read_text(encoding="utf-8-sig")))
    return merged


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.source)
    translated_records = load_jsonl(args.input)
    translated_by_id = {record["id"]: record for record in translated_records}
    source_by_id = {record["id"]: record for record in source_records}
    query_pairs: dict[str, Counter[str]] = defaultdict(Counter)
    description_pairs: dict[str, Counter[str]] = defaultdict(Counter)
    protected_values: set[str] = set()
    for record in source_records:
        translated = translated_by_id.get(record["id"])
        if translated is not None:
            query_pairs[record["query"]][translated["query"]] += 1
            walk_pairs(record["tools"], translated.get("tools", []), "", description_pairs)
        for call in record["function_calls"]:
            collect_protected_values(call["arguments"], "", protected_values)
    canonical_queries = {
        source: min(candidates, key=lambda value: query_rank(source, value, candidates[value]))
        for source, candidates in query_pairs.items()
    }
    canonical_descriptions = {
        source: min(candidates, key=lambda value: description_rank(source, value, candidates[value]))
        for source, candidates in description_pairs.items()
    }
    description_overrides = load_overrides(args.translations_dir, "glaive_description_overrides*.json")
    canonical_descriptions.update(
        {source: translated for source, translated in description_overrides.items() if isinstance(translated, str)}
    )
    canonical_descriptions.setdefault("", "")
    query_overrides = load_overrides(args.translations_dir, "glaive_query_overrides*.json")
    raw_argument_overrides = load_overrides(args.translations_dir, "glaive_argument_overrides*.json")
    argument_overrides = {
        key: value for key, value in raw_argument_overrides.items() if isinstance(value, dict)
    }
    stats: Counter[str] = Counter()
    repaired_records: list[dict[str, Any]] = []
    for source in source_records:
        translated = translated_by_id.get(source["id"])
        if translated is None:
            translated = copy.deepcopy(source)
            stats["missing_records_restored"] += 1
        repaired = copy.deepcopy(source)
        query = canonical_queries.get(source["query"], translated["query"])
        if source["id"] in query_overrides:
            query = query_overrides[source["id"]]
        if query != translated["query"]:
            stats["queries_repaired"] += 1
        repaired["query"] = query
        literal_overrides = build_literal_overrides(source["query"], query)
        repaired_calls: list[dict[str, Any]] = []
        translated_calls = translated.get("function_calls", [])
        for index, source_call in enumerate(source["function_calls"]):
            translated_call = translated_calls[index] if index < len(translated_calls) else {}
            translated_arguments = translated_call.get("arguments", {})
            repaired_calls.append(
                {
                    "name": source_call["name"],
                    "arguments": repair_arguments(
                        source_call["arguments"],
                        translated_arguments,
                        "",
                        protected_values,
                        argument_overrides,
                        literal_overrides,
                        stats,
                    ),
                }
            )
        repaired["function_calls"] = repaired_calls
        repaired["tools"] = rebuild_tools(source["tools"], translated.get("tools", []), canonical_descriptions)
        repaired_records.append(repaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in repaired_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    stats["records"] = len(repaired_records)
    stats["canonical_queries"] = len(canonical_queries)
    stats["canonical_descriptions"] = len(canonical_descriptions)
    stats["protected_values"] = len(protected_values)
    stats["source_ids"] = len(source_by_id)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
