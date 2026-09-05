"""Build reviewed Glaive argument translations from repeated clean candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repair_xlam_translation_integrity import CORRUPTION_PATTERN
from scan_xlam_argument_quality import ENGLISH, VIETNAMESE, WORD


DIACRITIC = re.compile(r"[À-ỹĐđ]")
CREATIVE_FUNCTION = re.compile(r"book|movie|song|music|lyric|podcast|restaurant", re.IGNORECASE)
CREATIVE_KEYS = {"album", "artist", "book_name", "book_title", "movie_title", "name", "query", "song", "song_name", "song_title", "title", "track", "track_name"}
PRODUCT_KEYS = {"item_name", "product", "product_name", "products", "service_or_product"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("original_translation", type=Path)
    parser.add_argument("findings", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle]


def collect_pairs(source: Any, translated: Any, key: str, pairs: dict[tuple[str, str], Counter[str]]) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, source_child in source.items():
            if child_key in translated:
                collect_pairs(source_child, translated[child_key], child_key, pairs)
        return
    if isinstance(source, list) and isinstance(translated, list):
        for source_child, translated_child in zip(source, translated):
            collect_pairs(source_child, translated_child, key, pairs)
        return
    if isinstance(source, str) and isinstance(translated, str):
        pairs[(key, source)][translated] += 1


def is_title_case_name(value: str) -> bool:
    words = WORD.findall(value)
    return 1 <= len(words) <= 8 and all(word[0].isupper() for word in words if word)


def should_preserve(functions: set[str], key: str, source: str) -> bool:
    if key in CREATIVE_KEYS and is_title_case_name(source) and all(CREATIVE_FUNCTION.search(name) for name in functions):
        return True
    if key in PRODUCT_KEYS and is_title_case_name(source):
        return True
    return False


def candidate_rank(candidate: str, count: int) -> tuple[int, int, int, int]:
    return (
        len(ENGLISH.findall(candidate)),
        -len(VIETNAMESE.findall(candidate)),
        -count,
        len(candidate),
    )


def is_clean_translation(source: str, candidate: str) -> bool:
    if candidate == source or CORRUPTION_PATTERN.search(candidate):
        return False
    if not (DIACRITIC.search(candidate) or VIETNAMESE.search(candidate)):
        return False
    word_count = max(1, len(WORD.findall(candidate)))
    return len(ENGLISH.findall(candidate)) <= max(1, word_count // 8)


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.source)
    translated_by_id = {record["id"]: record for record in load_jsonl(args.original_translation)}
    pairs: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for source_record in source_records:
        translated_record = translated_by_id.get(source_record["id"])
        if translated_record is None:
            continue
        collect_pairs(source_record["function_calls"], translated_record.get("function_calls", []), "", pairs)
    report = json.loads(args.findings.read_text(encoding="utf-8"))
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for finding in report["examples"]:
        targets[(finding["key"], finding["source"])].add(finding["function"])
    overrides: dict[str, dict[str, str]] = defaultdict(dict)
    unresolved: list[dict[str, Any]] = []
    for (key, source), functions in sorted(targets.items()):
        if should_preserve(functions, key, source):
            continue
        candidates = [
            (candidate, count)
            for candidate, count in pairs[(key, source)].items()
            if is_clean_translation(source, candidate)
        ]
        if not candidates:
            unresolved.append({"key": key, "source": source, "functions": sorted(functions)})
            continue
        candidate, _ = min(candidates, key=lambda item: candidate_rank(*item))
        overrides[key][source] = candidate
    args.output.write_text(json.dumps(overrides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"overrides": sum(map(len, overrides.values())), "unresolved": len(unresolved), "unresolved_examples": unresolved[:30]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
