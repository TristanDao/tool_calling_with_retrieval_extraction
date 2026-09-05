"""Scan xLAM function arguments for untranslated, mixed, or corrupted natural language."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repair_xlam_translation_integrity import (
    CORRUPTION_PATTERN,
    collect_protected_values,
    is_technical_value,
    load_argument_overrides,
)


WORD = re.compile(r"[A-Za-zÀ-ỹ]+(?:'[A-Za-zÀ-ỹ]+)?")
ENGLISH = re.compile(
    r"\b(?:a|an|and|are|as|at|be|best|by|can|check|contact|details|do|does|for|from|"
    r"get|good|hello|how|i|in|into|is|it|job|more|my|near|never|new|of|on|or|our|"
    r"please|review|sale|search|tell|thank|that|the|this|to|used|visit|what|where|with|you|your)\b",
    re.IGNORECASE,
)
VIETNAMESE = re.compile(
    r"\b(?:bạn|bằng|bán|các|cần|chi tiết|cho|chúng tôi|có|của|đến|để|được|gần|"
    r"hãy|hoặc|là|liên hệ|một|những|ở|tìm|tôi|trong|trên|từ|và|với|xin)\b",
    re.IGNORECASE,
)
QUOTED = re.compile(r"(?<!\w)'.*'(?!\w)|\".*?\"")
NATURAL_KEYS = {
    "caption", "content", "input", "instruction", "keyword", "keywords", "message", "phrase",
    "prompt", "q", "question", "query", "sentence", "subject", "summary", "text", "title", "topic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--translations-dir", type=Path)
    parser.add_argument("--override-prefix", default="xlam")
    return parser.parse_args()


def load_prefixed_overrides(directory: Path, prefix: str) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = defaultdict(dict)
    for path in sorted(directory.glob(f"{prefix}_argument_overrides*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for key, values in data.items():
            if isinstance(values, dict):
                merged[key].update(values)
    return merged


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle]


def looks_natural(key: str, source: str) -> bool:
    words = WORD.findall(source)
    if len(words) < 2:
        return False
    english_count = len(ENGLISH.findall(source))
    return key.lower() in NATURAL_KEYS or english_count >= 1 or (len(words) >= 4 and source != source.upper())


def classify(
    key: str,
    source: str,
    translated: str,
    overrides: dict[str, dict[str, str]],
) -> tuple[str, int, int] | None:
    english_count = len(ENGLISH.findall(translated))
    vietnamese_count = len(VIETNAMESE.findall(translated))
    if overrides.get(key, {}).get(source) == translated:
        return None
    if CORRUPTION_PATTERN.search(translated):
        return "corrupt", english_count, vietnamese_count
    if source == translated and looks_natural(key, source):
        return "unchanged_natural_candidate", english_count, vietnamese_count
    if source != translated and english_count >= 2 and vietnamese_count >= 1:
        return "mixed_language", english_count, vietnamese_count
    if source != translated and english_count >= 3 and vietnamese_count == 0:
        return "english_after_translation", english_count, vietnamese_count
    return None


def scan_value(
    source: Any,
    translated: Any,
    key: str,
    path: str,
    protected_values: set[str],
    record_id: str,
    function_name: str,
    overrides: dict[str, dict[str, str]],
    findings: list[dict[str, Any]],
) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, source_child in source.items():
            child_path = f"{path}.{child_key}" if path else child_key
            scan_value(
                source_child,
                translated[child_key],
                child_key,
                child_path,
                protected_values,
                record_id,
                function_name,
                overrides,
                findings,
            )
        return
    if isinstance(source, list) and isinstance(translated, list):
        for index, (source_child, translated_child) in enumerate(zip(source, translated)):
            scan_value(
                source_child,
                translated_child,
                key,
                f"{path}[{index}]",
                protected_values,
                record_id,
                function_name,
                overrides,
                findings,
            )
        return
    if not isinstance(source, str) or not isinstance(translated, str):
        return
    if is_technical_value(key, source, protected_values):
        return
    result = classify(key, source, translated, overrides)
    if result is None:
        return
    category, english_count, vietnamese_count = result
    findings.append(
        {
            "id": record_id,
            "function": function_name,
            "path": path,
            "key": key,
            "category": category,
            "english": english_count,
            "vietnamese": vietnamese_count,
            "source": source,
            "translated": translated,
        }
    )


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.source)
    translated_records = load_jsonl(args.translated)
    if len(source_records) != len(translated_records):
        raise ValueError("Source and translated JSONL files have different line counts")
    protected_values: set[str] = set()
    overrides = load_prefixed_overrides(args.translations_dir, args.override_prefix) if args.translations_dir else {}
    for record in source_records:
        for call in record["function_calls"]:
            collect_protected_values(call["arguments"], "", protected_values)
    for source_record, translated_record in zip(source_records, translated_records):
        source_literals = [match.group(0)[1:-1] for match in QUOTED.finditer(source_record["query"])]
        translated_literals = [match.group(0)[1:-1] for match in QUOTED.finditer(translated_record["query"])]
        if len(source_literals) == len(translated_literals):
            protected_values.update(source for source, translated in zip(source_literals, translated_literals) if source == translated)
    findings: list[dict[str, Any]] = []
    for source_record, translated_record in zip(source_records, translated_records):
        if source_record["id"] != translated_record["id"]:
            raise ValueError(f"Record ID mismatch: {source_record['id']} != {translated_record['id']}")
        for index, source_call in enumerate(source_record["function_calls"]):
            translated_call = translated_record["function_calls"][index]
            scan_value(
                source_call["arguments"],
                translated_call["arguments"],
                "",
                f"function_calls[{index}].arguments",
                protected_values,
                source_record["id"],
                source_call["name"],
                overrides,
                findings,
            )
    counts = Counter(item["category"] for item in findings)
    unique = {
        (item["key"], item["source"], item["translated"], item["category"])
        for item in findings
    }
    report = {
        "records": len(source_records),
        "findings": len(findings),
        "unique_findings": len(unique),
        "counts": dict(counts),
        "examples": findings,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
