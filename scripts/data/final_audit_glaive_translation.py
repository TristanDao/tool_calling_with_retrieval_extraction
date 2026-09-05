"""Run the final structural and translation-quality audit for Glaive JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repair_glaive_translation_integrity import build_literal_overrides
from repair_xlam_translation_integrity import CORRUPTION_PATTERN, collect_protected_values, is_technical_value
from scan_xlam_argument_quality import VIETNAMESE, classify, load_prefixed_overrides
from scan_xlam_query_quality import CORRUPTION as QUERY_CORRUPTION
from scan_xlam_query_quality import ENGLISH, QUOTED, URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("--translations-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append({"line": line_number, "error": str(error)})
                continue
            if not isinstance(value, dict):
                errors.append({"line": line_number, "error": "top-level value is not an object"})
                continue
            records.append(value)
    return records, errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_shape(source: Any, translated: Any) -> bool:
    if type(source) is not type(translated):
        return False
    if isinstance(source, dict):
        return list(source) == list(translated) and all(same_shape(source[key], translated[key]) for key in source)
    if isinstance(source, list):
        return len(source) == len(translated) and all(same_shape(a, b) for a, b in zip(source, translated))
    if isinstance(source, str):
        return True
    return source == translated


def protected_tool_view(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {child_key: protected_tool_view(child, child_key) for child_key, child in value.items()}
    if isinstance(value, list):
        return [protected_tool_view(child, key) for child in value]
    if key == "description" and isinstance(value, str):
        return "<translated-description>"
    return value


def value_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: value_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [value_shape(child) for child in value]
    return type(value).__name__


def quoted_values(text: str) -> list[str]:
    return [match.group(0)[1:-1] for match in QUOTED.finditer(text)]


def walk_arguments(
    source: Any,
    translated: Any,
    key: str,
    protected_values: set[str],
    overrides: dict[str, dict[str, str]],
    literal_overrides: dict[str, str],
    record_id: str,
    translated_query: str,
    counts: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, source_child in source.items():
            walk_arguments(source_child, translated[child_key], child_key, protected_values, overrides, literal_overrides, record_id, translated_query, counts, examples)
        return
    if isinstance(source, list) and isinstance(translated, list):
        for source_child, translated_child in zip(source, translated):
            walk_arguments(source_child, translated_child, key, protected_values, overrides, literal_overrides, record_id, translated_query, counts, examples)
        return
    if not isinstance(source, str) or not isinstance(translated, str):
        return
    if source in literal_overrides:
        counts["query_literal_arguments"] += 1
        expected = literal_overrides[source]
        if translated != expected:
            counts["query_literal_argument_mismatches"] += 1
            examples["query_literal_argument_mismatches"].append({"id": record_id, "source": source, "expected": expected, "actual": translated})
        return
    reviewed = overrides.get(key, {}).get(source)
    technical = is_technical_value(key, source, protected_values)
    if reviewed is not None:
        counts["reviewed_argument_values"] += 1
        if translated != reviewed:
            counts["reviewed_argument_mismatches"] += 1
            examples["reviewed_argument_mismatches"].append({"id": record_id, "key": key, "source": source, "expected": reviewed, "actual": translated})
    elif technical:
        counts["protected_argument_values"] += 1
        if translated != source:
            counts["protected_argument_changes"] += 1
            examples["protected_argument_changes"].append({"id": record_id, "key": key, "source": source, "translated": translated})
    quality = None if technical else classify(key, source, translated, overrides)
    if quality is not None:
        counts["argument_quality_issues"] += 1
        examples["argument_quality_issues"].append({"id": record_id, "key": key, "category": quality[0], "source": source, "translated": translated})
    if source != translated:
        counts["translated_argument_values"] += 1


def main() -> None:
    args = parse_args()
    source_records, source_errors = load_jsonl(args.source)
    translated_records, translated_errors = load_jsonl(args.translated)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts["source_json_errors"] = len(source_errors)
    counts["translated_json_errors"] = len(translated_errors)
    counts["source_lines"] = len(source_records) + len(source_errors)
    counts["translated_lines"] = len(translated_records) + len(translated_errors)
    source_ids = [record.get("id") for record in source_records]
    translated_ids = [record.get("id") for record in translated_records]
    counts["source_duplicate_ids"] = len(source_ids) - len(set(source_ids))
    counts["translated_duplicate_ids"] = len(translated_ids) - len(set(translated_ids))
    counts["id_order_mismatches"] = sum(a != b for a, b in zip(source_ids, translated_ids)) + abs(len(source_ids) - len(translated_ids))
    protected_values: set[str] = set()
    for record in source_records:
        for call in record.get("function_calls", []):
            collect_protected_values(call.get("arguments", {}), "", protected_values)
    overrides = load_prefixed_overrides(args.translations_dir, "glaive")
    description_overrides: dict[str, str] = {}
    for path in sorted(args.translations_dir.glob("glaive_description_overrides*.json")):
        description_overrides.update(json.loads(path.read_text(encoding="utf-8-sig")))
    for source, translated in zip(source_records, translated_records):
        record_id = source.get("id", "")
        counts["aligned_records"] += 1
        if not same_shape(source, translated):
            counts["shape_mismatches"] += 1
            examples["shape_mismatches"].append({"id": record_id})
            continue
        if source.get("id") != translated.get("id") or source.get("source") != translated.get("source") or source.get("has_tool_call") != translated.get("has_tool_call"):
            counts["identity_mismatches"] += 1
            examples["identity_mismatches"].append({"id": record_id})
        for source_call, translated_call in zip(source["function_calls"], translated["function_calls"]):
            if source_call["name"] != translated_call["name"] or value_shape(source_call["arguments"]) != value_shape(translated_call["arguments"]):
                counts["function_or_argument_key_mismatches"] += 1
                examples["function_or_argument_key_mismatches"].append({"id": record_id, "function": source_call["name"]})
            literal_overrides = build_literal_overrides(source["query"], translated["query"])
            walk_arguments(source_call["arguments"], translated_call["arguments"], "", protected_values, overrides, literal_overrides, record_id, translated["query"], counts, examples)
        if protected_tool_view(source["tools"]) != protected_tool_view(translated["tools"]):
            counts["tool_schema_mismatches"] += 1
            examples["tool_schema_mismatches"].append({"id": record_id})
        descriptions: list[tuple[str, str]] = []
        def collect_descriptions(left: Any, right: Any, key: str = "") -> None:
            if isinstance(left, dict) and isinstance(right, dict):
                for child_key in left:
                    collect_descriptions(left[child_key], right[child_key], child_key)
            elif isinstance(left, list) and isinstance(right, list):
                for a, b in zip(left, right):
                    collect_descriptions(a, b, key)
            elif key == "description" and isinstance(left, str) and isinstance(right, str):
                descriptions.append((left, right))
        collect_descriptions(source["tools"], translated["tools"])
        for description_source, description_translated in descriptions:
            counts["descriptions"] += 1
            expected = description_overrides.get(description_source)
            english = len(ENGLISH.findall(description_translated))
            vietnamese = len(VIETNAMESE.findall(description_translated))
            bad = False
            if expected is not None and description_translated != expected:
                bad = True
            elif description_source and (not description_translated or description_source == description_translated or CORRUPTION_PATTERN.search(description_translated)):
                bad = True
            elif english >= 4 and vietnamese >= 1 or english >= 5 and vietnamese == 0:
                bad = True
            if bad:
                counts["description_issues"] += 1
                examples["description_issues"].append({"id": record_id, "source": description_source, "translated": description_translated})
        query = translated["query"]
        unprotected_query = URL.sub("", QUOTED.sub("", query))
        if QUERY_CORRUPTION.search(query) or len(ENGLISH.findall(unprotected_query)) >= 4 or source["query"] == query:
            counts["query_quality_issues"] += 1
            examples["query_quality_issues"].append({"id": record_id, "query": query})
        source_literals = quoted_values(source["query"])
        translated_literals = quoted_values(query)
        if len(source_literals) == len(translated_literals):
            for source_literal, translated_literal in zip(source_literals, translated_literals):
                if source_literal == translated_literal:
                    continue
                english = len(ENGLISH.findall(translated_literal))
                vietnamese = len(VIETNAMESE.findall(translated_literal))
                if QUERY_CORRUPTION.search(translated_literal) or english >= 2 and vietnamese >= 1 or english >= 3 and vietnamese == 0:
                    counts["query_literal_quality_issues"] += 1
                    examples["query_literal_quality_issues"].append({"id": record_id, "source": source_literal, "translated": translated_literal})
        else:
            for source_literal in source_literals:
                if source_literal not in query:
                    counts["lost_query_literals"] += 1
                    examples["lost_query_literals"].append({"id": record_id, "literal": source_literal, "query": query})
    hard_error_keys = {
        "source_json_errors", "translated_json_errors", "source_duplicate_ids", "translated_duplicate_ids",
        "id_order_mismatches", "shape_mismatches", "identity_mismatches", "function_or_argument_key_mismatches",
        "tool_schema_mismatches", "description_issues", "query_quality_issues", "query_literal_quality_issues",
        "lost_query_literals", "query_literal_argument_mismatches", "reviewed_argument_mismatches",
        "protected_argument_changes", "argument_quality_issues",
    }
    line_count_ok = counts["source_lines"] == counts["translated_lines"] == 45593
    passed = line_count_ok and all(counts[key] == 0 for key in hard_error_keys)
    report = {
        "passed": passed,
        "source": str(args.source.resolve()),
        "translated": str(args.translated.resolve()),
        "source_sha256": sha256(args.source),
        "translated_sha256": sha256(args.translated),
        "counts": dict(counts),
        "examples": {key: value[:100] for key, value in examples.items()},
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "counts": dict(counts)}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
