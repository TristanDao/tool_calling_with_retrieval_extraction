"""Run the final integrity and translation-quality audit for xLAM JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from repair_xlam_translation_integrity import (
    CORRUPTION_PATTERN,
    collect_protected_values,
    is_technical_value,
    load_argument_overrides,
)
from scan_xlam_argument_quality import classify, looks_natural
from scan_xlam_query_quality import CORRUPTION as QUERY_CORRUPTION
from scan_xlam_query_quality import ENGLISH, QUOTED, URL


QUERY_ENGLISH_ALLOWLIST = {"xlam_40754"}
UNCHANGED_QUERY_ALLOWLIST = {"xlam_35157", "xlam_44363"}


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


def argument_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: argument_keys(child) for key, child in value.items()}
    if isinstance(value, list):
        return [argument_keys(child) for child in value]
    return type(value).__name__


def walk_arguments(
    source: Any,
    translated: Any,
    key: str,
    protected_values: set[str],
    overrides: dict[str, dict[str, str]],
    record_id: str,
    query_source: str,
    query_translated: str,
    counts: Counter[str],
    examples: dict[str, list[dict[str, Any]]],
) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, source_child in source.items():
            walk_arguments(
                source_child,
                translated[child_key],
                child_key,
                protected_values,
                overrides,
                record_id,
                query_source,
                query_translated,
                counts,
                examples,
            )
        return
    if isinstance(source, list) and isinstance(translated, list):
        for source_child, translated_child in zip(source, translated):
            walk_arguments(
                source_child,
                translated_child,
                key,
                protected_values,
                overrides,
                record_id,
                query_source,
                query_translated,
                counts,
                examples,
            )
        return
    if not isinstance(source, str) or not isinstance(translated, str):
        return
    reviewed = overrides.get(key, {}).get(source)
    technical = is_technical_value(key, source, protected_values)
    if reviewed is not None:
        counts["reviewed_argument_values"] += 1
        if translated != reviewed:
            counts["reviewed_argument_mismatches"] += 1
            examples["reviewed_argument_mismatches"].append(
                {"id": record_id, "key": key, "source": source, "expected": reviewed, "actual": translated}
            )
    elif technical:
        counts["protected_argument_values"] += 1
        if translated != source:
            counts["protected_argument_changes"] += 1
            examples["protected_argument_changes"].append(
                {"id": record_id, "key": key, "source": source, "translated": translated}
            )
    quality = None if technical else classify(key, source, translated, overrides)
    if quality is not None:
        category = quality[0]
        if category == "unchanged_natural_candidate":
            counts["reviewed_preserved_argument_occurrences"] += 1
        else:
            counts["actionable_argument_quality_issues"] += 1
            examples["actionable_argument_quality_issues"].append(
                {"id": record_id, "key": key, "category": category, "source": source, "translated": translated}
            )
    if source != translated:
        counts["translated_argument_values"] += 1
        pattern_source = re.compile(r"(?P<quote>['\"])" + re.escape(source) + r"(?P=quote)")
        if pattern_source.search(query_source):
            counts["changed_quoted_argument_literals"] += 1
            pattern_translated = re.compile(r"(?P<quote>['\"])" + re.escape(translated) + r"(?P=quote)")
            if not pattern_translated.search(query_translated):
                if pattern_source.search(query_translated):
                    counts["stale_query_argument_literals"] += 1
                    examples["stale_query_argument_literals"].append(
                        {"id": record_id, "source": source, "translated": translated, "query": query_translated}
                    )
                else:
                    counts["query_argument_literal_semantic_variants"] += 1


def main() -> None:
    args = parse_args()
    source_records, source_errors = load_jsonl(args.source)
    translated_records, translated_errors = load_jsonl(args.translated)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "shape_mismatches": [],
        "identity_mismatches": [],
        "tool_schema_mismatches": [],
        "description_issues": [],
        "query_issues": [],
        "protected_argument_changes": [],
        "reviewed_argument_mismatches": [],
        "actionable_argument_quality_issues": [],
        "stale_query_argument_literals": [],
    }
    counts["source_json_errors"] = len(source_errors)
    counts["translated_json_errors"] = len(translated_errors)
    counts["source_lines"] = len(source_records) + len(source_errors)
    counts["translated_lines"] = len(translated_records) + len(translated_errors)
    protected_values: set[str] = set()
    for record in source_records:
        for call in record["function_calls"]:
            collect_protected_values(call["arguments"], "", protected_values)
    overrides = load_argument_overrides(args.translations_dir)
    for source, translated in zip(source_records, translated_records):
        record_id = source.get("id", "")
        counts["aligned_records"] += 1
        if not same_shape(source, translated):
            counts["shape_mismatches"] += 1
            examples["shape_mismatches"].append({"id": record_id})
        if source.get("id") != translated.get("id") or source.get("source") != translated.get("source"):
            counts["identity_mismatches"] += 1
            examples["identity_mismatches"].append({"id": record_id})
        source_calls = source["function_calls"]
        translated_calls = translated["function_calls"]
        for source_call, translated_call in zip(source_calls, translated_calls):
            if source_call["name"] != translated_call["name"] or argument_keys(source_call["arguments"]) != argument_keys(translated_call["arguments"]):
                counts["function_or_argument_key_mismatches"] += 1
                examples["identity_mismatches"].append({"id": record_id, "function": source_call["name"]})
            walk_arguments(
                source_call["arguments"],
                translated_call["arguments"],
                "",
                protected_values,
                overrides,
                record_id,
                source["query"],
                translated["query"],
                counts,
                examples,
            )
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
            if not description_translated or CORRUPTION_PATTERN.search(description_translated):
                counts["description_issues"] += 1
                examples["description_issues"].append(
                    {"id": record_id, "source": description_source, "translated": description_translated}
                )
        query = translated["query"]
        unprotected_query = URL.sub("", QUOTED.sub("", query))
        query_english = len(ENGLISH.findall(unprotected_query))
        query_issue = bool(QUERY_CORRUPTION.search(query)) or query_english >= 4
        if query_issue and record_id not in QUERY_ENGLISH_ALLOWLIST:
            counts["query_quality_issues"] += 1
            examples["query_issues"].append({"id": record_id, "query": query})
        if source["query"] == query and record_id not in UNCHANGED_QUERY_ALLOWLIST:
            counts["unexpected_unchanged_queries"] += 1
            examples["query_issues"].append({"id": record_id, "query": query})
    hard_error_keys = {
        "source_json_errors",
        "translated_json_errors",
        "shape_mismatches",
        "identity_mismatches",
        "function_or_argument_key_mismatches",
        "tool_schema_mismatches",
        "description_issues",
        "query_quality_issues",
        "unexpected_unchanged_queries",
        "protected_argument_changes",
        "reviewed_argument_mismatches",
        "actionable_argument_quality_issues",
        "stale_query_argument_literals",
    }
    line_count_ok = counts["source_lines"] == counts["translated_lines"] == 60000
    passed = line_count_ok and all(counts[key] == 0 for key in hard_error_keys)
    report = {
        "passed": passed,
        "source": str(args.source.resolve()),
        "translated": str(args.translated.resolve()),
        "source_sha256": sha256(args.source),
        "translated_sha256": sha256(args.translated),
        "counts": dict(counts),
        "allowlists": {
            "query_english": sorted(QUERY_ENGLISH_ALLOWLIST),
            "unchanged_queries": sorted(UNCHANGED_QUERY_ALLOWLIST),
        },
        "examples": {key: value[:100] for key, value in examples.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "counts": dict(counts)}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
