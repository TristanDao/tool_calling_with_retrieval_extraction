"""Audit an xLAM Vietnamese JSONL translation against its English source."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


WORD_RE = re.compile(r"[A-Za-z]{2,}")
NON_LANGUAGE_VALUES = {"python", "none", "true", "false", "null"}


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                yield line_number, None, str(error)
                continue
            if not isinstance(value, dict):
                yield line_number, None, "top-level JSON value is not an object"
                continue
            yield line_number, value, None


PROTECTED_VALUE_KEYS = {"id", "source", "name", "type", "required", "default", "has_tool_call"}


def protected_view(value: Any, key: str | None = None, in_arguments: bool = False) -> Any:
    if isinstance(value, dict):
        return {
            name: protected_view(child, name, in_arguments or name == "arguments")
            for name, child in value.items()
        }
    if isinstance(value, list):
        return [protected_view(child, key, in_arguments) for child in value]
    if (key in PROTECTED_VALUE_KEYS and not in_arguments) or not isinstance(value, str):
        return value
    return "<translated-string>"


def natural_language_candidate(value: str) -> bool:
    compact = value.strip().lower()
    if compact in NON_LANGUAGE_VALUES:
        return False
    return len(WORD_RE.findall(value)) >= 2


def collect_equal_translatable(source: Any, translated: Any, path: str = "", in_arguments: bool = False) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    if isinstance(source, dict) and isinstance(translated, dict):
        for key, source_value in source.items():
            if key not in translated:
                continue
            child_path = f"{path}.{key}" if path else key
            translated_value = translated[key]
            child_in_arguments = in_arguments or key == "arguments"
            is_translatable = key == "description" or (key == "query" and path == "") or child_in_arguments
            if is_translatable and isinstance(source_value, str) and source_value == translated_value and natural_language_candidate(source_value):
                results.append((child_path, source_value))
            else:
                results.extend(collect_equal_translatable(source_value, translated_value, child_path, child_in_arguments))
    elif isinstance(source, list) and isinstance(translated, list):
        for index, (source_value, translated_value) in enumerate(zip(source, translated)):
            results.extend(collect_equal_translatable(source_value, translated_value, f"{path}[{index}]", in_arguments))
    return results


def id_number(value: dict[str, Any]) -> int:
    identifier = value.get("id")
    if not isinstance(identifier, str) or not identifier.startswith("xlam_"):
        return -1
    try:
        return int(identifier.removeprefix("xlam_"))
    except ValueError:
        return -1


def first_difference(source: Any, translated: Any, path: str = "") -> dict[str, Any] | None:
    if type(source) is not type(translated):
        return {"path": path, "source": source, "translated": translated}
    if isinstance(source, dict):
        if list(source) != list(translated):
            return {"path": path, "source_keys": list(source), "translated_keys": list(translated)}
        for key in source:
            difference = first_difference(source[key], translated[key], f"{path}.{key}" if path else key)
            if difference:
                return difference
        return None
    if isinstance(source, list):
        if len(source) != len(translated):
            return {"path": path, "source_length": len(source), "translated_length": len(translated)}
        for index, (left, right) in enumerate(zip(source, translated)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if source != translated:
        return {"path": path, "source": source, "translated": translated}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "source_json_errors": [],
        "translated_json_errors": [],
        "id_mismatches": [],
        "structure_mismatches": [],
        "untranslated_fields": [],
    }
    source_rows = iter(iter_jsonl(args.source))
    translated_rows = iter(iter_jsonl(args.translated))
    source_item = next(source_rows, None)
    translated_item = next(translated_rows, None)
    while source_item is not None or translated_item is not None:
        if source_item is None:
            counts["extra_translated_lines"] += 1
            translated_item = next(translated_rows, None)
            continue
        if translated_item is None:
            counts["missing_translated_lines"] += 1
            source_item = next(source_rows, None)
            continue
        source_line, source, source_error = source_item
        translated_line, translated, translated_error = translated_item
        counts["source_lines"] += 1
        if source_error:
            counts["source_json_errors"] += 1
            if len(examples["source_json_errors"]) < 20:
                examples["source_json_errors"].append({"line": source_line, "error": source_error})
            source_item = next(source_rows, None)
            continue
        if translated_error:
            counts["translated_lines"] += 1
            counts["translated_json_errors"] += 1
            if len(examples["translated_json_errors"]) < 20:
                examples["translated_json_errors"].append({"line": translated_line, "error": translated_error})
            translated_item = next(translated_rows, None)
            continue
        assert source is not None and translated is not None
        source_number = id_number(source)
        translated_number = id_number(translated)
        if source_number < translated_number:
            counts["missing_translated_lines"] += 1
            if len(examples["id_mismatches"]) < 100:
                examples["id_mismatches"].append({"source_line": source_line, "missing_id": source.get("id")})
            source_item = next(source_rows, None)
            continue
        if translated_number < source_number:
            counts["extra_translated_lines"] += 1
            if len(examples["id_mismatches"]) < 100:
                examples["id_mismatches"].append({"translated_line": translated_line, "extra_or_duplicate_id": translated.get("id")})
            translated_item = next(translated_rows, None)
            continue
        counts["translated_lines"] += 1
        counts["aligned_records"] += 1
        source_id = source.get("id")
        translated_id = translated.get("id")
        if source_id != translated_id:
            counts["id_mismatches"] += 1
            if len(examples["id_mismatches"]) < 20:
                examples["id_mismatches"].append({"source_line": source_line, "translated_line": translated_line, "source_id": source_id, "translated_id": translated_id})
        if protected_view(source) != protected_view(translated):
            counts["structure_mismatches"] += 1
            if len(examples["structure_mismatches"]) < 20:
                examples["structure_mismatches"].append(
                    {
                        "source_line": source_line,
                        "translated_line": translated_line,
                        "id": source_id,
                        "difference": first_difference(protected_view(source), protected_view(translated)),
                    }
                )
        untranslated = collect_equal_translatable(source, translated)
        if untranslated:
            counts["records_with_untranslated_fields"] += 1
            counts["untranslated_fields"] += len(untranslated)
            for field_path, _ in untranslated:
                if field_path == "query":
                    counts["untranslated_top_level_queries"] += 1
                elif field_path.endswith(".description"):
                    counts["untranslated_descriptions"] += 1
                elif ".arguments." in field_path:
                    counts["unchanged_argument_strings"] += 1
            if len(examples["untranslated_fields"]) < 100:
                examples["untranslated_fields"].append(
                    {"source_line": source_line, "translated_line": translated_line, "id": source_id, "fields": [{"path": path, "text": text} for path, text in untranslated]}
                )
        else:
            counts["records_without_exact_english_fields"] += 1
        source_item = next(source_rows, None)
        translated_item = next(translated_rows, None)

    report = {"counts": dict(counts), "examples": examples}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
