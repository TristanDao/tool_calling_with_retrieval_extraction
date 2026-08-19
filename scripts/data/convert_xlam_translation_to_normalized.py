#!/usr/bin/env python
"""Convert successful raw xLAM translations to the master schema."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.data.build_benchmark import _load_failed_indices, parse_xlam_sample
from src.data.translate import read_jsonl_generator
from src.data.translation_checkpoint import Checkpoint, save_atomic


def _merge_argument_values(original: Any, translated: Any) -> Any:
    if isinstance(original, dict):
        translated_dict = translated if isinstance(translated, dict) else {}
        return {
            key: _merge_argument_values(value, translated_dict.get(key))
            for key, value in original.items()
        }
    if isinstance(original, list):
        translated_list = translated if isinstance(translated, list) else []
        return [
            _merge_argument_values(value, translated_list[i] if i < len(translated_list) else None)
            for i, value in enumerate(original)
        ]
    return translated if translated is not None else original


def _merge_descriptions(original: Any, translated: Any) -> Any:
    if isinstance(original, dict):
        translated_dict = translated if isinstance(translated, dict) else {}
        return {
            key: translated_dict[key] if key == "description" and key in translated_dict
            else _merge_descriptions(value, translated_dict.get(key))
            for key, value in original.items()
        }
    if isinstance(original, list):
        translated_list = translated if isinstance(translated, list) else []
        return [
            _merge_descriptions(value, translated_list[i] if i < len(translated_list) else None)
            for i, value in enumerate(original)
        ]
    return original


def _repair_json_list_fields(sample: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(sample)
    for field in ("answers", "tools"):
        value = repaired.get(field)
        if not isinstance(value, str) or not value.lstrip().startswith("["):
            continue
        try:
            json.loads(value)
        except json.JSONDecodeError:
            candidates = [value.rstrip() + "]"]
            if value.rstrip().endswith("}"):
                candidates.append(value.rstrip()[:-1] + "]")
            for candidate in candidates:
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                repaired[field] = candidate
                break
    return repaired


def convert(
    source_path: Path,
    normalized_source_path: Path,
    translated_path: Path,
    failed_path: Path,
    output_path: Path,
    output_failed_path: Path,
    checkpoint_path: Path,
    last_processed_index: int,
) -> dict[str, int]:
    failed_indices = _load_failed_indices(failed_path)
    translated = iter(read_jsonl_generator(translated_path))
    normalized_source = iter(read_jsonl_generator(normalized_source_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_failed_path.parent.mkdir(parents=True, exist_ok=True)
    n_success = 0
    n_failed = 0
    n_fallback = 0

    with output_path.open("w", encoding="utf-8") as output:
        for source_index, _source_sample in read_jsonl_generator(source_path, end=last_processed_index):
            normalized_index, normalized_original = next(normalized_source)
            if source_index != normalized_index:
                raise RuntimeError(f"source index mismatch: {source_index} != {normalized_index}")
            if source_index in failed_indices:
                n_failed += 1
                continue
            _, translated_sample = next(translated)
            translated_sample = _repair_json_list_fields(translated_sample)
            normalized = parse_xlam_sample(translated_sample, source_index)
            if normalized is None:
                fallback_sample = dict(translated_sample)
                fallback_sample["tools"] = json.dumps(
                    normalized_original.get("tools", []), ensure_ascii=False
                )
                normalized = parse_xlam_sample(fallback_sample, source_index)
                n_fallback += 1
            if normalized is None:
                fallback_sample["answers"] = json.dumps(
                    normalized_original.get("function_calls", []), ensure_ascii=False
                )
                normalized = parse_xlam_sample(fallback_sample, source_index)
            if normalized is None:
                raise RuntimeError(f"cannot parse translated sample at index {source_index}")

            expected_calls = normalized_original.get("function_calls", [])
            translated_calls = normalized["function_calls"][:len(expected_calls)]
            normalized["function_calls"] = [
                {
                    "name": original_call["name"],
                    "arguments": _merge_argument_values(
                        original_call.get("arguments", {}),
                        translated_call.get("arguments", {}) if i < len(translated_calls) else original_call.get("arguments", {}),
                    ),
                }
                for i, original_call in enumerate(expected_calls)
                for translated_call in [translated_calls[i] if i < len(translated_calls) else {}]
            ]
            if len(translated_calls) < len(expected_calls):
                n_fallback += 1

            translated_tools = {
                tool.get("name"): tool
                for tool in normalized.get("tools", [])
                if isinstance(tool, dict)
            }
            normalized["tools"] = [
                {
                    **original_tool,
                    "description": translated_tools.get(original_tool.get("name"), {}).get(
                        "description", original_tool.get("description", "")
                    ),
                    "parameters": _merge_descriptions(
                        original_tool.get("parameters", {}),
                        translated_tools.get(original_tool.get("name"), {}).get("parameters", {}),
                    ),
                }
                for original_tool in normalized_original.get("tools", [])
            ]
            normalized["id"] = normalized_original["id"]
            normalized["source"] = normalized_original["source"]
            normalized["has_tool_call"] = True
            output.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            n_success += 1
        output.flush()
        os.fsync(output.fileno())

    shutil.copyfile(failed_path, output_failed_path)
    save_atomic(
        Checkpoint(
            dataset="xlam_normalized",
            last_processed_index=last_processed_index,
            total_success=n_success,
            total_failed=n_failed,
        ),
        checkpoint_path,
    )
    return {
        "success": n_success,
        "failed": n_failed,
        "fallback": n_fallback,
        "last_processed_index": last_processed_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/raw/xlam_raw.jsonl"))
    parser.add_argument("--normalized-source", type=Path, default=Path("data/normalized_en/xlam_normalized.jsonl"))
    parser.add_argument("--translated", type=Path, default=Path("data/translations/xlam_vi.jsonl"))
    parser.add_argument("--failed", type=Path, default=Path("data/translations/failed/xlam_failed.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/translations/xlam_normalized_vi.jsonl"))
    parser.add_argument("--output-failed", type=Path, default=Path("data/translations/failed/xlam_normalized_failed.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/translations/.checkpoint/xlam_normalized.json"))
    parser.add_argument("--last-processed-index", type=int, default=31380)
    args = parser.parse_args()

    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("xLAM normalized output/checkpoint already exists")
    result = convert(
        args.source,
        args.normalized_source,
        args.translated,
        args.failed,
        args.output,
        args.output_failed,
        args.checkpoint,
        args.last_processed_index,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
