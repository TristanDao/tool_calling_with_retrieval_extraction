"""Normalize raw EN data → master schema (EN, single-turn + multi-call).

Input:
  data/raw/glaive_raw.jsonl  (multi-turn chat EN)
  data/raw/xlam_raw.jsonl    (flat JSON EN)

Output:
  data/normalized_en/glaive_normalized.jsonl  (positive + optional negative)
  data/normalized_en/xlam_normalized.jsonl    (positive only)
  data/normalized_en/glaive_negative.jsonl    (if --include-negatives)

Master schema:
  {"id": "<source>_<idx>", "source": "glaive|xlam",
   "query": "EN text",
   "function_calls": [{"name": "en_func", "arguments": {"en_key": "value"}}],
   "tools": [{"name": "en_func", "description": "EN text", "parameters": {...}}],
   "has_tool_call": true|false}
"""

from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path
from typing import Any, Callable

from src.data.build_benchmark import (
    _extract_glaive_tool_from_system,
    _safe_json_loads,
    _split_glaive_chat_turns,
)
from src.data.normalize_schema import normalize_tool
from src.data.translate_guidelines import is_snake_case


def _build_tools_for_glaive(
    tool_schema_raw: dict[str, Any] | None,
    function_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if tool_schema_raw:
        normalized = normalize_tool(tool_schema_raw, dataset="glaive")
        tools.append(normalized)
    called_names = {fc["name"] for fc in function_calls}
    for cn in called_names:
        if cn and not any(t.get("name") == cn for t in tools):
            tools.append({
                "name": cn,
                "description": "",
                "feature_group": "",
                "parameters": {"type": "object", "properties": {}},
            })
    return tools


def _parse_glaive(raw: dict[str, Any], idx: int) -> dict[str, Any] | None:
    system_text = raw.get("system", "")
    chat_text = raw.get("chat", "")
    if not chat_text.strip():
        return None

    tool_schema_raw = _extract_glaive_tool_from_system(system_text)
    turns = _split_glaive_chat_turns(chat_text)
    if not turns:
        return None
    if not any(t["role"] == "user" for t in turns):
        return None

    query = ""
    function_calls: list[dict[str, Any]] = []
    past_function_turn = False
    for t in turns:
        if t["role"] == "user":
            if not query:
                query = t["content"]
            else:
                break
        elif t["role"] == "function":
            past_function_turn = True
        elif t["role"] == "assistant" and not past_function_turn:
            for fc in t.get("function_calls", []):
                function_calls.append(fc)

    if not query or not function_calls:
        return None

    return {
        "id": f"glaive_{idx:05d}",
        "source": "glaive",
        "query": query,
        "function_calls": function_calls,
        "tools": _build_tools_for_glaive(tool_schema_raw, function_calls),
        "has_tool_call": True,
    }


def _parse_glaive_negative(raw: dict[str, Any], idx: int) -> dict[str, Any] | None:
    system_text = raw.get("system", "")
    chat_text = raw.get("chat", "")
    if not chat_text.strip():
        return None

    tool_schema_raw = _extract_glaive_tool_from_system(system_text)
    turns = _split_glaive_chat_turns(chat_text)
    if not turns:
        return None

    query = ""
    for t in turns:
        if t["role"] == "user":
            query = t["content"]
            break
    if not query:
        return None

    has_fc = any(
        t["role"] == "assistant" and t.get("function_calls")
        for t in turns
    )
    if has_fc:
        return None

    tools: list[dict[str, Any]] = []
    if tool_schema_raw:
        normalized = normalize_tool(tool_schema_raw, dataset="glaive")
        tools.append(normalized)
    if not tools:
        return None

    return {
        "id": f"glaive_{idx:05d}",
        "source": "glaive",
        "query": query,
        "function_calls": [],
        "tools": tools,
        "has_tool_call": False,
    }


def _parse_xlam(raw: dict[str, Any], idx: int) -> dict[str, Any] | None:
    answers = _safe_json_loads(raw.get("answers", ""))
    tools_raw = _safe_json_loads(raw.get("tools", ""))
    query = raw.get("query", "")
    if not query or not isinstance(answers, list) or not isinstance(tools_raw, list):
        return None

    function_calls: list[dict[str, Any]] = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name", "")
        if not is_snake_case(name):
            continue
        args = ans.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        function_calls.append({"name": name, "arguments": args})
    if not function_calls:
        return None

    tools: list[dict[str, Any]] = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue
        normalized = normalize_tool(t, dataset="xlam")
        tools.append(normalized)

    return {
        "id": f"xlam_{idx:05d}",
        "source": "xlam",
        "query": query,
        "function_calls": function_calls,
        "tools": tools,
        "has_tool_call": True,
    }


def normalize_dataset(
    input_path: Path,
    output_path: Path,
    parser: Callable[[dict[str, Any], int], dict[str, Any] | None],
    source: str,
    negative_parser: Callable[[dict[str, Any], int], dict[str, Any] | None] | None = None,
    negative_output_path: Path | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if negative_output_path:
        negative_output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    parsed_positive = 0
    parsed_negative = 0
    skipped_no_fc = 0
    skipped_parse = 0
    n_calls: list[int] = []

    neg_file = open(negative_output_path, "w", encoding="utf-8") if negative_output_path else None
    try:
        with output_path.open("w", encoding="utf-8") as out:
            with input_path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_parse += 1
                        continue

                    sample = parser(raw, idx)
                    if sample is not None:
                        out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        parsed_positive += 1
                        n_calls.append(len(sample["function_calls"]))
                        continue

                    if negative_parser is not None and neg_file is not None:
                        neg = negative_parser(raw, idx)
                        if neg is not None:
                            neg_file.write(json.dumps(neg, ensure_ascii=False) + "\n")
                            parsed_negative += 1
                            continue

                    skipped_no_fc += 1
    finally:
        if neg_file:
            neg_file.close()

    stats = {
        "source": source,
        "total_raw": total,
        "parsed": parsed_positive,
        "parsed_positive": parsed_positive,
        "parsed_negative": parsed_negative,
        "skipped_no_fc": skipped_no_fc,
        "skipped_parse_error": skipped_parse,
        "calls_per_sample": {
            "min": min(n_calls) if n_calls else 0,
            "max": max(n_calls) if n_calls else 0,
            "mean": round(sum(n_calls) / max(len(n_calls), 1), 2),
        },
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw EN → master schema (single-turn + multi-call)"
    )
    parser.add_argument(
        "--input-glaive", type=Path,
        default=Path("data/raw/glaive_raw.jsonl"),
    )
    parser.add_argument(
        "--input-xlam", type=Path,
        default=Path("data/raw/xlam_raw.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/normalized_en"),
    )
    parser.add_argument(
        "--source", type=str, choices=["glaive", "xlam", "all"], default="all",
    )
    parser.add_argument(
        "--include-negatives", action="store_true",
        help="Also extract Glaive samples without function calls as negatives",
    )
    args = parser.parse_args()

    all_stats: dict[str, dict[str, Any]] = {}

    glaive_total = 0
    if args.source in ("glaive", "all"):
        out = args.output_dir / "glaive_normalized.jsonl"
        neg_out = args.output_dir / "glaive_negative.jsonl" if args.include_negatives else None
        neg_parser = _parse_glaive_negative if args.include_negatives else None
        stats = normalize_dataset(
            args.input_glaive, out, _parse_glaive, "glaive",
            negative_parser=neg_parser, negative_output_path=neg_out,
        )
        all_stats["glaive"] = stats
        glaive_total = stats["parsed_positive"] + stats["parsed_negative"]
        print(f"[normalize] glaive: {stats['total_raw']} raw → {stats['parsed_positive']} positive")
        if args.include_negatives:
            print(f"[normalize] glaive: {stats['parsed_negative']} negative → {neg_out}")

    if args.source in ("xlam", "all"):
        out = args.output_dir / "xlam_normalized.jsonl"
        stats = normalize_dataset(args.input_xlam, out, _parse_xlam, "xlam")
        all_stats["xlam"] = stats
        print(f"[normalize] xlam: {stats['total_raw']} raw → {stats['parsed_positive']} positive")

    total_pos = sum(s["parsed_positive"] for s in all_stats.values())
    total_neg = sum(s.get("parsed_negative", 0) for s in all_stats.values())
    total_raw = sum(s["total_raw"] for s in all_stats.values())
    print(f"[normalize] TOTAL: {total_raw} raw → {total_pos} positive + {total_neg} negative")

    stats_path = args.output_dir / "normalize_stats.yaml"
    with stats_path.open("w") as f:
        yaml.dump(all_stats, f, allow_unicode=True, default_flow_style=False)
    print(f"[normalize] stats → {stats_path}")


if __name__ == "__main__":
    main()
