"""Convert canonical records to native Qwen3.5 chat messages and tool calls."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SYSTEM_PROMPT_VI = "Bạn là trợ lý AI có khả năng sử dụng công cụ."
SYSTEM_PROMPT_EN = "You are an AI assistant capable of using tools."


def _format_tool_for_chat(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a master-schema tool to the OpenAI-compatible native tool shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        },
    }


def _format_tool_call_for_chat(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Tool-call arguments must be an object: {call.get('name')!r}")
    return {
        "type": "function",
        "function": {
            "name": call.get("name", ""),
            "arguments": arguments,
        },
    }


def _negative_response(language: str) -> str:
    if language == "en":
        return "I cannot complete that request right now."
    return "Hiện tại tôi chưa thể thực hiện yêu cầu này."


def convert_sample(sample: dict[str, Any], language: str | None = None) -> dict[str, Any] | None:
    """Convert one canonical record, retaining negative examples as normal answers."""
    selected_language = language or sample.get("language", "vi")
    if selected_language not in {"en", "vi"}:
        raise ValueError("language must be 'en' or 'vi'")
    query = sample.get("query", "")
    function_calls = sample.get("function_calls", [])
    tools = sample.get("tools", [])
    if not isinstance(query, str) or not query.strip() or not isinstance(function_calls, list):
        return None
    if not isinstance(tools, list) or not tools:
        return None

    formatted_tools: list[dict[str, Any]] = []
    tool_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            return None
        formatted_tool = _format_tool_for_chat(tool)
        formatted_tools.append(formatted_tool)
        tool_names.add(str(tool["name"]))

    formatted_calls: list[dict[str, Any]] = []
    for call in function_calls:
        if not isinstance(call, dict) or call.get("name") not in tool_names:
            return None
        formatted_calls.append(_format_tool_call_for_chat(call))

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_EN if selected_language == "en" else SYSTEM_PROMPT_VI,
        },
        {"role": "user", "content": query},
    ]
    if formatted_calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": formatted_calls})
    else:
        response = sample.get("assistant_content") or sample.get("negative_response")
        if not isinstance(response, str) or not response.strip():
            response = _negative_response(selected_language)
        messages.append({"role": "assistant", "content": response})

    result: dict[str, Any] = {
        "id": sample.get("id"),
        "source": sample.get("source"),
        "language": selected_language,
        "messages": messages,
        "tools": formatted_tools,
    }
    return result


def _write_converted(
    rows: Iterable[tuple[dict[str, Any], str | None]], output_path: Path,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as output:
        for sample, language in rows:
            result = convert_sample(sample, language=language)
            if result is None:
                skipped += 1
                continue
            output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            converted += 1
    return converted, skipped


def convert_rows(
    rows: Iterable[dict[str, Any]], output_path: Path, language: str | None = None,
) -> tuple[int, int]:
    """Convert in-memory records, inferring language from each row when omitted."""
    if language is not None and language not in {"en", "vi"}:
        raise ValueError("language must be 'en' or 'vi'")
    return _write_converted(((row, language) for row in rows), output_path)


def convert_file(
    input_path: Path, output_path: Path, language: str | None = None,
) -> tuple[int, int]:
    """Convert a JSONL file to native Qwen3.5 JSONL rows."""
    def rows() -> Iterable[dict[str, Any]]:
        with input_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    yield {}
                    continue
                yield sample if isinstance(sample, dict) else {}

    return convert_rows(rows(), output_path, language=language)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-splits", action="store_true")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_vi"))
    parser.add_argument("--language", choices=["en", "vi"], default="vi")
    args = parser.parse_args()

    if args.all_splits:
        for split in ("train", "val", "test"):
            input_path = args.benchmark_dir / f"{split}.jsonl"
            output_path = args.benchmark_dir / "instruction" / f"{split}_chat.jsonl"
            if not input_path.exists():
                print(f"[convert] SKIP {split} - {input_path} not found")
                continue
            converted, skipped = convert_file(input_path, output_path, language=args.language)
            print(f"[convert] {split}: {converted} converted, {skipped} skipped -> {output_path}")
        return

    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --all-splits is used")
    converted, skipped = convert_file(args.input, args.output, language=args.language)
    print(f"[convert] {converted} converted, {skipped} skipped -> {args.output}")


if __name__ == "__main__":
    main()
