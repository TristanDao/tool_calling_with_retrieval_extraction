#!/usr/bin/env python
"""Smoke-test native Qwen3.5 tool-call rendering for one or more checkpoints."""

from __future__ import annotations

import argparse
from typing import Any

from src.models.slm.native_qwen import render_chat_template, tokenize_assistant_only


def _smoke_rows() -> list[dict[str, Any]]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_tutors",
                "description": "Find tutors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "online": {"type": "boolean"},
                        "scores": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["subject"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_courses",
                "description": "Find courses.",
                "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}},
            },
        },
    ]
    return [
        {
            "id": "smoke_positive",
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý."},
                {"role": "user", "content": "Tìm gia sư Toán học trực tuyến."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search_tutors",
                                "arguments": {
                                    "subject": "Toán học",
                                    "online": True,
                                    "scores": [8.5, 9],
                                },
                            },
                        }
                    ],
                },
            ],
            "tools": tools,
        },
        {
            "id": "smoke_multi",
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý."},
                {"role": "user", "content": "Tìm gia sư và khóa học."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "search_tutors", "arguments": {"subject": "Toán"}},
                        },
                        {
                            "type": "function",
                            "function": {"name": "search_courses", "arguments": {"topic": "Python"}},
                        },
                    ],
                },
            ],
            "tools": tools,
        },
        {
            "id": "smoke_negative",
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý."},
                {"role": "user", "content": "Hãy giải thích khái niệm này."},
                {"role": "assistant", "content": "Tôi sẽ giải thích khái niệm này bằng kiến thức hiện có."},
            ],
            "tools": tools,
        },
    ]


def _assistant_segment(rendered: str) -> str:
    marker = "<|im_start|>assistant\n"
    return rendered.rsplit(marker, 1)[-1]


def smoke_model(model_name: str, max_seq_length: int) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Native Qwen smoke test requires transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    rows = _smoke_rows()
    for row in rows:
        rendered = render_chat_template(tokenizer, row, enable_thinking=False)
        assistant = _assistant_segment(rendered)
        if "# Tools" not in rendered:
            raise AssertionError(f"{model_name}: tool definitions were not rendered")
        if row["id"] == "smoke_negative":
            if "<tool_call>" in assistant:
                raise AssertionError(f"{model_name}: negative row rendered a tool call")
        else:
            expected_calls = len(row["messages"][-1]["tool_calls"])
            if assistant.count("<tool_call>") != expected_calls:
                raise AssertionError(f"{model_name}: expected {expected_calls} native calls")
        tokenize_assistant_only(tokenizer, row, max_seq_length=max_seq_length)
    print(f"[native-smoke] {model_name}: {len(rows)} rows passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=None,
        help="Checkpoint ID; repeat for multiple models.",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()
    models = args.models or ["unsloth/Qwen3.5-4B", "unsloth/Qwen3.5-2B"]
    for model_name in models:
        smoke_model(model_name, args.max_seq_length)


if __name__ == "__main__":
    main()
