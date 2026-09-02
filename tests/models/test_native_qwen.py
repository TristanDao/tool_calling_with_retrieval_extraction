"""Tests for native Qwen3.5 rendering and assistant-only labels."""

from __future__ import annotations

from typing import Any

import pytest
from src.models.slm.native_qwen import (
    render_chat_template,
    tokenize_assistant_only,
    validate_native_row,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tools, tokenize, enable_thinking
        rendered = "".join(f"<{message['role']}>{message.get('content', '')}" for message in messages)
        if messages and messages[-1].get("tool_calls"):
            rendered += "<tool_call>" + messages[-1]["tool_calls"][0]["function"]["name"]
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _row() -> dict[str, Any]:
    return {
        "id": "sample_1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "query"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "lookup", "arguments": {"x": 1}}}
                ],
            },
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
    }


def test_validate_native_row():
    valid, errors = validate_native_row(_row())
    assert valid, errors


def test_render_uses_tokenizer_template():
    rendered = render_chat_template(FakeTokenizer(), _row())
    assert "<tool_call>lookup" in rendered


def test_tokenize_masks_prompt_only():
    features = tokenize_assistant_only(FakeTokenizer(), _row(), max_seq_length=1000)
    first_target = next(index for index, label in enumerate(features["labels"]) if label != -100)
    assert first_target > 0
    assert all(label == -100 for label in features["labels"][:first_target])
    assert all(label != -100 for label in features["labels"][first_target:])


def test_tokenize_fails_when_target_is_truncated():
    with pytest.raises(ValueError, match="truncated"):
        tokenize_assistant_only(FakeTokenizer(), _row(), max_seq_length=10)
