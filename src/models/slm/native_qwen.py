"""Render native Qwen3.5 messages and create assistant-only SFT labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def validate_native_row(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate the serialized native row before it reaches a tokenizer."""
    errors: list[str] = []
    messages = row.get("messages")
    tools = row.get("tools")
    if not isinstance(messages, list) or len(messages) < 3:
        errors.append("messages must contain system, user, and assistant")
    if not isinstance(tools, list):
        errors.append("tools must be a list")
    if errors:
        return False, errors
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles[:2] != ["system", "user"] or roles[-1] != "assistant":
        errors.append("messages must start with system/user and end with assistant")
    assistant = messages[-1]
    if not isinstance(assistant, dict):
        errors.append("assistant message is not an object")
        return False, errors
    tool_calls = assistant.get("tool_calls", [])
    if tool_calls is None:
        tool_calls = []
    if not isinstance(tool_calls, list):
        errors.append("assistant.tool_calls must be a list")
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            errors.append("tool call function is invalid")
        if not isinstance(arguments, dict):
            errors.append("tool call arguments must be an object")
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            errors.append("tool definition is invalid")
    if tool_calls and not isinstance(assistant.get("content", ""), str):
        errors.append("assistant content must be a string when tool calls are present")
    return len(errors) == 0, errors


def render_chat_template(
    tokenizer: Any,
    row: dict[str, Any],
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Render a row with the chat_template shipped by its exact checkpoint."""
    valid, errors = validate_native_row(row)
    if not valid:
        raise ValueError(f"Invalid native row {row.get('id')}: {errors}")
    return tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        values = encoded["input_ids"]
    else:
        values = encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def _prefix_length(prefix: Sequence[int], sequence: Sequence[int]) -> int:
    if list(sequence[: len(prefix)]) != list(prefix):
        common = 0
        for left, right in zip(prefix, sequence, strict=False):
            if left != right:
                break
            common += 1
        raise ValueError(
            "Chat template prompt is not a prefix of the training render "
            f"(common tokens: {common}, prompt tokens: {len(prefix)})"
        )
    return len(prefix)


def tokenize_assistant_only(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize a native row and mask system/user tokens from the loss."""
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    valid, errors = validate_native_row(row)
    if not valid:
        raise ValueError(f"Invalid native row {row.get('id')}: {errors}")
    messages = row["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"Native row {row.get('id')} has no final assistant message")
    prompt_row = dict(row)
    prompt_row["messages"] = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_row["messages"],
        tools=prompt_row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = render_chat_template(
        tokenizer,
        row,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = _token_ids(tokenizer, prompt_text)
    full_ids = _token_ids(tokenizer, full_text)
    prompt_length = _prefix_length(prompt_ids, full_ids)
    if prompt_length >= len(full_ids):
        raise ValueError(f"Native row {row.get('id')} has no assistant target tokens")
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]
    if prompt_length >= len(full_ids):
        raise ValueError(f"Native row {row.get('id')} is truncated before assistant target")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * prompt_length + full_ids[prompt_length:],
    }


@dataclass
class AssistantOnlyCollator:
    """Right-pad pre-tokenized rows while preserving the assistant loss mask."""

    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        result: dict[str, Any] = {}
        for key, pad_value in (
            ("input_ids", pad_id),
            ("attention_mask", 0),
            ("labels", self.label_pad_token_id),
        ):
            result[key] = torch.tensor(
                [feature[key] + [pad_value] * (max_length - len(feature[key])) for feature in features],
                dtype=torch.long,
            )
        return result
