"""Parse native Qwen3.5 tool-call output into the shared evaluation contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_RE = re.compile(
    r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE
)
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE
)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """One normalized function call from model output."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    """Normalized output and parser diagnostics."""

    calls: list[ParsedToolCall]
    errors: list[str]

    @property
    def is_negative(self) -> bool:
        return not self.calls and not self.errors


def _parse_parameter_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def _parse_native(text: str) -> tuple[list[ParsedToolCall], list[str]]:
    calls: list[ParsedToolCall] = []
    errors: list[str] = []
    opening_tags = len(re.findall(r"<tool_call\b", text, re.IGNORECASE))
    closing_tags = len(re.findall(r"</tool_call\s*>", text, re.IGNORECASE))
    if opening_tags != closing_tags:
        errors.append("unbalanced native tool_call tags")
    for block in _TOOL_BLOCK_RE.findall(text):
        function_matches = list(_FUNCTION_RE.finditer(block))
        if not function_matches:
            errors.append("tool_call block has no function block")
            continue
        for function_match in function_matches:
            name = function_match.group(1).strip()
            body = function_match.group(2)
            arguments: dict[str, Any] = {}
            for parameter_match in _PARAMETER_RE.finditer(body):
                parameter_name = parameter_match.group(1).strip()
                if parameter_name in arguments:
                    errors.append(f"duplicate parameter: {parameter_name}")
                    continue
                arguments[parameter_name] = _parse_parameter_value(parameter_match.group(2))
            calls.append(ParsedToolCall(name=name, arguments=arguments))
    return calls, errors


def _parse_json_fallback(text: str) -> tuple[list[ParsedToolCall], list[str]]:
    candidate = text.strip()
    fenced = _CODE_FENCE_RE.match(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return [], []
    if isinstance(value, dict) and "name" in value:
        candidates: list[Any] = [value]
    elif isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        candidates = value["tool_calls"]
    elif isinstance(value, list):
        candidates = value
    else:
        return [], []
    calls: list[ParsedToolCall] = []
    errors: list[str] = []
    for item in candidates:
        if not isinstance(item, dict):
            errors.append("JSON tool call is not an object")
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append("JSON tool call has no function name")
            continue
        arguments = function.get("arguments", {})
        if not isinstance(arguments, dict):
            errors.append("JSON tool-call arguments are not an object")
            arguments = {}
        calls.append(ParsedToolCall(name.strip(), arguments))
    return calls, errors


def parse_tool_output(text: str, valid_tool_names: set[str] | None = None) -> ParsedOutput:
    """Parse native XML calls, with a strict JSON fallback for API adapters."""
    native_calls, errors = _parse_native(text or "")
    calls = native_calls
    if not calls and not errors:
        if re.search(r"<tool_call\b|</tool_call\s*>", text or "", re.IGNORECASE):
            errors.append("malformed native tool call")
        else:
            calls, errors = _parse_json_fallback(text or "")
    if valid_tool_names is not None:
        errors.extend(
            f"unknown tool: {call.name}"
            for call in calls
            if call.name not in valid_tool_names
        )
    return ParsedOutput(calls=calls, errors=errors)
