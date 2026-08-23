"""Typed records shared by evaluation modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class GoldRecord:
    id: str
    source: str
    query: str
    function_calls: tuple[FunctionCall, ...]
    tools: tuple[dict[str, Any], ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_positive(self) -> bool:
        return bool(self.function_calls)

    @property
    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {
            str(tool.get("name")): tool
            for tool in self.tools
            if isinstance(tool, dict) and tool.get("name")
        }


@dataclass(frozen=True)
class PredictionRecord:
    id: str
    function_calls: tuple[FunctionCall, ...]
    ranked_tools: tuple[str, ...] | None = None
    telemetry: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parse_valid: bool = True
    parse_error: str | None = None

    @property
    def predicts_call(self) -> bool:
        return bool(self.function_calls)


@dataclass(frozen=True)
class CallAlignment:
    gold: FunctionCall | None
    prediction: FunctionCall | None
