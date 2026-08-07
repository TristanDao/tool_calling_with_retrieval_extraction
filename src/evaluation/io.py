"""JSONL adapters for canonical gold and heterogeneous model predictions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.evaluation.types import FunctionCall, GoldRecord, PredictionRecord

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_NO_TOOL_CALL_RE = re.compile(r"<no_tool_call\s*/?>", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_arguments(value: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(value, dict):
        return value, True
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}, False
        return (parsed, True) if isinstance(parsed, dict) else ({}, False)
    return {}, False


def _parse_call(value: Any) -> tuple[FunctionCall | None, bool]:
    if not isinstance(value, dict):
        return None, False
    function = value.get("function")
    source = function if isinstance(function, dict) else value
    name = source.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, False
    arguments, valid = _parse_arguments(source.get("arguments", {}))
    return FunctionCall(name=name.strip(), arguments=arguments), valid


def _parse_call_list(values: Any) -> tuple[tuple[FunctionCall, ...], bool]:
    if not isinstance(values, list):
        return (), False
    calls: list[FunctionCall] = []
    valid = True
    for value in values:
        call, item_valid = _parse_call(value)
        valid = valid and item_valid
        if call is not None:
            calls.append(call)
    return tuple(calls), valid


def _parse_raw_output(raw_output: str) -> tuple[tuple[FunctionCall, ...], bool, str | None]:
    if _NO_TOOL_CALL_RE.search(raw_output) and not _TOOL_CALL_RE.search(raw_output):
        return (), True, None
    blocks = _TOOL_CALL_RE.findall(raw_output)
    if not blocks:
        return (), False, "raw_output has no recognized tool-call marker"
    values: list[Any] = []
    for block in blocks:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            return (), False, f"invalid JSON inside <tool_call>: {exc}"
        if isinstance(parsed, list):
            values.extend(parsed)
        else:
            values.append(parsed)
    calls, valid = _parse_call_list(values)
    return calls, valid, None if valid else "invalid tool call in raw_output"


def _legacy_conversation(record: dict[str, Any]) -> tuple[str, tuple[FunctionCall, ...]]:
    conversation = record.get("conversation", [])
    if not isinstance(conversation, list):
        return "", ()
    query = ""
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") == "user" and not query:
            content = turn.get("content", "")
            query = content if isinstance(content, str) else ""
        if turn.get("role") == "assistant" and "function_calls" in turn:
            calls, _ = _parse_call_list(turn.get("function_calls"))
            return query, calls
    return query, ()


def parse_gold_record(record: dict[str, Any]) -> GoldRecord:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Gold record is missing a valid id")
    query = record.get("query", "")
    calls_value = record.get("function_calls")
    if calls_value is None and "conversation" in record:
        legacy_query, calls = _legacy_conversation(record)
        query = query or legacy_query
    else:
        calls, valid = _parse_call_list(calls_value)
        if not valid:
            raise ValueError(f"Gold record {record_id} has invalid function_calls")
    tools = record.get("tools", [])
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError(f"Gold record {record_id} has invalid tools")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return GoldRecord(
        id=record_id,
        source=str(record.get("source", "unknown")),
        query=query if isinstance(query, str) else "",
        function_calls=calls,
        tools=tuple(tools),
        metadata=metadata,
    )


def _ranked_tool_names(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for item in value:
        name = item if isinstance(item, str) else item.get("name") if isinstance(item, dict) else None
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return tuple(names)


def _telemetry(record: dict[str, Any]) -> dict[str, float]:
    raw = record.get("telemetry", {})
    raw = raw if isinstance(raw, dict) else {}
    fields = (
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "gpu_peak_memory_mb",
    )
    result: dict[str, float] = {}
    for field_name in fields:
        value = raw.get(field_name, record.get(field_name))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[field_name] = float(value)
    return result


def parse_prediction_record(record: dict[str, Any]) -> PredictionRecord:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Prediction record is missing a valid id")
    error: str | None = None
    if "function_calls" in record:
        calls, valid = _parse_call_list(record.get("function_calls"))
        if not valid:
            error = "invalid function_calls"
    elif "tool_calls" in record:
        calls, valid = _parse_call_list(record.get("tool_calls"))
        if not valid:
            error = "invalid OpenAI-style tool_calls"
    elif "function_call" in record:
        call, valid = _parse_call(record.get("function_call"))
        calls = (call,) if call is not None else ()
        if not valid:
            error = "invalid function_call"
    elif isinstance(record.get("raw_output"), str):
        calls, valid, error = _parse_raw_output(record["raw_output"])
    elif record.get("no_call") is True:
        calls, valid = (), True
    else:
        calls, valid = (), False
        error = "missing function_calls or raw_output"
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return PredictionRecord(
        id=record_id,
        function_calls=calls,
        ranked_tools=_ranked_tool_names(record.get("ranked_tools")),
        telemetry=_telemetry(record),
        metadata=metadata,
        parse_valid=valid,
        parse_error=error,
    )


def _unique_by_id(records: Iterable[GoldRecord | PredictionRecord], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in records:
        if record.id in result:
            raise ValueError(f"Duplicate {label} id: {record.id}")
        result[record.id] = record
    return result


def load_gold(path: Path) -> list[GoldRecord]:
    return [parse_gold_record(record) for record in load_jsonl(path)]


def load_predictions(path: Path) -> list[PredictionRecord]:
    return [parse_prediction_record(record) for record in load_jsonl(path)]


def align_predictions(
    gold: list[GoldRecord],
    predictions: list[PredictionRecord],
    strict_coverage: bool = True,
) -> list[PredictionRecord]:
    gold_by_id = _unique_by_id(gold, "gold")
    predictions_by_id = _unique_by_id(predictions, "prediction")
    missing = sorted(set(gold_by_id) - set(predictions_by_id))
    extra = sorted(set(predictions_by_id) - set(gold_by_id))
    if strict_coverage and (missing or extra):
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}; "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )
    aligned: list[PredictionRecord] = []
    for item in gold:
        prediction = predictions_by_id.get(item.id)
        if prediction is None:
            prediction = PredictionRecord(
                id=item.id,
                function_calls=(),
                parse_valid=False,
                parse_error="missing prediction",
            )
        aligned.append(prediction)
    return aligned
