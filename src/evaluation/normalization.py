"""Deterministic schema-aware normalization for argument comparison."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.evaluation.config import NormalizationConfig
from src.evaluation.types import FunctionCall

_SPACE_RE = re.compile(r"\s+")
_SCALED_NUMBER_RE = re.compile(
    r"^([+-]?[0-9]+(?:[.,][0-9]+)?)\s*(nghìn|ngàn|triệu|tỷ)$",
    re.IGNORECASE,
)
_SCALES = {"nghìn": 1_000, "ngàn": 1_000, "triệu": 1_000_000, "tỷ": 1_000_000_000}
_TRUE_VALUES = {"true", "1", "yes", "y", "có", "đúng"}
_FALSE_VALUES = {"false", "0", "no", "n", "không", "sai"}
_IDENTIFIER_RE = re.compile(
    r"(?:^|_)(?:id|uuid|iban|account_number|card_number|passport_number|"
    r"visa_number|insurance_number)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArgumentNormalizer:
    config: NormalizationConfig

    def normalize_text(self, value: str) -> str:
        result = unicodedata.normalize(self.config.unicode_form, value)
        if self.config.trim_whitespace:
            result = result.strip()
        if self.config.collapse_whitespace:
            result = _SPACE_RE.sub(" ", result)
        if self.config.casefold_strings:
            result = result.casefold()
        return result

    def _alias(self, value: Any, tool_name: str, path: str) -> Any:
        if not isinstance(value, str):
            return value
        key = self.normalize_text(value)
        parameter_key = f"{tool_name}.{path}" if path else tool_name
        local = self.config.parameter_aliases.get(parameter_key, {})
        for aliases in (local, self.config.global_aliases):
            normalized_aliases = {
                self.normalize_text(str(alias)): replacement for alias, replacement in aliases.items()
            }
            if key in normalized_aliases:
                return normalized_aliases[key]
        return value

    def _is_identifier(self, schema: dict[str, Any], path: str) -> bool:
        schema_format = str(schema.get("format", "")).casefold()
        explicit = schema.get("x-identifier") is True or schema.get("x-sensitive") is True
        return explicit or schema_format in {"uuid", "iban"} or bool(
            _IDENTIFIER_RE.search(path.split(".")[-1])
        )

    def _identifier(self, value: str) -> str:
        result = unicodedata.normalize(self.config.unicode_form, value)
        if self.config.trim_whitespace:
            result = result.strip()
        return result.casefold() if self.config.casefold_identifiers else result

    def _number(self, value: Any, integer: bool) -> int | float | Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                return value
            return int(value) if integer and float(value).is_integer() else value
        if not isinstance(value, str):
            return value
        text = self.normalize_text(value).replace("\u00a0", " ")
        scaled = _SCALED_NUMBER_RE.fullmatch(text)
        if scaled:
            base = float(scaled.group(1).replace(",", "."))
            result = base * _SCALES[scaled.group(2).casefold()]
            return int(result) if integer and result.is_integer() else result
        compact = text.replace(" ", "")
        if integer:
            separators = [separator for separator in (",", ".") if separator in compact]
            if len(separators) == 1:
                separator = separators[0]
                groups = compact.split(separator)
                if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
                    compact = "".join(groups)
                elif len(groups) == 2:
                    compact = groups[0] + "." + groups[1]
                else:
                    return value
            elif len(separators) == 2:
                if compact.rfind(",") > compact.rfind("."):
                    compact = compact.replace(".", "").replace(",", ".")
                else:
                    compact = compact.replace(",", "")
        elif "," in compact and "." in compact:
            if compact.rfind(",") > compact.rfind("."):
                compact = compact.replace(".", "").replace(",", ".")
            else:
                compact = compact.replace(",", "")
        elif compact.count(",") == 1:
            left, right = compact.split(",")
            compact = left + right if len(right) == 3 else left + "." + right
        elif compact.count(".") > 1:
            compact = compact.replace(".", "")
        try:
            number = float(compact)
        except ValueError:
            return value
        if integer:
            return int(number) if number.is_integer() else value
        return int(number) if number.is_integer() else number

    def _boolean(self, value: Any) -> bool | Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (str, int)):
            normalized = self.normalize_text(str(value))
            if normalized in _TRUE_VALUES:
                return True
            if normalized in _FALSE_VALUES:
                return False
        return value

    def _date(self, value: str, schema_format: str) -> str:
        text = self.normalize_text(value)
        if schema_format == "date":
            for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(text, date_format).date().isoformat()
                except ValueError:
                    continue
            return text
        candidate = text.replace("z", "+00:00")
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return text
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat()

    def normalize_value(
        self,
        value: Any,
        schema: dict[str, Any] | None,
        tool_name: str,
        path: str,
    ) -> Any:
        schema = schema or {}
        if isinstance(value, str) and self._is_identifier(schema, path):
            return self._identifier(value)
        value = self._alias(value, tool_name, path)
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            non_null = [item for item in schema_type if item != "null"]
            schema_type = non_null[0] if len(non_null) == 1 else None
        if schema_type == "integer" and self.config.coerce_numbers:
            return self._number(value, integer=True)
        if schema_type == "number" and self.config.coerce_numbers:
            return self._number(value, integer=False)
        if schema_type == "boolean" and self.config.coerce_booleans:
            return self._boolean(value)
        schema_format = str(schema.get("format", "")).casefold()
        if (
            schema_type == "string"
            and isinstance(value, str)
            and self.config.normalize_dates
            and schema_format in {"date", "date-time"}
        ):
            return self._date(value, schema_format)
        if schema_type == "array" and isinstance(value, list):
            item_schema = schema.get("items", {})
            normalized = [
                self.normalize_value(item, item_schema, tool_name, f"{path}[]") for item in value
            ]
            full_path = f"{tool_name}.{path}" if path else tool_name
            if full_path in self.config.unordered_array_paths:
                return sorted(normalized, key=canonical_json)
            return normalized
        if schema_type == "object" and isinstance(value, dict):
            properties = schema.get("properties", {})
            return {
                str(key): self.normalize_value(
                    child,
                    properties.get(key, {}),
                    tool_name,
                    f"{path}.{key}" if path else str(key),
                )
                for key, child in value.items()
            }
        if isinstance(value, str):
            return self.normalize_text(value)
        return value

    def normalize_arguments(
        self,
        call: FunctionCall,
        tool_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        parameters = (tool_schema or {}).get("parameters", {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        return {
            key: self.normalize_value(value, properties.get(key, {}), call.name, key)
            for key, value in call.arguments.items()
        }

    def normalize_call(
        self,
        call: FunctionCall,
        tool_schema: dict[str, Any] | None,
    ) -> FunctionCall:
        return FunctionCall(
            name=call.name,
            arguments=self.normalize_arguments(call, tool_schema),
        )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def flatten_arguments(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                flattened.update(flatten_arguments(child, path))
            else:
                flattened[path] = child
        return flattened
    return {prefix: value}
