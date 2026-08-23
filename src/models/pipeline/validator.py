"""JSON Schema validator + coercion cho argument đã trích xuất (§Phase 4).

Đây là mắt xích khiến Method 2 **không thể** sinh JSON sai cú pháp: JSON được
*lắp ráp* từ schema chứ không được *sinh ra*. Validator chỉ còn phải lo hai việc:

1. **Coerce type** theo đúng khai báo schema (string "30" → int 30 khi
   `type: integer`), không đoán ngoài schema.
2. **Required thiếu** → thử lại param đó với ngưỡng `has_value` thấp hơn (0.3);
   vẫn thiếu thì đánh dấu `incomplete` — error class `I` của §9 experimental_plan.

Key không có trong schema bị loại: ArgA yêu cầu không được thừa key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.models.crossencoder.data_collator import UNSUPPORTED_TYPES, iter_parameters
from src.models.crossencoder.normalize import NormalizerConfig, SpanNormalizer

STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete"
STATUS_INVALID = "invalid"


@dataclass
class ValidationResult:
    arguments: dict[str, Any]
    status: str = STATUS_OK
    missing_required: list[str] = field(default_factory=list)
    dropped_keys: list[str] = field(default_factory=list)
    coerced_keys: list[str] = field(default_factory=list)
    unsupported_params: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


class ArgumentValidator:
    def __init__(
        self,
        fallback_threshold: float = 0.3,
        normalizer: SpanNormalizer | None = None,
        use_jsonschema: bool = True,
    ) -> None:
        self.fallback_threshold = fallback_threshold
        self.normalizer = normalizer or SpanNormalizer(NormalizerConfig())
        self.use_jsonschema = use_jsonschema

    def validate(
        self,
        arguments: dict[str, Any],
        tool_schema: dict[str, Any],
        predictions: Sequence[Any] | None = None,
    ) -> ValidationResult:
        params = {p["name"]: p for p in iter_parameters(tool_schema)}
        result = ValidationResult(arguments={})

        for key, value in arguments.items():
            param = params.get(key)
            if param is None:
                result.dropped_keys.append(key)
                continue
            if param["routing_type"] in UNSUPPORTED_TYPES:
                result.unsupported_params.append(key)
                continue
            coerced, changed = self._coerce(value, param)
            if coerced is None:
                result.dropped_keys.append(key)
                continue
            if changed:
                result.coerced_keys.append(key)
            result.arguments[key] = coerced

        self._recover_missing_required(result, params, predictions)

        if self.use_jsonschema:
            result.schema_errors = self._jsonschema_errors(result.arguments, tool_schema)
            if result.schema_errors and result.status == STATUS_OK:
                result.status = STATUS_INVALID
        return result

    def _recover_missing_required(
        self,
        result: ValidationResult,
        params: dict[str, dict[str, Any]],
        predictions: Sequence[Any] | None,
    ) -> None:
        by_name = {p.name: p for p in (predictions or [])}
        for name, param in params.items():
            if not param.get("required") or name in result.arguments:
                continue
            if param["routing_type"] in UNSUPPORTED_TYPES:
                result.unsupported_params.append(name)
                continue
            pred = by_name.get(name)
            # Thử lại với ngưỡng thấp hơn cho riêng param này.
            if pred is not None and pred.value is not None and pred.has_value_prob >= self.fallback_threshold:
                coerced, changed = self._coerce(pred.value, param)
                if coerced is not None:
                    result.arguments[name] = coerced
                    if changed:
                        result.coerced_keys.append(name)
                    continue
            result.missing_required.append(name)
        if result.missing_required:
            result.status = STATUS_INCOMPLETE

    def _coerce(self, value: Any, param: dict[str, Any]) -> tuple[Any, bool]:
        """Ép value về đúng type schema. Trả `(value|None, đã_đổi)`."""
        value_type = param.get("value_type") or param.get("type") or "string"
        routing = param["routing_type"]

        if routing == "enum":
            enum_values = param.get("enum") or []
            if value in enum_values:
                return value, False
            lowered = {str(v).lower(): v for v in enum_values}
            match = lowered.get(str(value).lower())
            return (match, True) if match is not None else (None, False)

        if routing == "boolean":
            if isinstance(value, bool):
                return value, False
            text = str(value).strip().lower()
            if text in ("true", "1", "có", "yes"):
                return True, True
            if text in ("false", "0", "không", "no"):
                return False, True
            return None, False

        if value_type in ("integer", "number"):
            if isinstance(value, bool):
                return None, False
            if isinstance(value, (int, float)):
                return (int(value), True) if value_type == "integer" and float(value).is_integer() and not isinstance(value, int) else (value, False)
            normalized = self.normalizer.normalize(str(value), param)
            if not normalized.ok or isinstance(normalized.value, str):
                return None, False
            return normalized.value, True

        if value is None:
            return None, False
        text = str(value).strip()
        return (text, text != value) if text else (None, False)

    def _jsonschema_errors(self, arguments: dict[str, Any], tool_schema: dict[str, Any]) -> list[str]:
        try:
            import jsonschema
        except ImportError:  # pragma: no cover - jsonschema là dependency bắt buộc
            return []

        schema = tool_schema.get("parameters")
        if not isinstance(schema, dict) or "properties" not in schema:
            return []
        # Bỏ required khỏi schema: thiếu required đã được xử lý riêng thành
        # `incomplete`, không cần báo trùng thành `invalid`.
        relaxed = {k: v for k, v in schema.items() if k not in ("required", "additionalProperties")}
        relaxed["properties"] = {
            name: {k: v for k, v in spec.items() if k != "required"}
            for name, spec in schema["properties"].items()
            if isinstance(spec, dict)
        }
        validator = jsonschema.Draft7Validator(relaxed)
        return [error.message for error in validator.iter_errors(arguments)]
