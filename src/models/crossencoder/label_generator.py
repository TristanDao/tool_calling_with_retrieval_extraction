"""Rule-based label generator for Cross-Encoder.

For each (query, parameter, gold_value) produces training labels:
- string/number: substring match → (start, end) token positions, has_value=1
- enum: match gold value with enum list → enum class id, has_value=1
- boolean: keyword-based cue (có/không/muốn/...) → true/false, has_value=1
- null: param optional + no gold value → has_value=0
- non-verbatim value → mark as not_trainable (skip in training)
"""

import re
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_STRING,
    normalize_schema_type,
)


SKIP_LABEL = "skip"


BOOLEAN_TRUE_CUES = [
    r"\bcó\s+", r"\bmuốn\s+", r"\bcần\s+", r"\bnên\s+",
    r"\bđược\s+", r"\bđồng ý\b", r"\bchấp nhận\b", r"\bcầu\b",
    r"\byes\b", r"\btrue\b", r"\bđúng\b",
]

BOOLEAN_FALSE_CUES = [
    r"\bkhông\s+(có\s+)?", r"\bchưa\s+", r"\bđừng\b", r"\bcấm\b",
    r"\bkhông\s+được\b", r"\bkhông\s+muốn\b", r"\bkhông\s+cần\b",
    r"\bno\b", r"\bfalse\b", r"\bsai\b", r"\btừ chối\b",
]


@dataclass
class LabelGeneratorConfig:
    tokenizer_name: str = "BAAI/bge-m3"
    max_length: int = 1024


class LabelGenerator:
    def __init__(self, config: LabelGeneratorConfig) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.tokenizer_name, use_fast=True
        )
        self.true_patterns = [re.compile(p, re.IGNORECASE) for p in BOOLEAN_TRUE_CUES]
        self.false_patterns = [re.compile(p, re.IGNORECASE) for p in BOOLEAN_FALSE_CUES]

    def _align_span(
        self,
        query: str,
        value: str,
        max_length: int,
    ) -> tuple[int, int] | None:
        idx = query.find(value)
        if idx < 0:
            return None
        char_start = idx
        char_end = idx + len(value) - 1
        offsets = self.tokenizer(
            query,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length - 2,
        )["offset_mapping"]
        start_tok: int | None = None
        end_tok: int | None = None
        for tok_idx, (s, e) in enumerate(offsets):
            if start_tok is None and s <= char_start < e:
                start_tok = tok_idx + 1
            if s <= char_end < e:
                end_tok = tok_idx + 1
        if start_tok is None or end_tok is None:
            return None
        if end_tok >= max_length - 1:
            return None
        return start_tok, end_tok

    def _detect_boolean(self, query: str, param_name: str) -> int | None:
        kw = param_name.lower()
        lower_q = query.lower()
        for pat in self.false_patterns:
            if pat.search(lower_q):
                for cue in ["cấm", "không", "chưa", "đừng", "từ chối"]:
                    if cue in kw or cue in lower_q:
                        return 1
                return 1
        for pat in self.true_patterns:
            if pat.search(lower_q):
                return 0
        return None

    def generate(
        self,
        query: str,
        param: dict[str, Any],
        gold_value: Any,
        is_required: bool = True,
    ) -> dict[str, Any] | str:
        schema_type = normalize_schema_type(param.get("type", "string"))
        result: dict[str, Any] = {
            "has_value": 0,
            "span_start": 0,
            "span_end": 0,
            "enum_label": 0,
            "boolean_label": 0,
            "schema_type": schema_type,
        }

        if gold_value is None:
            if is_required:
                result["has_value"] = 1
                result["span_start"] = 0
                result["span_end"] = 0
                result["schema_type"] = schema_type
            return result

        if schema_type in (SCHEMA_TYPE_STRING, SCHEMA_TYPE_NUMBER):
            value_str = str(gold_value)
            span = self._align_span(query, value_str, self.config.max_length)
            if span is None:
                if is_required:
                    result["has_value"] = 1
                    return result
                return SKIP_LABEL
            start_tok, end_tok = span
            result["has_value"] = 1
            result["span_start"] = start_tok
            result["span_end"] = end_tok
            return result

        if schema_type == SCHEMA_TYPE_ENUM:
            enum_values = param.get("enum", [])
            value_str = str(gold_value)
            if value_str not in enum_values:
                if is_required:
                    result["has_value"] = 1
                    return result
                return SKIP_LABEL
            result["has_value"] = 1
            result["enum_label"] = enum_values.index(value_str)
            return result

        if schema_type == SCHEMA_TYPE_BOOLEAN:
            detected = self._detect_boolean(query, param.get("name", ""))
            if detected is None:
                if is_required:
                    result["has_value"] = 1
                    return result
                return SKIP_LABEL
            result["has_value"] = 1
            result["boolean_label"] = detected
            return result

        if is_required:
            result["has_value"] = 1
            return result
        return SKIP_LABEL
