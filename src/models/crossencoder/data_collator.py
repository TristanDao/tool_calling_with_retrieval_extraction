"""Data collator for Cross-Encoder.

Each sample = 1 (query, parameter) pair. Tokenize theo BERT-QA format:
[CLS] <query> [SEP] Param=... Type=...[. Enum=...] [SEP]

Query luôn là segment đầu tiên nên token index của span label (tính từ 1, ngay
sau [CLS]) không đổi khi batch được pad bên phải.
"""

import re
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


SCHEMA_TYPE_STRING = "string"
SCHEMA_TYPE_NUMBER = "number"
SCHEMA_TYPE_BOOLEAN = "boolean"
SCHEMA_TYPE_ENUM = "enum"
SCHEMA_TYPE_ARRAY = "array"
SCHEMA_TYPE_OBJECT = "object"

_VALID_TYPES = (
    SCHEMA_TYPE_STRING,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_ARRAY,
    SCHEMA_TYPE_OBJECT,
)

#: Type không được model hỗ trợ — §3.2 method2_plan, báo cáo coverage riêng.
UNSUPPORTED_TYPES = (SCHEMA_TYPE_ARRAY, SCHEMA_TYPE_OBJECT)


_TYPE_MAP: dict[str, str] = {
    "str": SCHEMA_TYPE_STRING,
    "string": SCHEMA_TYPE_STRING,
    "int": SCHEMA_TYPE_NUMBER,
    "integer": SCHEMA_TYPE_NUMBER,
    "float": SCHEMA_TYPE_NUMBER,
    "double": SCHEMA_TYPE_NUMBER,
    "number": SCHEMA_TYPE_NUMBER,
    "bool": SCHEMA_TYPE_BOOLEAN,
    "boolean": SCHEMA_TYPE_BOOLEAN,
    "enum": SCHEMA_TYPE_ENUM,
    "list": SCHEMA_TYPE_ARRAY,
    "array": SCHEMA_TYPE_ARRAY,
    "object": SCHEMA_TYPE_OBJECT,
    "dict": SCHEMA_TYPE_OBJECT,
}

#: Type gốc theo JSON Schema — normalizer cần phân biệt integer với number.
_RAW_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "object": "object",
    "dict": "object",
}

_OPTIONAL_SUFFIX_RE = re.compile(r",\s*optional\s*$", re.IGNORECASE)
_LIST_GENERIC_RE = re.compile(r"^list(\[.*\])?$", re.IGNORECASE)


def _clean_type(raw: Any) -> str:
    if isinstance(raw, (list, tuple)):
        # JSON Schema cho phép type là list, vd ["string", "null"].
        non_null = [t for t in raw if str(t).lower() != "null"]
        raw = non_null[0] if non_null else ""
    return _OPTIONAL_SUFFIX_RE.sub("", str(raw or "").strip()).lower()


def normalize_schema_type(raw: Any) -> str:
    """Type dùng để route head. `integer` gộp vào `number`."""
    cleaned = _clean_type(raw)
    if not cleaned:
        return SCHEMA_TYPE_STRING
    if cleaned in _TYPE_MAP:
        return _TYPE_MAP[cleaned]
    if _LIST_GENERIC_RE.match(cleaned):
        return SCHEMA_TYPE_ARRAY
    return SCHEMA_TYPE_STRING


def raw_schema_type(raw: Any) -> str:
    """Type gốc, giữ nguyên `integer` để normalizer coerce đúng."""
    cleaned = _clean_type(raw)
    if not cleaned:
        return "string"
    if cleaned in _RAW_TYPE_MAP:
        return _RAW_TYPE_MAP[cleaned]
    if _LIST_GENERIC_RE.match(cleaned):
        return "array"
    return "string"


def resolve_param_type(param: dict[str, Any]) -> str:
    """Routing type của một parameter.

    JSON Schema biểu diễn enum bằng ``{"type": "string", "enum": [...]}`` chứ
    không phải ``"type": "enum"``, nên phải xét sự có mặt của khoá ``enum`` trước.
    """
    if param.get("enum"):
        return SCHEMA_TYPE_ENUM
    return normalize_schema_type(param.get("type", SCHEMA_TYPE_STRING))


def iter_parameters(tool_schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Duyệt parameter của một tool theo unified master structure.

    Nhận cả ``{"parameters": {"type": "object", "properties": {...},
    "required": [...]}}`` lẫn dạng phẳng ``{"parameters": {"name": {...}}}``.
    """
    params = tool_schema.get("parameters") or {}
    properties = params.get("properties")
    if properties is None:
        properties = {k: v for k, v in params.items() if isinstance(v, dict)}
        required: list[str] = []
    else:
        required = params.get("required") or []
    required_set = set(required)
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        param = dict(spec)
        param["name"] = name
        param["required"] = name in required_set
        param["routing_type"] = resolve_param_type(param)
        param["value_type"] = raw_schema_type(param.get("type", "string"))
        yield param


def build_schema_question(param: dict[str, Any]) -> str:
    name = param["name"]
    desc = str(param.get("description", "") or "").strip()
    routing_type = param.get("routing_type") or resolve_param_type(param)
    if routing_type == SCHEMA_TYPE_ENUM:
        enum_values = param.get("enum", []) or []
        enum_str = "|".join(str(v) for v in enum_values)
        return f"Param={name}. Desc={desc}. Type=enum. Enum={enum_str}"
    declared = param.get("value_type") or raw_schema_type(param.get("type", "string"))
    return f"Param={name}. Desc={desc}. Type={declared}"


@dataclass
class CollatorConfig:
    tokenizer_name: str = "xlm-roberta-base"
    max_length: int = 256
    padding: str = "longest"
    truncation: str = "only_first"


class CrossEncoderCollator:
    def __init__(
        self,
        config: CollatorConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = tokenizer or AutoTokenizer.from_pretrained(
            config.tokenizer_name, use_fast=True
        )

    def encode_one(
        self,
        query: str,
        param: dict[str, Any],
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tokenize 1 cặp (query, param). Không pad — pad ở `__call__`."""
        question = build_schema_question(param)
        encoded = self.tokenizer(
            query,
            question,
            max_length=self.config.max_length,
            padding=False,
            truncation=self.config.truncation,
        )
        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        token_type_ids = list(encoded.get("token_type_ids") or [0] * len(input_ids))
        query_token_mask = self._query_token_mask(encoded, len(input_ids))

        item: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "query_token_mask": query_token_mask,
            "schema_type": param.get("routing_type") or resolve_param_type(param),
        }
        if labels is not None:
            item["labels"] = labels
        return item

    def _query_token_mask(self, encoded: Any, length: int) -> list[bool]:
        """True tại token thuộc segment query (không tính special token)."""
        seq_ids = None
        if hasattr(encoded, "sequence_ids"):
            try:
                seq_ids = encoded.sequence_ids(0)
            except (ValueError, TypeError):
                seq_ids = None
        if seq_ids is not None:
            return [sid == 0 for sid in seq_ids]
        # Fallback cho tokenizer chậm: cắt tại SEP đầu tiên.
        ids = list(encoded["input_ids"])
        sep_id = self.tokenizer.sep_token_id
        first_sep = ids.index(sep_id) if sep_id in ids else length
        mask = [False] * length
        for i in range(1, min(first_sep, length)):
            mask[i] = True
        return mask

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(b["input_ids"]) for b in batch)
        if self.config.padding == "max_length":
            max_len = self.config.max_length

        def _pad(seq: list[Any], fill: Any) -> list[Any]:
            return list(seq) + [fill] * (max_len - len(seq))

        out: dict[str, Any] = {
            "input_ids": torch.tensor(
                [_pad(b["input_ids"], pad_id) for b in batch], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [_pad(b["attention_mask"], 0) for b in batch], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [_pad(b["token_type_ids"], 0) for b in batch], dtype=torch.long
            ),
            "query_token_mask": torch.tensor(
                [_pad(b["query_token_mask"], False) for b in batch], dtype=torch.bool
            ),
            "schema_type": [b["schema_type"] for b in batch],
        }
        if "labels" in batch[0]:
            out["labels"] = self._collate_labels([b["labels"] for b in batch], max_len)
        return out

    def _collate_labels(
        self, labels_list: list[dict[str, Any]], max_len: int
    ) -> dict[str, Any]:
        def _clip(idx: Any) -> int:
            return min(max(int(idx), 0), max_len - 1)

        return {
            "has_value": torch.tensor(
                [lbl["has_value"] for lbl in labels_list], dtype=torch.long
            ),
            "span_start": torch.tensor(
                [_clip(lbl["span_start"]) for lbl in labels_list], dtype=torch.long
            ),
            "span_end": torch.tensor(
                [_clip(lbl["span_end"]) for lbl in labels_list], dtype=torch.long
            ),
            "enum_label": torch.tensor(
                [lbl["enum_label"] for lbl in labels_list], dtype=torch.long
            ),
            "boolean_label": torch.tensor(
                [lbl["boolean_label"] for lbl in labels_list], dtype=torch.long
            ),
            "schema_type": [lbl["schema_type"] for lbl in labels_list],
        }
