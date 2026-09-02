"""Data collator for Cross-Encoder.

Each sample = 1 (query, parameter) pair. Tokenize theo BERT-QA format:
[CLS] <query> [SEP] Param=... Type=...[. Enum=...] [SEP]
"""

import re
from dataclasses import dataclass
from typing import Any

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


def build_schema_question(param: dict[str, Any]) -> str:
    name = param["name"]
    desc = param.get("description", "").strip()
    ptype = normalize_schema_type(param)
    if ptype == SCHEMA_TYPE_ENUM:
        enum_values = param.get("enum", [])
        enum_str = "|".join(str(v) for v in enum_values)
        return f"Param={name}. Desc={desc}. Type=enum. Enum={enum_str}"
    return f"Param={name}. Desc={desc}. Type={ptype}"


_TYPE_MAP: dict[str, str] = {
    "str": SCHEMA_TYPE_STRING,
    "string": SCHEMA_TYPE_STRING,
    "int": SCHEMA_TYPE_NUMBER,
    "integer": SCHEMA_TYPE_NUMBER,
    "float": SCHEMA_TYPE_NUMBER,
    "number": SCHEMA_TYPE_NUMBER,
    "bool": SCHEMA_TYPE_BOOLEAN,
    "boolean": SCHEMA_TYPE_BOOLEAN,
    "enum": SCHEMA_TYPE_ENUM,
    "list": SCHEMA_TYPE_ARRAY,
    "array": SCHEMA_TYPE_ARRAY,
    "object": SCHEMA_TYPE_OBJECT,
    "dict": SCHEMA_TYPE_OBJECT,
}

_OPTIONAL_SUFFIX_RE = re.compile(r",\s*optional\s*$", re.IGNORECASE)
_LIST_GENERIC_RE = re.compile(r"^list(\[.*\])?$", re.IGNORECASE)


def normalize_schema_type(raw: str | dict[str, Any]) -> str:
    if isinstance(raw, dict):
        if isinstance(raw.get("enum"), list) and raw["enum"]:
            return SCHEMA_TYPE_ENUM
        raw = raw.get("type", "string")
    if not raw:
        return SCHEMA_TYPE_STRING
    cleaned = _OPTIONAL_SUFFIX_RE.sub("", str(raw).strip()).lower()
    if cleaned in _TYPE_MAP:
        return _TYPE_MAP[cleaned]
    if _LIST_GENERIC_RE.match(cleaned):
        return SCHEMA_TYPE_ARRAY
    return SCHEMA_TYPE_STRING


@dataclass
class CollatorConfig:
    tokenizer_name: str = "BAAI/bge-m3"
    max_length: int = 1024
    padding: str = "max_length"
    truncation: str = "only_first"


class CrossEncoderCollator:
    def __init__(self, config: CollatorConfig) -> None:
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.tokenizer_name, use_fast=True
        )

    def encode_one(
        self,
        query: str,
        param: dict[str, Any],
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        question = build_schema_question(param)
        encoded = self.tokenizer(
            query,
            question,
            max_length=self.config.max_length,
            padding=self.config.padding,
            truncation=self.config.truncation,
            return_tensors="pt",
        )
        sep_positions = (encoded["input_ids"][0] == self.tokenizer.sep_token_id).nonzero(
            as_tuple=True
        )[0]
        if sep_positions.numel() >= 1:
            first_sep = sep_positions[0].item()
            query_len = first_sep
        else:
            query_len = self.config.max_length - 1
        token_ids = encoded["input_ids"][0].tolist()
        cls_idx = 0
        query_token_mask = [False] * len(token_ids)
        for i in range(cls_idx + 1, min(query_len, len(token_ids))):
            if token_ids[i] != self.tokenizer.pad_token_id:
                query_token_mask[i] = True

        item: dict[str, Any] = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "token_type_ids": encoded.get("token_type_ids", encoded["attention_mask"])[0],
            "query_token_mask": torch.tensor(query_token_mask, dtype=torch.bool),
            "schema_type": normalize_schema_type(param.get("type", "string")),
        }
        if labels is not None:
            item["labels"] = labels
        return item

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        keys = [k for k in batch[0].keys() if k != "labels"]
        out: dict[str, Any] = {}
        for k in keys:
            out[k] = torch.stack([b[k] for b in batch])
        if "labels" in batch[0]:
            out["labels"] = self._collate_labels([b["labels"] for b in batch])
        return out

    def _collate_labels(self, labels_list: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["has_value"] = torch.tensor(
            [lbl["has_value"] for lbl in labels_list], dtype=torch.long
        )
        out["span_start"] = torch.tensor(
            [lbl["span_start"] for lbl in labels_list], dtype=torch.long
        )
        out["span_end"] = torch.tensor(
            [lbl["span_end"] for lbl in labels_list], dtype=torch.long
        )
        out["enum_label"] = torch.tensor(
            [lbl["enum_label"] for lbl in labels_list], dtype=torch.long
        )
        out["boolean_label"] = torch.tensor(
            [lbl["boolean_label"] for lbl in labels_list], dtype=torch.long
        )
        out["schema_type"] = [lbl["schema_type"] for lbl in labels_list]
        return out
