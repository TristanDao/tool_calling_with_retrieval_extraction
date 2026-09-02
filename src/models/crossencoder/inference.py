"""Inference pipeline for Cross-Encoder.

Convert (query, tool_schema) → dict of arguments.
"""

from typing import Any

import torch
from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_STRING,
    build_schema_question,
    normalize_schema_type,
)
from src.models.crossencoder.model import CrossEncoderForExtraction
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@torch.no_grad()
def extract_arguments(
    model: CrossEncoderForExtraction,
    tokenizer: PreTrainedTokenizerBase,
    query: str,
    tool_schema: dict[str, Any],
    device: str | torch.device = "cpu",
    has_value_threshold: float = 0.5,
    max_length: int = 1024,
) -> dict[str, Any]:
    model.eval()
    args: dict[str, Any] = {}
    parameters = tool_schema.get("parameters", {})

    for param_name, param_def in parameters.items():
        param = {"name": param_name, **param_def}
        question = build_schema_question(param)
        encoded = tokenizer(
            query,
            question,
            max_length=max_length,
            padding="max_length",
            truncation="only_first",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        token_type_ids = encoded.get("token_type_ids", attention_mask).to(device)

        sep_positions = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if sep_positions.numel() >= 1:
            first_sep = sep_positions[0].item()
        else:
            first_sep = max_length - 1
        query_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        query_token_mask[0, 1:first_sep] = (
            input_ids[0, 1:first_sep] != tokenizer.pad_token_id
        )

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            query_token_mask=query_token_mask,
        )

        has_value_prob = torch.sigmoid(outputs["has_value"]).item()
        if has_value_prob < has_value_threshold:
            continue

        schema_type = normalize_schema_type(param)
        if schema_type in (SCHEMA_TYPE_STRING, SCHEMA_TYPE_NUMBER):
            start = int(outputs["span_start"][0].argmax().item())
            end = int(outputs["span_end"][0].argmax().item())
            if start > end:
                start, end = end, start
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            span_tokens = tokens[start : end + 1]
            value = tokenizer.convert_tokens_to_string(span_tokens).strip()
            args[param_name] = value
        elif schema_type == SCHEMA_TYPE_ENUM:
            enum_values = param.get("enum", [])
            if not enum_values:
                continue
            idx = int(outputs["enum_logits"][0].argmax().item())
            if 0 <= idx < len(enum_values):
                args[param_name] = enum_values[idx]
        elif schema_type == SCHEMA_TYPE_BOOLEAN:
            idx = int(outputs["boolean_logits"][0].argmax().item())
            args[param_name] = bool(idx == 0)

    return args


def load_cross_encoder(
    model_path: str,
    device: str | torch.device = "cpu",
) -> tuple[CrossEncoderForExtraction, PreTrainedTokenizerBase]:
    from src.models.crossencoder.model import CrossEncoderForExtraction

    model = CrossEncoderForExtraction.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model.to(device)
    return model, tokenizer
