"""Hierarchical heads for Cross-Encoder parameter extraction.

Output structure:
- has_value: (B,) — binary logit, luôn được dự đoán.
- span_start: (B, L) — logit cho từng token; masked ngoài query tokens.
- span_end: (B, L) — tương tự span_start.
- enum_logits: (B, max_enum_size) — N-way classification; dùng khi schema type=enum.
- boolean_logits: (B, 2) — true/false; dùng khi schema type=boolean.

Type routing dựa trên schema question, không phải output head.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadConfig:
    hidden_size: int = 1024
    max_enum_size: int = 20
    dropout: float = 0.1


class CrossEncoderHeads(nn.Module):
    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.has_value = nn.Linear(config.hidden_size, 1)
        self.span_start = nn.Linear(config.hidden_size, 1)
        self.span_end = nn.Linear(config.hidden_size, 1)
        self.enum_head = nn.Linear(config.hidden_size, config.max_enum_size)
        self.boolean_head = nn.Linear(config.hidden_size, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.dropout(hidden_states)
        cls_hidden = hidden_states[:, 0]
        has_value_logit = self.has_value(cls_hidden).squeeze(-1)
        boolean_logits = self.boolean_head(cls_hidden)
        enum_logits = self.enum_head(cls_hidden)
        span_start = self.span_start(hidden_states).squeeze(-1)
        span_end = self.span_end(hidden_states).squeeze(-1)
        if query_token_mask is not None:
            span_start = span_start.masked_fill(~query_token_mask, -1e4)
            span_end = span_end.masked_fill(~query_token_mask, -1e4)
        return {
            "has_value": has_value_logit,
            "span_start": span_start,
            "span_end": span_end,
            "enum_logits": enum_logits,
            "boolean_logits": boolean_logits,
        }
