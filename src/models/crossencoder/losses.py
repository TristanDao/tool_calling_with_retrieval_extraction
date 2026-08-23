"""Hierarchical loss for Cross-Encoder.

L = BCE(has_value)
  + I[has_value=1, type in {string,number}] * (CE(span_start) + CE(span_end))
  + I[has_value=1, type=enum]            * CE(enum)
  + I[has_value=1, type=boolean]         * CE(boolean)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SPAN_TYPES = ("string", "number")
ENUM_TYPE = "enum"
BOOLEAN_TYPE = "boolean"


@dataclass
class LossConfig:
    has_value_weight: float = 1.0
    span_weight: float = 1.0
    enum_weight: float = 1.0
    boolean_weight: float = 1.0
    span_loss_combiner: str = "sum"


def _index_for(
    present_idx: list[int],
    schema_type_present: list[str],
    target_types: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    rows = [i for i, t in zip(present_idx, schema_type_present) if t in target_types]
    if not rows:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.tensor(rows, dtype=torch.long, device=device)


class HierarchicalLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        device = outputs["has_value"].device
        has_value_label = labels["has_value"].float()
        l_has = F.binary_cross_entropy_with_logits(
            outputs["has_value"], has_value_label
        ) * self.config.has_value_weight

        present_idx = (labels["has_value"] == 1).nonzero(as_tuple=True)[0].tolist()
        if not present_idx:
            l_sub = torch.zeros((), device=device)
            total = l_has + l_sub
            return {
                "loss": total,
                "loss_has_value": l_has.detach(),
                "loss_sub": l_sub.detach(),
            }

        batch_size = int(labels["has_value"].shape[0])
        schema_type_list: list[str] = labels.get("schema_type") or ["string"] * batch_size
        if len(schema_type_list) != batch_size:
            raise ValueError(
                f"schema_type has {len(schema_type_list)} entries but batch is {batch_size}"
            )
        # Phải index theo vị trí thật trong batch, không phải 0..len(present_idx).
        schema_type_present = [schema_type_list[i] for i in present_idx]

        span_idx = _index_for(present_idx, schema_type_present, SPAN_TYPES, device)
        enum_idx = _index_for(present_idx, schema_type_present, (ENUM_TYPE,), device)
        bool_idx = _index_for(present_idx, schema_type_present, (BOOLEAN_TYPE,), device)

        l_sub = torch.zeros((), device=device)
        parts: dict[str, torch.Tensor] = {}

        if span_idx.numel() > 0:
            l_span_start = F.cross_entropy(
                outputs["span_start"][span_idx], labels["span_start"][span_idx]
            )
            l_span_end = F.cross_entropy(
                outputs["span_end"][span_idx], labels["span_end"][span_idx]
            )
            combined = (l_span_start + l_span_end) / 2.0 if self.config.span_loss_combiner == "mean" else (l_span_start + l_span_end)
            l_sub = l_sub + combined * self.config.span_weight
            parts["loss_span"] = combined.detach()

        if enum_idx.numel() > 0:
            l_enum = F.cross_entropy(
                outputs["enum_logits"][enum_idx], labels["enum_label"][enum_idx]
            )
            l_sub = l_sub + l_enum * self.config.enum_weight
            parts["loss_enum"] = l_enum.detach()

        if bool_idx.numel() > 0:
            l_bool = F.cross_entropy(
                outputs["boolean_logits"][bool_idx], labels["boolean_label"][bool_idx]
            )
            l_sub = l_sub + l_bool * self.config.boolean_weight
            parts["loss_boolean"] = l_bool.detach()

        total = l_has + l_sub
        return {
            "loss": total,
            "loss_has_value": l_has.detach(),
            "loss_sub": l_sub.detach(),
            **parts,
        }
