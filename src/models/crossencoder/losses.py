"""Hierarchical loss for Cross-Encoder.

L = BCE(has_value)
  + I[has_value=1, type in {string,number}] * (CE(span_start) + CE(span_end))
  + I[has_value=1, type=enum]            * CE(enum)
  + I[has_value=1, type=boolean]         * CE(boolean)
  + BCE(should_call)                       [chỉ trên hàng cấp tool]

Một batch trộn hai loại hàng: hàng cấp **parameter** (query, param) và hàng cấp
**tool** (query, tool) của ablation §6.1. Chúng phải được tách theo `schema_type`
trước khi tính bất cứ thứ gì — hàng cấp tool không có nhãn `has_value`/span/enum,
để lọt vào các head đó là dạy nhãn rác; và ngược lại hàng parameter không có nhãn
`should_call`. Khi head `should_call` tắt, batch chỉ có một loại hàng và công
thức thu về đúng bản cũ.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

SPAN_TYPES = ("string", "number")
ENUM_TYPE = "enum"
BOOLEAN_TYPE = "boolean"
SHOULD_CALL_TYPE = "should_call"


@dataclass
class LossConfig:
    has_value_weight: float = 1.0
    span_weight: float = 1.0
    enum_weight: float = 1.0
    boolean_weight: float = 1.0
    should_call_weight: float = 1.0
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
        batch_size = int(labels["has_value"].shape[0])
        schema_type_list: list[str] = labels.get("schema_type") or ["string"] * batch_size
        if len(schema_type_list) != batch_size:
            raise ValueError(
                f"schema_type has {len(schema_type_list)} entries but batch is {batch_size}"
            )

        param_rows = [i for i, t in enumerate(schema_type_list) if t != SHOULD_CALL_TYPE]
        tool_rows = [i for i, t in enumerate(schema_type_list) if t == SHOULD_CALL_TYPE]

        l_should = torch.zeros((), device=device)
        should_parts: dict[str, torch.Tensor] = {}
        if tool_rows and "should_call" in outputs and "should_call" in labels:
            idx = torch.tensor(tool_rows, dtype=torch.long, device=device)
            l_should = F.binary_cross_entropy_with_logits(
                outputs["should_call"][idx], labels["should_call"][idx].float()
            ) * self.config.should_call_weight
            should_parts["loss_should_call"] = l_should.detach()

        if not param_rows:
            # Batch toàn hàng cấp tool. BCE trên tensor rỗng trả NaN, nên phải
            # thoát trước chứ không để nó lan vào `loss`.
            return {
                "loss": l_should,
                "loss_has_value": torch.zeros((), device=device),
                "loss_sub": torch.zeros((), device=device),
                **should_parts,
            }

        param_idx = torch.tensor(param_rows, dtype=torch.long, device=device)
        l_has = F.binary_cross_entropy_with_logits(
            outputs["has_value"][param_idx], labels["has_value"][param_idx].float()
        ) * self.config.has_value_weight

        present_idx = [i for i in param_rows if int(labels["has_value"][i]) == 1]
        if not present_idx:
            l_sub = torch.zeros((), device=device)
            return {
                "loss": l_has + l_sub + l_should,
                "loss_has_value": l_has.detach(),
                "loss_sub": l_sub.detach(),
                **should_parts,
            }

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

        total = l_has + l_sub + l_should
        return {
            "loss": total,
            "loss_has_value": l_has.detach(),
            "loss_sub": l_sub.detach(),
            **parts,
            **should_parts,
        }
