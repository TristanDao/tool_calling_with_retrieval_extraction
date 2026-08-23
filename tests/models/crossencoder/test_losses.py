"""Regression test cho bug §2.3a — index sai trong HierarchicalLoss.

Bản lỗi lấy `schema_type[0..len(present_idx)-1]` thay vì `schema_type[present_idx]`,
nên khi số sample có has_value=1 nhỏ hơn batch size, type bị gán lệch: span loss
áp lên sample enum, enum loss áp lên sample boolean. Loss vẫn giảm nên bug không
có triệu chứng — chỉ gradient mới lộ ra.
"""

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.losses import HierarchicalLoss, LossConfig


BATCH = 4
SEQ_LEN = 8
MAX_ENUM = 5


def _make_outputs() -> dict:
    torch.manual_seed(0)
    return {
        "has_value": torch.randn(BATCH, requires_grad=True),
        "span_start": torch.randn(BATCH, SEQ_LEN, requires_grad=True),
        "span_end": torch.randn(BATCH, SEQ_LEN, requires_grad=True),
        "enum_logits": torch.randn(BATCH, MAX_ENUM, requires_grad=True),
        "boolean_logits": torch.randn(BATCH, 2, requires_grad=True),
    }


def _make_labels() -> dict:
    # Batch cố ý: [string(hv=1), enum(hv=0), boolean(hv=1), string(hv=0)]
    return {
        "has_value": torch.tensor([1, 0, 1, 0]),
        "span_start": torch.tensor([2, 0, 0, 0]),
        "span_end": torch.tensor([3, 0, 0, 0]),
        "enum_label": torch.tensor([0, 1, 0, 0]),
        "boolean_label": torch.tensor([0, 0, 1, 0]),
        "schema_type": ["string", "enum", "boolean", "string"],
    }


def _grad_rows_nonzero(grad: "torch.Tensor | None") -> list[int]:
    if grad is None:  # head không tham gia loss nào → autograd không tạo grad
        return []
    flat = grad.reshape(grad.size(0), -1)
    return [i for i in range(flat.size(0)) if flat[i].abs().sum().item() > 0]


def _grad_is_empty(grad: "torch.Tensor | None") -> bool:
    return grad is None or grad.abs().sum().item() == 0.0


def test_loss_routes_each_head_to_the_right_batch_rows() -> None:
    outputs = _make_outputs()
    labels = _make_labels()

    loss_dict = HierarchicalLoss(LossConfig())(outputs, labels)
    loss_dict["loss"].backward()

    # Chỉ row 0 (string, has_value=1) được áp span loss.
    assert _grad_rows_nonzero(outputs["span_start"].grad) == [0]
    assert _grad_rows_nonzero(outputs["span_end"].grad) == [0]
    # Chỉ row 2 (boolean, has_value=1) được áp boolean loss.
    assert _grad_rows_nonzero(outputs["boolean_logits"].grad) == [2]
    # Sample enum duy nhất có has_value=0 → enum loss không được kích hoạt.
    assert _grad_rows_nonzero(outputs["enum_logits"].grad) == []


def test_loss_reports_component_terms() -> None:
    outputs = _make_outputs()
    loss_dict = HierarchicalLoss(LossConfig())(outputs, _make_labels())

    assert "loss_span" in loss_dict
    assert "loss_boolean" in loss_dict
    assert "loss_enum" not in loss_dict
    assert loss_dict["loss"].item() == pytest.approx(
        loss_dict["loss_has_value"].item() + loss_dict["loss_sub"].item(), rel=1e-5
    )


def test_loss_with_no_present_sample_only_has_value_term() -> None:
    outputs = _make_outputs()
    labels = _make_labels()
    labels["has_value"] = torch.zeros(BATCH, dtype=torch.long)

    loss_dict = HierarchicalLoss(LossConfig())(outputs, labels)
    loss_dict["loss"].backward()

    assert loss_dict["loss_sub"].item() == 0.0
    assert _grad_is_empty(outputs["span_start"].grad)
    assert _grad_is_empty(outputs["enum_logits"].grad)


def test_loss_rejects_schema_type_length_mismatch() -> None:
    outputs = _make_outputs()
    labels = _make_labels()
    labels["schema_type"] = ["string", "enum"]

    with pytest.raises(ValueError, match="schema_type"):
        HierarchicalLoss(LossConfig())(outputs, labels)


def test_span_weight_zero_disables_span_gradient() -> None:
    outputs = _make_outputs()
    config = LossConfig(span_weight=0.0)

    HierarchicalLoss(config)(outputs, _make_labels())["loss"].backward()

    assert _grad_is_empty(outputs["span_start"].grad)
    assert _grad_rows_nonzero(outputs["boolean_logits"].grad) == [2]
