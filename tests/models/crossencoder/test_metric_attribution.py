"""Hai lỗi ĐO ĐẠC trong `evaluate.py`, phát hiện khi đọc kết quả Phase 3.

C. `by_source` truyền **cả batch** cho mọi source có mặt trong batch. File val
   xen kẽ xlam/glaive từng dòng nên gần như batch nào cũng bị đếm chéo: xlam
   không có enum nào vẫn báo 24, và số của glaive trùng khít số của xlam.

D. Metric argmax enum trên cả `max_enum_size=20` chiều, trong khi
   `inference.py` cắt về `len(param["enum"])` trước khi argmax. Enum của
   custom_vi chỉ có 2-5 giá trị → 85% không gian output là vô nghĩa, và metric
   phạt model cho lỗi mà pipeline thật không thể mắc.
"""

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.data_collator import (  # noqa: E402
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_STRING,
)
from src.models.crossencoder.evaluate import ComponentMetrics, _take  # noqa: E402


def _outputs(enum_logits: list[list[float]]) -> dict:
    n = len(enum_logits)
    return {
        "has_value": torch.full((n,), 10.0),  # sigmoid ~ 1
        "span_start": torch.zeros(n, 8),
        "span_end": torch.zeros(n, 8),
        "enum_logits": torch.tensor(enum_logits),
        "boolean_logits": torch.zeros(n, 2),
    }


def _labels(schema_types: list[str], enum_labels: list[int], enum_sizes: list[int]) -> dict:
    n = len(schema_types)
    return {
        "has_value": torch.ones(n, dtype=torch.long),
        "span_start": torch.zeros(n, dtype=torch.long),
        "span_end": torch.zeros(n, dtype=torch.long),
        "enum_label": torch.tensor(enum_labels, dtype=torch.long),
        "boolean_label": torch.zeros(n, dtype=torch.long),
        "schema_type": schema_types,
        "enum_size": enum_sizes,
    }


def test_enum_argmax_is_masked_to_the_schema_range():
    """Logit lớn nhất nằm NGOÀI enum list thì phải bị bỏ qua, không tính là sai.

    Ở đây enum chỉ có 2 giá trị; model đặt logit cao nhất ở vị trí 5 (không tồn
    tại) nhưng trong hai vị trí hợp lệ thì vị trí 1 cao hơn — và nhãn là 1.
    """
    logits = [[0.1, 0.9, 0.0, 0.0, 0.0, 9.0]]
    metrics = ComponentMetrics()

    metrics.update(_outputs(logits), _labels([SCHEMA_TYPE_ENUM], [1], [2]))

    assert metrics.compute()["enum_accuracy"] == 1.0


def test_unmasked_argmax_would_have_scored_it_wrong():
    """Chứng minh mask thật sự đổi kết quả: bỏ `enum_size` là sai ngay."""
    logits = [[0.1, 0.9, 0.0, 0.0, 0.0, 9.0]]
    metrics = ComponentMetrics()

    labels = _labels([SCHEMA_TYPE_ENUM], [1], [2])
    labels.pop("enum_size")
    metrics.update(_outputs(logits), labels)

    assert metrics.compute()["enum_accuracy"] == 0.0


def test_enum_size_zero_leaves_logits_alone():
    """Không biết kích thước enum thì không mask, chứ không mask thành rỗng."""
    metrics = ComponentMetrics()

    metrics.update(_outputs([[0.0, 5.0, 0.0]]), _labels([SCHEMA_TYPE_ENUM], [1], [0]))

    assert metrics.compute()["enum_accuracy"] == 1.0


def test_take_selects_rows_from_tensors_and_lists():
    data = {
        "t": torch.tensor([[0.0], [1.0], [2.0]]),
        "l": ["a", "b", "c"],
        "scalar": 7,
    }

    picked = _take(data, [0, 2])

    assert picked["t"].tolist() == [[0.0], [2.0]]
    assert picked["l"] == ["a", "c"]
    assert picked["scalar"] == 7


def test_per_source_metrics_only_count_their_own_rows():
    """Batch trộn 2 source: mỗi source chỉ được đếm phần của mình.

    Trước khi vá, cả hai source đều nhận cả batch nên `enum_total` của mỗi bên
    bằng tổng của cả batch — đúng triệu chứng xlam-báo-24-enum-dù-có-0.
    """
    sources = ["xlam", "glaive"]
    outputs = _outputs([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    labels = _labels([SCHEMA_TYPE_STRING, SCHEMA_TYPE_ENUM], [0, 1], [0, 2])

    per_source = {}
    for source in set(sources):
        rows = [i for i, name in enumerate(sources) if name == source]
        m = ComponentMetrics()
        m.update(_take(outputs, rows), _take(labels, rows))
        per_source[source] = m.compute()

    assert per_source["xlam"]["n_enum"] == 0, "xlam không có dòng enum nào"
    assert per_source["glaive"]["n_enum"] == 1
    assert per_source["xlam"]["n_span"] == 1
    assert per_source["glaive"]["n_span"] == 0
