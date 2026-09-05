"""Ablation §6.1 — head `should_call` thay ngưỡng cosine τ.

Ba thứ phải đúng, và cả ba đều là loại hỏng-âm-thầm:

1. Tắt thì phải **không để lại dấu vết**: state dict trùng khít checkpoint cũ,
   output không mọc thêm khoá. Nếu không, `run01` không nạp lại được và baseline
   biến mất.
2. Hai loại hàng trong cùng batch **không được huấn luyện chéo head của nhau**.
   Hàng cấp tool không có nhãn span/enum; để nó lọt vào các head đó là dạy nhãn
   rác mà không có gì báo.
3. Negative chỉ đến từ sample no-call — xem docstring `ShouldCallConfig`.
"""

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.data_collator import (  # noqa: E402
    SCHEMA_TYPE_SHOULD_CALL,
    build_schema_question,
)
from src.models.crossencoder.dataset import (  # noqa: E402
    CrossEncoderPairsConfig,
    ShouldCallConfig,
    balance_should_call,
    build_pairs,
)
from src.models.crossencoder.heads import CrossEncoderHeads, HeadConfig  # noqa: E402
from src.models.crossencoder.losses import HierarchicalLoss, LossConfig  # noqa: E402
from src.models.sources import SourceSpec  # noqa: E402

HIDDEN = 8


# ------------------------------------------------------------------- head


def test_tat_thi_state_dict_khong_doi():
    off = CrossEncoderHeads(HeadConfig(hidden_size=HIDDEN))
    on = CrossEncoderHeads(HeadConfig(hidden_size=HIDDEN, enable_should_call=True))

    assert "should_call.weight" not in off.state_dict()
    assert set(on.state_dict()) - set(off.state_dict()) == {
        "should_call.weight",
        "should_call.bias",
    }


def test_checkpoint_cu_khong_co_khoa_van_dung_lai_duoc():
    """`from_pretrained` dựng HeadConfig(**dict đã lưu) — dict cũ thiếu khoá."""
    old_checkpoint_config = {"hidden_size": HIDDEN, "max_enum_size": 20, "dropout": 0.1}

    config = HeadConfig(**old_checkpoint_config)

    assert config.enable_should_call is False
    assert CrossEncoderHeads(config).should_call is None


def test_output_chi_moc_them_khoa_khi_bat():
    hidden = torch.zeros(2, 5, HIDDEN)
    mask = torch.ones(2, 5, dtype=torch.bool)

    off = CrossEncoderHeads(HeadConfig(hidden_size=HIDDEN))(hidden, mask)
    on = CrossEncoderHeads(HeadConfig(hidden_size=HIDDEN, enable_should_call=True))(hidden, mask)

    assert "should_call" not in off
    assert on["should_call"].shape == (2,)


# ------------------------------------------------------------------- loss


def _outputs(batch: int, with_should_call: bool = True) -> dict:
    out = {
        "has_value": torch.zeros(batch, requires_grad=True),
        "span_start": torch.zeros(batch, 6, requires_grad=True),
        "span_end": torch.zeros(batch, 6, requires_grad=True),
        "enum_logits": torch.zeros(batch, 4, requires_grad=True),
        "boolean_logits": torch.zeros(batch, 2, requires_grad=True),
    }
    if with_should_call:
        out["should_call"] = torch.zeros(batch, requires_grad=True)
    return out


def _labels(schema_types: list[str], has_value: list[int], should_call: list[int]) -> dict:
    n = len(schema_types)
    return {
        "has_value": torch.tensor(has_value, dtype=torch.long),
        "span_start": torch.ones(n, dtype=torch.long),
        "span_end": torch.tensor([2] * n, dtype=torch.long),
        "enum_label": torch.zeros(n, dtype=torch.long),
        "boolean_label": torch.zeros(n, dtype=torch.long),
        "should_call": torch.tensor(should_call, dtype=torch.long),
        "schema_type": schema_types,
    }


def test_hang_cap_tool_khong_lot_vao_head_has_value():
    loss = HierarchicalLoss(LossConfig())
    only_params = loss(
        _outputs(2), _labels(["string", "string"], [1, 0], [0, 0])
    )
    mixed = loss(
        _outputs(3),
        _labels(["string", "string", SCHEMA_TYPE_SHOULD_CALL], [1, 0, 0], [0, 0, 1]),
    )

    # Thêm một hàng cấp tool KHÔNG được làm đổi loss của head has_value.
    assert mixed["loss_has_value"] == pytest.approx(float(only_params["loss_has_value"]))
    assert mixed["loss_should_call"] > 0


def test_hang_cap_parameter_khong_lot_vao_head_should_call():
    loss = HierarchicalLoss(LossConfig())

    result = loss(
        _outputs(2), _labels(["string", "string"], [1, 1], [1, 1])
    )

    # Không có hàng cấp tool nào → head should_call không được tính, dù nhãn có.
    assert "loss_should_call" not in result


def test_batch_toan_hang_cap_tool_khong_ra_nan():
    """BCE trên tensor rỗng trả NaN — batch cuối epoch có thể rơi vào ca này."""
    loss = HierarchicalLoss(LossConfig())

    result = loss(
        _outputs(2), _labels([SCHEMA_TYPE_SHOULD_CALL] * 2, [0, 0], [1, 0])
    )

    assert torch.isfinite(result["loss"])
    assert float(result["loss_has_value"]) == 0.0


def test_should_call_weight_duoc_ap_dung():
    labels = _labels([SCHEMA_TYPE_SHOULD_CALL], [0], [1])
    base = HierarchicalLoss(LossConfig())(_outputs(1), labels)
    doubled = HierarchicalLoss(LossConfig(should_call_weight=2.0))(_outputs(1), labels)

    assert float(doubled["loss"]) == pytest.approx(2 * float(base["loss"]))


# -------------------------------------------------------------- question


def test_question_cap_tool_dung_khuon_khoa_gia_tri():
    param = {
        "name": "vi_search_restaurants",
        "description": "Tìm quán ăn",
        "param_names": ["location", "cuisine"],
        "routing_type": SCHEMA_TYPE_SHOULD_CALL,
    }

    assert build_schema_question(param) == (
        "Tool=vi_search_restaurants. Desc=Tìm quán ăn. Params=location|cuisine"
    )


# ---------------------------------------------------------------- dataset


def _write_source(tmp_path, rows) -> SourceSpec:
    import json

    path = tmp_path / "src.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return SourceSpec("fake_train", path, "train")


TOOL_A = {
    "name": "tool_a",
    "description": "Công cụ A",
    "parameters": {
        "type": "object",
        "properties": {"x": {"type": "string", "description": "X"}},
        "required": ["x"],
    },
}
TOOL_B = {**TOOL_A, "name": "tool_b", "description": "Công cụ B"}


def _pairs_config(tmp_path, **kwargs) -> CrossEncoderPairsConfig:
    return CrossEncoderPairsConfig(
        output_dir=tmp_path,
        stats_path=tmp_path / "stats.json",
        decontamination_path=tmp_path / "deco.json",
        allow_build_decontamination=True,
        should_call=ShouldCallConfig(enabled=True, **kwargs),
    )


def test_negative_chi_lay_tu_sample_no_call(tmp_path):
    rows = [
        {"id": "p1", "source": "s", "query": "Tìm quán ăn ở Huế",
         "function_calls": [{"name": "tool_a", "arguments": {"x": "Huế"}}],
         "tools": [TOOL_A, TOOL_B]},
        {"id": "n1", "source": "s", "query": "Xin chào", "function_calls": [],
         "tools": [TOOL_A, TOOL_B]},
    ]
    spec = _write_source(tmp_path, rows)

    rows_by_split, stats = build_pairs(_pairs_config(tmp_path), specs=(spec,))

    tool_rows = [
        r for r in rows_by_split["train"]
        if r["labels"]["schema_type"] == SCHEMA_TYPE_SHOULD_CALL
    ]
    by_sample = {r["sample_id"]: r["labels"]["should_call"] for r in tool_rows}
    assert by_sample == {"p1": 1, "n1": 0}
    # `tool_b` là candidate SAI của p1 nhưng không được sinh thành negative.
    assert not [r for r in tool_rows if r["sample_id"] == "p1" and r["tool_name"] == "tool_b"]
    assert stats["should_call"]["train"]["positive_rate"] == 0.5


def test_tat_thi_khong_sinh_hang_nao(tmp_path):
    rows = [
        {"id": "n1", "source": "s", "query": "Xin chào", "function_calls": [],
         "tools": [TOOL_A]},
        {"id": "p1", "source": "s", "query": "Tìm quán ăn ở Huế",
         "function_calls": [{"name": "tool_a", "arguments": {"x": "Huế"}}],
         "tools": [TOOL_A]},
    ]
    spec = _write_source(tmp_path, rows)
    config = _pairs_config(tmp_path)
    config.should_call = ShouldCallConfig(enabled=False)

    rows_by_split, stats = build_pairs(config, specs=(spec,))

    assert "should_call" not in stats
    assert not [
        r for r in rows_by_split["train"]
        if r["labels"]["schema_type"] == SCHEMA_TYPE_SHOULD_CALL
    ]


def test_hang_cap_tool_dung_khuon_hang_cap_parameter(tmp_path):
    """`CrossEncoderDataset` không biết gì về loại hàng này — khuôn phải khớp."""
    rows = [{"id": "n1", "source": "s", "query": "Xin chào", "function_calls": [],
             "tools": [TOOL_A]}]
    spec = _write_source(tmp_path, rows)

    rows_by_split, _ = build_pairs(_pairs_config(tmp_path), specs=(spec,))
    row = rows_by_split["train"][0]

    assert set(row) >= {"sample_id", "query", "tool_name", "param", "labels", "split"}
    assert row["param"]["routing_type"] == SCHEMA_TYPE_SHOULD_CALL
    assert row["param"]["param_names"] == ["x"]
    # has_value=0 → `_prepare` bỏ qua bước căn char span (hàng này không có span).
    assert row["labels"]["has_value"] == 0


def test_can_bang_giu_tron_lop_hiem():
    positives = {"train": [{"i": i} for i in range(100)]}
    negatives = {"train": [{"i": i} for i in range(10)]}

    balanced, report = balance_should_call(
        positives, negatives, ShouldCallConfig(positive_rate=0.5)
    )

    assert report["train"] == {
        "positive": 10, "negative": 10,
        "positive_available": 100, "negative_available": 10,
        "positive_rate": 0.5,
    }
    assert len(balanced["train"]) == 20


def test_can_bang_xac_dinh_theo_seed():
    positives = {"train": [{"i": i} for i in range(50)]}
    negatives = {"train": [{"i": i} for i in range(10)]}

    first, _ = balance_should_call(positives, negatives, ShouldCallConfig())
    second, _ = balance_should_call(positives, negatives, ShouldCallConfig())

    assert [r["i"] for r in first["train"]] == [r["i"] for r in second["train"]]
