"""Pipeline dùng head `should_call` thay ngưỡng τ (ablation §6.1).

Điểm quan trọng nhất ở đây không phải "chạy được" mà là **không im lặng**:
nếu checkpoint không có head mà config vẫn ghi `abstention: should_call`, pipeline
phải dừng. Quay lặng lẽ về τ nghĩa là báo cáo số của baseline dưới tên ablation —
đúng lớp lỗi đã xảy ra với `strategy` ở §7.10b, phát hiện ra sau khi đã tiêu
xong giờ GPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossencoder"))
from conftest import FakeTokenizer  # noqa: E402

from src.models.biencoder.retrieve import RetrievalThresholds, ToolRetriever  # noqa: E402
from src.models.crossencoder.heads import CrossEncoderHeads, HeadConfig  # noqa: E402
from src.models.crossencoder.inference import (  # noqa: E402
    CrossEncoderExtractor,
    ExtractionConfig,
)
from src.models.pipeline.method2 import (  # noqa: E402
    ABSTENTION_SHOULD_CALL,
    ABSTENTION_TAU,
    MODE_PIPELINE,
    Method2Pipeline,
)
from src.models.pipeline.validator import ArgumentValidator  # noqa: E402
from models.pipeline.test_method2_end_to_end import (  # noqa: E402
    OTHER_TOOL,
    QUERY,
    TOOL,
    FakeCrossEncoder,
)

HIDDEN = 4


class ModelWithShouldCall(FakeCrossEncoder):
    """FakeCrossEncoder + head `should_call` trả logit đặt tay."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit
        self.heads = CrossEncoderHeads(
            HeadConfig(hidden_size=HIDDEN, enable_should_call=True)
        )

    def __call__(self, input_ids, attention_mask, query_token_mask, token_type_ids=None):
        out = super().__call__(input_ids, attention_mask, query_token_mask, token_type_ids)
        out["should_call"] = torch.full((input_ids.shape[0],), self.logit)
        return out


def _pipeline(model, **kwargs) -> Method2Pipeline:
    import numpy as np

    names = [TOOL["name"], OTHER_TOOL["name"]]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    class FakeEncoder:
        def encode(self, texts, **_):
            return np.array([[0.95, 0.05] for _ in texts], dtype="float32")

    retriever = ToolRetriever(
        FakeEncoder(),
        names,
        embeddings,
        # τ = 0.9: theo cơ chế cũ, query này (top-1 = 0.95) LUÔN được gọi.
        RetrievalThresholds(tau=0.9, tau_call=0.9, k_max=3, strategy="absolute"),
        normalize_query=False,
    )
    extractor = CrossEncoderExtractor(
        model, FakeTokenizer(), ExtractionConfig(max_length=64, batch_size=8)
    )
    return Method2Pipeline(
        retriever,
        extractor,
        ArgumentValidator(),
        {TOOL["name"]: TOOL, OTHER_TOOL["name"]: OTHER_TOOL},
        retrieval_scope="candidates",
        **kwargs,
    )


def _sample() -> dict:
    return {
        "id": "s1",
        "query": QUERY,
        "function_calls": [{"name": TOOL["name"], "arguments": {"location": "Hà Nội"}}],
        "tools": [TOOL, OTHER_TOOL],
    }


def test_checkpoint_khong_co_head_thi_dung_han():
    with pytest.raises(ValueError, match="enable_should_call"):
        _pipeline(FakeCrossEncoder(), abstention=ABSTENTION_SHOULD_CALL)


def test_head_noi_khong_thi_abstain_du_cosine_rat_cao():
    pipeline = _pipeline(
        ModelWithShouldCall(logit=-6.0),
        abstention=ABSTENTION_SHOULD_CALL,
        should_call_threshold=0.5,
    )

    prediction = pipeline.run_sample(_sample(), MODE_PIPELINE)["prediction"]

    assert prediction["metadata"]["abstained"] is True
    assert prediction["function_calls"] == []
    # Cùng sample, cùng ranking, cơ chế τ sẽ GỌI — chênh lệch này chính là ablation.
    baseline = _pipeline(ModelWithShouldCall(logit=-6.0), abstention=ABSTENTION_TAU)
    assert baseline.run_sample(_sample(), MODE_PIPELINE)["prediction"]["function_calls"]


def test_head_noi_co_thi_goi_binh_thuong():
    pipeline = _pipeline(
        ModelWithShouldCall(logit=6.0), abstention=ABSTENTION_SHOULD_CALL
    )

    prediction = pipeline.run_sample(_sample(), MODE_PIPELINE)["prediction"]

    assert prediction["metadata"]["abstained"] is False
    assert [c["name"] for c in prediction["function_calls"]] == [TOOL["name"]]


def test_prob_duoc_ghi_lai_de_quet_nguong_offline():
    pipeline = _pipeline(
        ModelWithShouldCall(logit=2.0), abstention=ABSTENTION_SHOULD_CALL
    )

    metadata = pipeline.run_sample(_sample(), MODE_PIPELINE)["prediction"]["metadata"]

    assert metadata["should_call_prob"] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-5)


def test_che_do_tau_khong_ghi_prob():
    pipeline = _pipeline(ModelWithShouldCall(logit=2.0), abstention=ABSTENTION_TAU)

    metadata = pipeline.run_sample(_sample(), MODE_PIPELINE)["prediction"]["metadata"]

    assert "should_call_prob" not in metadata


def test_chi_phi_head_tinh_vao_cross_encode_khong_phai_retrieve():
    """Đây là compute của Cross-Encoder; dồn vào `t_retrieve` là đổ nhầm sổ."""
    pipeline = _pipeline(
        ModelWithShouldCall(logit=6.0), abstention=ABSTENTION_SHOULD_CALL
    )
    baseline = _pipeline(ModelWithShouldCall(logit=6.0), abstention=ABSTENTION_TAU)

    with_head = pipeline.run_sample(_sample(), MODE_PIPELINE)["timings"]
    without = baseline.run_sample(_sample(), MODE_PIPELINE)["timings"]

    assert with_head.t_cross_encode > 0
    assert with_head.t_retrieve == pytest.approx(without.t_retrieve, rel=5.0)


def test_oracle_khong_dung_abstention():
    """Oracle bỏ qua retrieval hoàn toàn, nên cũng không có chỗ cho head này."""
    pipeline = _pipeline(
        ModelWithShouldCall(logit=-6.0), abstention=ABSTENTION_SHOULD_CALL
    )

    prediction = pipeline.run_sample(_sample(), "oracle")["prediction"]

    assert prediction["metadata"]["abstained"] is False
    assert "should_call_prob" not in prediction["metadata"]


def test_extractor_bao_dung_minh_co_head_hay_khong():
    with_head = CrossEncoderExtractor(
        ModelWithShouldCall(logit=0.0), FakeTokenizer(), ExtractionConfig()
    )
    without = CrossEncoderExtractor(
        FakeCrossEncoder(), FakeTokenizer(), ExtractionConfig()
    )

    assert with_head.has_should_call is True
    assert without.has_should_call is False
    assert without.should_call_prob(QUERY, TOOL) is None


def test_khong_co_ranking_thi_tra_none_chu_khong_phai_khong():
    """`None` = "chưa hỏi model", 0.0 = "model trả lời không". Khác nhau."""
    pipeline = _pipeline(
        ModelWithShouldCall(logit=6.0), abstention=ABSTENTION_SHOULD_CALL
    )

    assert pipeline._should_call_prob(QUERY, [], {}) is None


def test_model_gia_khong_co_thuoc_tinh_heads_van_hoi_duoc():
    """`has_should_call` phải chịu được model không có `.heads`."""
    extractor = CrossEncoderExtractor(
        SimpleNamespace(to=lambda d: None, eval=lambda: None),
        FakeTokenizer(),
        ExtractionConfig(),
    )

    assert extractor.has_should_call is False
