"""Smoke test toàn tuyến Bi → Cross → Validator bằng model giả.

Mục đích: kiểm tra **wiring** (batch theo param, offset → text, ghép argument,
validator, prediction contract, latency 4 giai đoạn) trước khi tốn giờ GPU.
Chất lượng model không nằm trong phạm vi test này — logit được đặt tay để span
rơi đúng vào chỗ mình muốn.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "crossencoder"))
from conftest import FakeTokenizer  # noqa: E402

from src.models.biencoder.retrieve import RetrievalThresholds, ToolRetriever  # noqa: E402
from src.models.crossencoder.inference import (  # noqa: E402
    CrossEncoderExtractor,
    ExtractionConfig,
)
from src.models.pipeline.method2 import MODE_ORACLE, MODE_PIPELINE, Method2Pipeline  # noqa: E402
from src.models.pipeline.validator import ArgumentValidator  # noqa: E402

QUERY = "Tìm quán phở bò ở Hà Nội"
TOOL = {
    "name": "vi_search_restaurants",
    "description": "Tìm quán ăn theo địa điểm",
    "doc_text": "vi_search_restaurants. Tìm quán ăn theo địa điểm",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Địa điểm"},
            "spicy": {"type": "boolean", "description": "Có cay không"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["location"],
    },
}
OTHER_TOOL = {
    "name": "vi_order_food",
    "description": "Đặt món ăn",
    "doc_text": "vi_order_food. Đặt món ăn",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class FakeCrossEncoder:
    """Trả logit đặt tay: has_value cao, span trỏ vào 2 token cuối của query."""

    def __init__(self, has_value_logit: float = 4.0) -> None:
        self.has_value_logit = has_value_logit

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, query_token_mask, token_type_ids=None):
        batch, length = input_ids.shape
        query_lengths = query_token_mask.sum(dim=1)
        span_start = torch.zeros(batch, length)
        span_end = torch.zeros(batch, length)
        for row in range(batch):
            last = int(query_lengths[row])  # token cuối của query (index 1..last)
            span_start[row, last - 1] = 10.0
            span_end[row, last] = 10.0
        return {
            "has_value": torch.full((batch,), self.has_value_logit),
            "span_start": span_start.masked_fill(~query_token_mask, -1e4),
            "span_end": span_end.masked_fill(~query_token_mask, -1e4),
            "enum_logits": torch.zeros(batch, 20),
            "boolean_logits": torch.tensor([[0.0, 5.0]] * batch),  # → False
        }


class FakeEncoder:
    """SentenceTransformer giả: embedding trùng khớp token đầu tiên của tên tool."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def encode(self, texts, **kwargs):
        import numpy as np

        return np.array([self.mapping[t] for t in texts], dtype="float32")


@pytest.fixture
def pipeline():
    import numpy as np

    tool_pool = {TOOL["name"]: TOOL, OTHER_TOOL["name"]: OTHER_TOOL}
    names = [TOOL["name"], OTHER_TOOL["name"]]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    encoder = FakeEncoder({QUERY: [0.95, 0.05], "Xin chào": [0.2, 0.2]})
    retriever = ToolRetriever(
        encoder,
        names,
        embeddings,
        RetrievalThresholds(tau=0.5, tau_call=0.9, k_max=3, strategy="absolute"),
        normalize_query=False,
    )
    extractor = CrossEncoderExtractor(
        FakeCrossEncoder(),
        FakeTokenizer(),
        ExtractionConfig(max_length=64, batch_size=8),
    )
    return Method2Pipeline(
        retriever, extractor, ArgumentValidator(), tool_pool, retrieval_scope="candidates"
    )


def _sample(sample_id: str = "s1", query: str = QUERY) -> dict:
    return {
        "id": sample_id,
        "query": query,
        "function_calls": [
            {"name": TOOL["name"], "arguments": {"location": "Hà Nội", "spicy": False}}
        ],
        "tools": [TOOL, OTHER_TOOL],
    }


def test_pipeline_produces_prediction_contract(pipeline):
    outcome = pipeline.run_sample(_sample(), mode=MODE_PIPELINE)

    prediction = outcome["prediction"]
    assert prediction["id"] == "s1"
    assert [c["name"] for c in prediction["function_calls"]] == [TOOL["name"]]
    assert prediction["ranked_tools"][0]["name"] == TOOL["name"]
    assert set(prediction["telemetry"]) >= {
        "latency_ms",
        "t_query_embed", "t_retrieve", "t_cross_encode", "t_validate",
    }
    assert "cost_usd" not in prediction["telemetry"]
    assert prediction["metadata"]["cost_basis"] == "unavailable_gpu_cost"
    assert prediction["metadata"]["abstained"] is False


def test_span_becomes_argument_text_from_the_original_query(pipeline):
    outcome = pipeline.run_sample(_sample(), mode=MODE_PIPELINE)

    arguments = outcome["prediction"]["function_calls"][0]["arguments"]
    # Logit giả trỏ vào 2 token cuối → "Hà Nội", lấy theo offset trên query gốc.
    assert arguments["location"] == "Hà Nội"
    assert arguments["spicy"] is False


def test_unsupported_array_param_never_enters_arguments(pipeline):
    outcome = pipeline.run_sample(_sample(), mode=MODE_PIPELINE)

    assert "tags" not in outcome["prediction"]["function_calls"][0]["arguments"]
    unsupported = outcome["raw"]["calls"][0]["unsupported_params"]
    assert "tags" in unsupported


def test_abstention_returns_no_call_below_tau(pipeline):
    sample = _sample("neg", query="Xin chào")
    sample["function_calls"] = []

    outcome = pipeline.run_sample(sample, mode=MODE_PIPELINE)

    assert outcome["prediction"]["function_calls"] == []
    assert outcome["prediction"]["metadata"]["abstained"] is True


def test_oracle_mode_skips_retrieval_entirely(pipeline):
    outcome = pipeline.run_sample(_sample(), mode=MODE_ORACLE)

    assert [c["name"] for c in outcome["prediction"]["function_calls"]] == [TOOL["name"]]
    assert outcome["prediction"]["ranked_tools"] == []
    assert outcome["timings"].t_query_embed == 0.0
    assert outcome["timings"].t_retrieve == 0.0
    assert outcome["timings"].t_cross_encode > 0.0


def test_run_dataset_writes_contract_files(pipeline, tmp_path):
    report = pipeline.run_dataset([_sample(), _sample("s2")], MODE_PIPELINE, tmp_path)

    assert report["n_samples"] == 2
    predictions = [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [p["id"] for p in predictions] == ["s1", "s2"]
    assert (tmp_path / "raw_predictions_pipeline.jsonl").exists()
    assert (tmp_path / "errors_pipeline.jsonl").exists()

    latency = json.loads((tmp_path / "latency_pipeline.json").read_text(encoding="utf-8"))
    assert latency["n"] == 2
    assert latency["total"]["p95_ms"] >= latency["total"]["p50_ms"]


def test_oracle_mode_writes_separate_file(pipeline, tmp_path):
    pipeline.run_dataset([_sample()], MODE_ORACLE, tmp_path)

    assert (tmp_path / "oracle_predictions.jsonl").exists()
    assert not (tmp_path / "predictions.jsonl").exists()


def test_low_has_value_confidence_drops_optional_but_keeps_required(tmp_path):
    import numpy as np

    # has_value logit âm → prob < 0.5 nên mọi param bị lọc; validator phải lấy
    # lại `location` (required) ở ngưỡng fallback 0.3.
    extractor = CrossEncoderExtractor(
        FakeCrossEncoder(has_value_logit=-0.5),
        FakeTokenizer(),
        ExtractionConfig(max_length=64, has_value_threshold=0.5, fallback_has_value_threshold=0.3),
    )
    retriever = ToolRetriever(
        FakeEncoder({QUERY: [0.95, 0.05]}),
        [TOOL["name"], OTHER_TOOL["name"]],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
        RetrievalThresholds(tau=0.5, tau_call=0.9, k_max=3),
        normalize_query=False,
    )
    pipeline = Method2Pipeline(
        retriever,
        extractor,
        ArgumentValidator(fallback_threshold=0.3),
        {TOOL["name"]: TOOL, OTHER_TOOL["name"]: OTHER_TOOL},
    )

    outcome = pipeline.run_sample(_sample(), mode=MODE_PIPELINE)

    arguments = outcome["prediction"]["function_calls"][0]["arguments"]
    assert arguments["location"] == "Hà Nội"
    assert "spicy" not in arguments
