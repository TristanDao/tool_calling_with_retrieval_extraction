"""Incomplete gates cannot pass without per-call argument evaluation."""

from src.models.crossencoder.evaluate import check_gates


def test_missing_argument_gate_is_incomplete() -> None:
    result = check_gates({"enum_accuracy": 1.0}, {"enum_accuracy": 0.9, "argument_em": 0.7})
    assert not result["passed"]
    assert not result["complete"]
    assert result["missing_metrics"] == ["argument_em"]


def test_all_measured_gates_pass() -> None:
    result = check_gates({"enum_accuracy": 0.95, "argument_em": 0.75},
                         {"enum_accuracy": 0.9, "argument_em": 0.7})
    assert result["passed"] and result["complete"]


def test_model_preserves_optional_should_call_head(monkeypatch) -> None:
    from types import SimpleNamespace
    import torch
    from src.models.crossencoder.model import CrossEncoderForExtraction
    from src.models.crossencoder.heads import HeadConfig

    encoder = torch.nn.Identity()
    encoder.config = SimpleNamespace(hidden_size=8)
    monkeypatch.setattr("src.models.crossencoder.model.AutoModel.from_pretrained", lambda *args: encoder)
    model = CrossEncoderForExtraction("fake", HeadConfig(enable_should_call=True))
    assert model.head_config.hidden_size == 8
    assert model.heads.should_call is not None
