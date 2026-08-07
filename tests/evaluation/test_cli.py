"""Tests for CLI configuration and safe method mappings."""

from __future__ import annotations

import pytest
from src.evaluation.cli import _prediction_mapping


def test_prediction_mapping_rejects_path_like_method_name() -> None:
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        _prediction_mapping(["../escape=predictions.jsonl"], minimum_count=0)


def test_prediction_mapping_accepts_named_methods() -> None:
    mapping = _prediction_mapping(
        ["method_1=first.jsonl", "method-2=second.jsonl"],
        minimum_count=2,
    )
    assert set(mapping) == {"method_1", "method-2"}
