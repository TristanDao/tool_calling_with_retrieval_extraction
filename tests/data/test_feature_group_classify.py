"""Tests for feature-group classification helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.data import feature_group_classify as module


def test_smoke_test_feature_group_api_success(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Tài chính & Ngân hàng")
                    )
                ]
            )

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(module, "AsyncOpenAI", lambda **kwargs: FakeClient())
    cfg = {
        "api": {
            "api_key": "test-key",
            "base_url": "https://example.test/v1",
            "model": "test-model",
            "temperature": 0.0,
            "max_tokens": 64,
        },
        "categories": ["Tài chính & Ngân hàng", "Khác"],
    }
    tool = {"name": "get_stock_price", "description": "Lấy giá cổ phiếu"}

    result = asyncio.run(module.smoke_test_feature_group_api(tool, cfg))

    assert result["ok"] is True
    assert result["feature_group"] == "Tài chính & Ngân hàng"
    assert result["valid_category"] is True


def test_smoke_test_feature_group_api_missing_key():
    result = asyncio.run(
        module.smoke_test_feature_group_api(
            {"name": "tool"},
            {"api": {"api_key": ""}},
        )
    )

    assert result == {
        "ok": False,
        "model": "",
        "error": "api_key missing in config",
    }
