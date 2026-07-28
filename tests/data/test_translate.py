"""Tests for translation pipeline (no API calls required)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest


def test_snake_case_validator():
    from src.data.translate_guidelines import is_snake_case
    assert is_snake_case("search_tutors") is True
    assert is_snake_case("get_user_profile_v2") is True
    assert is_snake_case("a") is True
    assert is_snake_case("SearchTutors") is False
    assert is_snake_case("search-tutors") is False
    assert is_snake_case("searchTutors") is False
    assert is_snake_case("123abc") is False
    assert is_snake_case("") is False
    assert is_snake_case("a b") is False


def test_extract_function_names_glaive():
    from src.data.translate_guidelines import extract_function_names_glaive
    chat = (
        "USER: I want a tutor.\n\n"
        "A: <functioncall> {\"name\": \"search_tutors\", \"arguments\": '{\"subject\": \"Math\"}'} <|endoftext|>\n\n"
        "A: <functioncall> {\"name\": \"get_user_profile\", \"arguments\": '{}'} <|endoftext|>\n"
    )
    names = extract_function_names_glaive(chat)
    assert names == ["search_tutors", "get_user_profile"]


def test_extract_function_names_xlam():
    from src.data.translate_guidelines import extract_function_names_xlam
    answers = json.dumps([
        {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
        {"name": "web_chain_details", "arguments": {"chain_slug": "ethereum"}},
    ])
    names = extract_function_names_xlam(answers)
    assert names == ["live_giveaways_by_type", "web_chain_details"]


def test_extract_function_names_xlam_invalid_json():
    from src.data.translate_guidelines import extract_function_names_xlam
    assert extract_function_names_xlam("not json") == []
    assert extract_function_names_xlam("") == []


def test_check_identifier_integrity_glaive_ok():
    from src.data.translate_guidelines import check_identifier_integrity
    original = {
        "system": 'SYSTEM: {"name": "search_tutors", "description": "Find tutors"}',
        "chat": 'A: <functioncall> {"name": "search_tutors", "arguments": \'{"subject": "Math"}\'} <|endoftext|>',
    }
    translated = {
        "system": 'SYSTEM: {"name": "search_tutors", "description": "Tìm gia sư"}',
        "chat": 'A: <functioncall> {"name": "search_tutors", "arguments": \'{"subject": "Toán"}\'} <|endoftext|>',
    }
    ok, reason = check_identifier_integrity(original, translated, "glaive")
    assert ok is True, reason


def test_check_identifier_integrity_glaive_missing_function():
    from src.data.translate_guidelines import check_identifier_integrity
    original = {
        "system": "SYSTEM: {}",
        "chat": 'A: <functioncall> {"name": "search_tutors", "arguments": \'{}\'} <|endoftext|>',
    }
    translated = {
        "system": "SYSTEM: {}",
        "chat": 'A: <functioncall> {"name": "search_tutors_changed", "arguments": \'{}\'} <|endoftext|>',
    }
    ok, reason = check_identifier_integrity(original, translated, "glaive")
    assert ok is False
    assert "function names changed" in reason


def test_check_identifier_integrity_xlam_ok():
    from src.data.translate_guidelines import check_identifier_integrity
    original = {
        "id": 0,
        "query": "Find live giveaways",
        "answers": json.dumps([{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}]),
        "tools": json.dumps([{"name": "live_giveaways_by_type", "description": "..."}]),
    }
    translated = {
        "id": 0,
        "query": "Tìm chương trình tặng quà",
        "answers": json.dumps([{"name": "live_giveaways_by_type", "arguments": {"type": "beta"}}]),
        "tools": json.dumps([{"name": "live_giveaways_by_type", "description": "..."}]),
    }
    ok, reason = check_identifier_integrity(original, translated, "xlam")
    assert ok is True, reason


def test_check_identifier_integrity_xlam_missing_key():
    from src.data.translate_guidelines import check_identifier_integrity
    original = {
        "id": 0,
        "query": "Find live giveaways",
        "answers": json.dumps([]),
        "tools": json.dumps([]),
    }
    translated = {"id": 0, "query": "Tìm...", "answers": "[]", "tools": "[]"}
    translated["query"] = ""
    ok, reason = check_identifier_integrity(original, translated, "xlam")
    assert ok is True


def test_check_required_fields_present_glaive():
    from src.data.translate_guidelines import check_required_fields_present
    assert check_required_fields_present({"system": "x", "chat": "y"}, "glaive") == (True, "ok")
    assert check_required_fields_present({"system": "", "chat": "y"}, "glaive")[0] is False
    assert check_required_fields_present({"system": "x"}, "glaive")[0] is False


def test_check_required_fields_present_xlam():
    from src.data.translate_guidelines import check_required_fields_present
    sample = {"id": 0, "query": "q", "answers": "[]", "tools": "[]"}
    assert check_required_fields_present(sample, "xlam") == (True, "ok")
    bad = {"id": 0, "query": "", "answers": "[]", "tools": "[]"}
    assert check_required_fields_present(bad, "xlam")[0] is False


def test_checkpoint_atomic_save_load():
    from src.data.translation_checkpoint import Checkpoint, load, save_atomic
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_path = Path(tmpdir) / "test.json"
        cp = Checkpoint(dataset="test")
        cp.advance(n_success=10, n_failed=2)
        save_atomic(cp, cp_path)

        loaded = load(cp_path, dataset="test")
        assert loaded.dataset == "test"
        assert loaded.total_success == 10
        assert loaded.total_failed == 2
        assert loaded.last_processed_index == 12


def test_checkpoint_atomic_overwrites_existing():
    from src.data.translation_checkpoint import Checkpoint, load, save_atomic
    with tempfile.TemporaryDirectory() as tmpdir:
        cp_path = Path(tmpdir) / "test.json"
        save_atomic(Checkpoint(dataset="test", last_processed_index=100), cp_path)

        cp = Checkpoint(dataset="test", last_processed_index=50, total_success=48, total_failed=2)
        save_atomic(cp, cp_path)
        loaded = load(cp_path, dataset="test")
        assert loaded.last_processed_index == 50
        assert loaded.total_success == 48


def test_checkpoint_load_nonexistent_returns_default():
    from src.data.translation_checkpoint import Checkpoint, load
    with tempfile.TemporaryDirectory() as tmpdir:
        cp = load(Path(tmpdir) / "missing.json", dataset="glaive")
        assert cp.dataset == "glaive"
        assert cp.last_processed_index == 0


def test_parse_response_clean_json():
    from src.data.translate import parse_response
    out = parse_response('{"a": 1, "b": "x"}')
    assert out == {"a": 1, "b": "x"}


def test_parse_response_with_code_fence():
    from src.data.translate import parse_response
    out = parse_response("```json\n{\"a\": 1}\n```")
    assert out == {"a": 1}


def test_parse_response_with_surrounding_text():
    from src.data.translate import parse_response
    out = parse_response("Here is the translation:\n{\"a\": 1}\nDone.")
    assert out == {"a": 1}


def test_parse_response_invalid_raises():
    from src.data.translate import parse_response
    with pytest.raises(ValueError):
        parse_response("no json here at all")


def test_translate_prompt_glaive():
    from src.data.translate_guidelines import build_translate_prompt
    sample = {
        "system": "SYSTEM: {\"name\": \"x\"}",
        "chat": "USER: hi",
    }
    prompt = build_translate_prompt(sample, "glaive")
    assert "system" in prompt
    assert "chat" in prompt
    assert "x" in prompt


def test_translate_prompt_xlam():
    from src.data.translate_guidelines import build_translate_prompt
    sample = {"id": 0, "query": "hi", "answers": "[]", "tools": "[]"}
    prompt = build_translate_prompt(sample, "xlam")
    assert "query" in prompt
    assert "answers" in prompt
    assert "tools" in prompt


def test_translate_prompt_unknown_dataset():
    from src.data.translate_guidelines import build_translate_prompt
    with pytest.raises(ValueError):
        build_translate_prompt({}, "unknown")


def test_read_jsonl_generator_skips_invalid():
    from src.data.translate import read_jsonl_generator
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.jsonl"
        p.write_text(
            '{"a": 1}\n'
            'not valid json\n'
            '{"a": 2}\n'
            '\n'
            '{"a": 3}\n',
            encoding="utf-8",
        )
        results = list(read_jsonl_generator(p))
        assert len(results) == 3
        assert [r[0] for r in results] == [0, 2, 4]
        assert [r[1]["a"] for r in results] == [1, 2, 3]


def test_read_jsonl_generator_respects_start_end():
    from src.data.translate import read_jsonl_generator
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.jsonl"
        p.write_text("\n".join(f'{{"i": {i}}}' for i in range(10)), encoding="utf-8")
        results = list(read_jsonl_generator(p, start=3, end=6))
        assert [r[0] for r in results] == [3, 4, 5]
        assert [r[1]["i"] for r in results] == [3, 4, 5]


def test_qa_rule_check_pass():
    from src.data.qa_translation import rule_check_sample, QAConfig
    cfg = QAConfig(
        input_path=Path("/tmp/in"),
        output_path=Path("/tmp/out"),
        report_path=Path("/tmp/r"),
        dataset="glaive",
        rule_check_enabled=True,
        require_keys=["system", "chat"],
        identifier_pattern="^[a-z][a-z0-9_]*$",
        llm_judge_enabled=False,
        sample_rate=0.0,
        api_base_url="",
        api_key="",
        api_model="",
        api_temperature=0.0,
        api_max_tokens=0,
        concurrency=1,
        fail_on_rule_violation=False,
    )
    sample = {
        "system": 'SYSTEM: {"name": "search_tutors", "description": "x"}',
        "chat": 'A: <functioncall> {"name": "search_tutors", "arguments": \'{}\'} <|endoftext|>',
    }
    ok, errors = rule_check_sample(sample, cfg)
    assert ok is True, errors


def test_qa_rule_check_camel_case_function():
    from src.data.qa_translation import rule_check_sample, QAConfig
    cfg = QAConfig(
        input_path=Path("/tmp/in"),
        output_path=Path("/tmp/out"),
        report_path=Path("/tmp/r"),
        dataset="glaive",
        rule_check_enabled=True,
        require_keys=["system", "chat"],
        identifier_pattern="^[a-z][a-z0-9_]*$",
        llm_judge_enabled=False,
        sample_rate=0.0,
        api_base_url="",
        api_key="",
        api_model="",
        api_temperature=0.0,
        api_max_tokens=0,
        concurrency=1,
        fail_on_rule_violation=False,
    )
    sample = {
        "system": "SYSTEM: {}",
        "chat": 'A: <functioncall> {"name": "SearchTutors", "arguments": \'{}\'} <|endoftext|>',
    }
    ok, errors = rule_check_sample(sample, cfg)
    assert ok is False
    assert any("snake_case" in e for e in errors)


def test_translation_config_backup_model():
    from src.data.translate import TranslationConfig
    cfg = TranslationConfig.from_dict({
        "input": "in.jsonl",
        "output": "out.jsonl",
        "failed_output": "failed.jsonl",
        "checkpoint": "cp.json",
        "dataset": "xlam",
        "api": {
            "base_url": "http://x",
            "api_key": "k",
            "model": "main",
            "backup_model": "backup",
        },
    })
    assert cfg.api_model == "main"
    assert cfg.api_backup_model == "backup"


def test_translation_config_no_backup_model():
    from src.data.translate import TranslationConfig
    cfg = TranslationConfig.from_dict({
        "input": "in.jsonl",
        "output": "out.jsonl",
        "failed_output": "failed.jsonl",
        "checkpoint": "cp.json",
        "dataset": "xlam",
        "api": {
            "base_url": "http://x",
            "api_key": "k",
            "model": "main",
        },
    })
    assert cfg.api_model == "main"
    assert cfg.api_backup_model is None


def test_translation_config_empty_backup_model():
    from src.data.translate import TranslationConfig
    cfg = TranslationConfig.from_dict({
        "input": "in.jsonl",
        "output": "out.jsonl",
        "failed_output": "failed.jsonl",
        "checkpoint": "cp.json",
        "dataset": "xlam",
        "api": {
            "base_url": "http://x",
            "api_key": "k",
            "model": "main",
            "backup_model": "",
        },
    })
    assert cfg.api_backup_model is None


def test_resolve_env_placeholder():
    from src.data.translate import _resolve_env_placeholders
    import os
    os.environ["TEST_XLAM_VAR"] = "hello"
    assert _resolve_env_placeholders("${TEST_XLAM_VAR}") == "hello"
    assert _resolve_env_placeholders("${NONEXISTENT_VAR_XLAM}") == ""
    del os.environ["TEST_XLAM_VAR"]


def test_resolve_env_placeholder_double_underscore():
    from src.data.translate import _resolve_env_placeholders
    import os
    os.environ["FOO__BAR"] = "double_underscore_val"
    assert _resolve_env_placeholders("${FOO_BAR}") == "double_underscore_val"
    del os.environ["FOO__BAR"]


def test_is_fallback_worthy_recognizes_rate_limit():
    from src.data.translate import _is_fallback_worthy
    from openai import RateLimitError, APITimeoutError
    fake_rate = RateLimitError.__new__(RateLimitError)
    fake_rate.status_code = 429
    assert _is_fallback_worthy(fake_rate) is True

    fake_timeout = APITimeoutError.__new__(APITimeoutError)
    assert _is_fallback_worthy(fake_timeout) is True

    assert _is_fallback_worthy(ValueError("not http")) is False


def test_translate_one_uses_backup_on_rate_limit(monkeypatch):
    """When main model returns RateLimitError, fall back to backup model."""
    from src.data.translate import (
        TranslationConfig,
        translate_one,
        _is_fallback_worthy,
    )
    from openai import RateLimitError

    cfg = TranslationConfig(
        input_path=Path("/tmp/in"),
        output_path=Path("/tmp/out"),
        failed_path=Path("/tmp/failed"),
        checkpoint_path=Path("/tmp/cp"),
        log_path=Path("/tmp/log"),
        dataset="glaive",
        start_index=0,
        end_index=None,
        batch_size=1,
        concurrency=1,
        max_retries=2,
        retry_initial_delay=0.0,
        retry_backoff=2.0,
        api_base_url="http://x",
        api_key="k",
        api_model="main_model",
        api_backup_model="backup_model",
        api_temperature=0.1,
        api_timeout=10.0,
        api_max_tokens=1024,
        progress_log_every_n_batches=5,
    )

    call_log: list[str] = []
    fake_response_main = RateLimitError.__new__(RateLimitError)
    fake_response_main.status_code = 429

    class _Msg:
        content = '{"system": "SYSTEM: {}", "chat": "USER: hi"}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    async def fake_create(**kwargs):
        model = kwargs.get("model")
        call_log.append(model)
        if model == "main_model":
            raise RateLimitError.__new__(RateLimitError)
        return _Resp()

    class _FakeClient:
        chat = type("Chat", (), {"completions": type("CC", (), {"create": staticmethod(fake_create)})()})()

    import asyncio
    semaphore = asyncio.Semaphore(1)

    async def run():
        return await translate_one(
            _FakeClient(),
            {"system": "SYSTEM: {}", "chat": "USER: hi"},
            42,
            cfg,
            semaphore,
            log_callback=lambda m: call_log.append(f"LOG:{m}"),
        )

    result = asyncio.run(run())
    assert result[0] == 42
    assert result[1] is not None
    assert result[1] == {"system": "SYSTEM: {}", "chat": "USER: hi"}
    assert "main_model" in call_log
    assert "backup_model" in call_log


def test_translate_one_no_backup_just_retries(monkeypatch):
    """When no backup model configured, retry on main only."""
    from src.data.translate import TranslationConfig, translate_one
    from openai import APITimeoutError

    cfg = TranslationConfig(
        input_path=Path("/tmp/in"),
        output_path=Path("/tmp/out"),
        failed_path=Path("/tmp/failed"),
        checkpoint_path=Path("/tmp/cp"),
        log_path=Path("/tmp/log"),
        dataset="glaive",
        start_index=0,
        end_index=None,
        batch_size=1,
        concurrency=1,
        max_retries=2,
        retry_initial_delay=0.0,
        retry_backoff=2.0,
        api_base_url="http://x",
        api_key="k",
        api_model="main_only",
        api_backup_model=None,
        api_temperature=0.1,
        api_timeout=10.0,
        api_max_tokens=1024,
        progress_log_every_n_batches=5,
    )

    call_count = {"n": 0}

    class _Msg:
        content = '{"system": "SYSTEM: {}", "chat": "USER: hi"}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    async def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise APITimeoutError.__new__(APITimeoutError)
        return _Resp()

    class _FakeClient:
        chat = type("Chat", (), {"completions": type("CC", (), {"create": staticmethod(fake_create)})()})()

    import asyncio
    semaphore = asyncio.Semaphore(1)

    async def run():
        return await translate_one(
            _FakeClient(),
            {"system": "SYSTEM: {}", "chat": "USER: hi"},
            99,
            cfg,
            semaphore,
        )

    result = asyncio.run(run())
    assert result[0] == 99
    assert result[1] is not None
    assert call_count["n"] == 2
