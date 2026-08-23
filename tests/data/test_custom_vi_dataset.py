"""Tests for CustomTools-VI generation and validation."""

import hashlib
import json
from pathlib import Path

import pytest
from src.data.audit_custom_vi_dataset import BOOLEAN_SURFACES
from src.data.custom_vi_dataset import Config, build_tools, generate, reconstruct, validate


@pytest.fixture(scope="module")
def generated_dataset() -> tuple[Config, list[dict], list[dict]]:
    config = Config.load(Path("configs/data/custom_vi.yaml"))
    tools, records = generate(config)
    return config, tools, records


def test_tool_ontology_has_expected_shape() -> None:
    tools = build_tools()
    assert len(tools) == 40
    assert len({tool["name"] for tool in tools}) == 40
    assert len({tool["x-domain"] for tool in tools}) == 10
    assert sum(tool["x-tool-split"] == "seen" for tool in tools) == 20
    assert sum(tool["x-tool-split"] == "dev_unseen" for tool in tools) == 10
    assert sum(tool["x-tool-split"] == "test_unseen" for tool in tools) == 10
    assert all(1 <= len(tool["parameters"]["properties"]) <= 5 for tool in tools)


def test_full_generation_passes_contract(
    generated_dataset: tuple[Config, list[dict], list[dict]],
) -> None:
    config, tools, records = generated_dataset
    report = validate(tools, records, config)
    assert len(records) == 8000
    assert report["status"] == "passed", report["errors"][:3]
    assert report["normalized_duplicate_queries"] == 0
    assert report["cross_split_scenario_family_leakage"] == 0
    assert report["argument_mention_reconstruction_passed"] == 8000
    assert report["lexical_near_duplicate_audit"]["pairs"] == 0


def test_argument_mentions_reconstruct_gold_calls(
    generated_dataset: tuple[Config, list[dict], list[dict]],
) -> None:
    _, _, records = generated_dataset
    for record in records:
        assert reconstruct(record) == record["function_calls"]
        spans = sorted(
            (mention["start"], mention["end"])
            for mention in record["metadata"]["argument_mentions"]
        )
        assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:], strict=False))
        assert "audit_status" not in record["metadata"]


def test_boolean_surfaces_and_zodiac_are_semantically_consistent(
    generated_dataset: tuple[Config, list[dict], list[dict]],
) -> None:
    _, _, records = generated_dataset
    for record in records:
        for mention in record["metadata"]["argument_mentions"]:
            key = mention["parameter_path"]
            value = mention["canonical_value"]
            if key in BOOLEAN_SURFACES and isinstance(value, bool):
                assert mention["surface"] == BOOLEAN_SURFACES[key][value]
        for call in record["function_calls"]:
            if call["name"] == "vi_lookup_zodiac_information" and "zodiac" in call["arguments"]:
                zodiac_order = [
                    "Tý",
                    "Sửu",
                    "Dần",
                    "Mão",
                    "Thìn",
                    "Tỵ",
                    "Ngọ",
                    "Mùi",
                    "Thân",
                    "Dậu",
                    "Tuất",
                    "Hợi",
                ]
                expected = zodiac_order[(call["arguments"]["birth_year"] - 4) % 12]
                assert call["arguments"]["zodiac"] == expected


def test_generation_is_reproducible(
    generated_dataset: tuple[Config, list[dict], list[dict]],
) -> None:
    config, expected_tools, expected_records = generated_dataset
    actual_tools, actual_records = generate(config)

    def digest(value: object) -> str:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    assert digest(actual_tools) == digest(expected_tools)
    assert digest(actual_records) == digest(expected_records)
