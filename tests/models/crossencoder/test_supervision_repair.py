"""Mention supervision preserves units, repeated-call identity, and absent labels."""

import copy

import pytest

from src.models.crossencoder.supervision import annotated_pairs


def sample() -> dict:
    query = "Vay 100 triệu rồi vay 20 triệu."
    return {"id": "case", "query": query, "tools": [{"name": "loan", "parameters": {
        "properties": {"principal_vnd": {"type": "integer"}, "optional": {"type": "string"}},
        "required": ["principal_vnd"]}}],
        "function_calls": [{"name": "loan", "arguments": {"principal_vnd": value}}
                           for value in [100000000, 20000000]],
        "metadata": {"tool_split": "seen", "argument_mentions": [
            {"call_index": i, "parameter_path": "principal_vnd", "surface": text,
             "start": query.index(text), "end": query.index(text) + len(text), "canonical_value": value}
            for i, (text, value) in enumerate([("100 triệu", 100000000), ("20 triệu", 20000000)])]}}


def test_full_numeric_unit_and_call_specific_offsets() -> None:
    data = sample()
    rows = annotated_pairs(data, "train", "custom_train")
    positive = [r for r in rows if r["labels"]["has_value"]]
    assert [data["query"][r["labels"]["char_start"]:r["labels"]["char_end"]] for r in positive] == ["100 triệu", "20 triệu"]
    assert [r["call_index"] for r in positive] == [0, 1]
    assert all(r["surface_normalizes_to_gold"] for r in positive)
    assert len([r for r in rows if r["labels"]["has_value"] == 0]) == 2


@pytest.mark.parametrize("field,value", [("start", -1), ("end", 999), ("surface", "100"), ("canonical_value", 100)])
def test_rejects_inconsistent_annotation(field, value) -> None:
    data = sample()
    data["metadata"]["argument_mentions"][0][field] = value
    with pytest.raises(ValueError, match="Invalid mention"):
        annotated_pairs(data, "train", "custom_train")


def test_missing_or_duplicate_mentions_are_not_guessed() -> None:
    data = sample()
    data["metadata"]["argument_mentions"] = []
    with pytest.raises(ValueError, match="Missing mention"):
        annotated_pairs(data, "train", "custom_train")
    data = sample()
    data["metadata"]["argument_mentions"].append(copy.deepcopy(data["metadata"]["argument_mentions"][0]))
    with pytest.raises(ValueError, match="Duplicate mention"):
        annotated_pairs(data, "train", "custom_train")


def test_alias_surface_does_not_pretend_to_be_canonical() -> None:
    data = {"id": "alias", "query": "Tới Sài Gòn", "tools": [{"name": "search", "parameters": {
        "properties": {"location": {"type": "string"}}, "required": ["location"]}}],
        "function_calls": [{"name": "search", "arguments": {"location": "TP. Hồ Chí Minh"}}],
        "metadata": {"tool_split": "seen", "argument_mentions": [{"call_index": 0, "parameter_path": "location",
                    "surface": "Sài Gòn", "start": 4, "end": 11, "canonical_value": "TP. Hồ Chí Minh"}]}}
    row = annotated_pairs(data, "train", "custom_train")[0]
    assert row["labels"]["has_value"] == 1
    assert row["surface_normalizes_to_gold"] is False
