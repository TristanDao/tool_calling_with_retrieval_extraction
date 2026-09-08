"""Strict exclusion holds through deterministic negative replacement."""

import pytest

from src.models.biencoder.strict_unseen import assert_no_exposure, filter_training_rows


def test_replace_negatives_without_dropping_rows_or_gold() -> None:
    rows = [{"sample_id": "a", "positive": "gold", "all_gold": ["gold"],
             "negatives": ["heldout", "n1"], "candidates": ["gold", "heldout"]}]
    pool = ["gold", "heldout", "n1", "n2", "n3"]
    cleaned = filter_training_rows(rows, pool, {"heldout"}, 3)
    assert len(cleaned) == 1
    assert cleaned[0]["positive"] == "gold"
    assert len(cleaned[0]["negatives"]) == 3
    assert_no_exposure(cleaned, {"heldout"})
    assert cleaned == filter_training_rows(rows, pool, {"heldout"}, 3)
    assert "heldout" in rows[0]["negatives"]


def test_rejects_heldout_positive_and_existing_exposure() -> None:
    with pytest.raises(ValueError):
        filter_training_rows([{"positive": "heldout"}], ["heldout", "n"], {"heldout"})
    with pytest.raises(ValueError):
        assert_no_exposure([{"negatives": ["heldout"]}], {"heldout"})


def test_mining_never_encodes_heldout_documents(tmp_path, monkeypatch) -> None:
    import json
    import numpy as np
    from src.models.biencoder.train import mine_hard_negatives
    from src.models.sources import load_jsonl, write_jsonl

    encoded = []

    class Encoder:
        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            return np.ones((len(texts), 2))

    monkeypatch.setattr("src.models.biencoder.index.load_encoder", lambda *args: Encoder())
    monkeypatch.setattr("src.models.biencoder.tool_pool.load_tool_pool", lambda *args: {
        name: {"doc_text": name} for name in ["gold", "heldout", "n1", "n2", "n3"]})
    pairs = tmp_path / "train.jsonl"
    output = tmp_path / "mined.jsonl"
    exclusion = tmp_path / "excluded.json"
    exclusion.write_text(json.dumps({"excluded_tools": ["heldout"]}))
    write_jsonl(pairs, [{"positive": "gold", "query": "query", "negatives": ["n1"]}])
    mine_hard_negatives("fake", pairs, output, top_k=1, skip_top=1, n_negatives=3,
                        excluded_tools_path=exclusion)
    assert "heldout" not in encoded
    rows = list(load_jsonl(output))
    assert len(rows[0]["negatives"]) == 3
    assert_no_exposure(rows, {"heldout"})
