"""Tests for raw Glaive single-turn filtering."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.filter_single_turn import filter_glaive_single_turn


def test_filter_glaive_preserves_raw_shape(tmp_path: Path):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "filtered.jsonl"
    index_map = tmp_path / "index.jsonl"
    source.write_text(
        json.dumps(
            {
                "system": 'SYSTEM: {"name": "tool_a", "description": "A", "parameters": {}}',
                "chat": 'USER: Call A\n\nA: <functioncall> {"name": "tool_a", "arguments": \'{}\'} <|endoftext|>',
            }
        )
        + "\n"
        + json.dumps({"system": "", "chat": "USER: Hello\n\nA: Hi"})
        + "\n",
        encoding="utf-8",
    )

    stats = filter_glaive_single_turn(source, output, index_map)

    assert stats == {"total_raw": 2, "kept": 1, "skipped": 1}
    filtered = json.loads(output.read_text(encoding="utf-8"))
    assert set(filtered) == {"system", "chat"}
    mapping = json.loads(index_map.read_text(encoding="utf-8"))
    assert mapping == {"filtered_index": 0, "source_index": 0}
