"""Exclude held-out tool descriptions from gradient training and negative mining."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_excluded(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = payload["excluded_tools"]
    if not names:
        raise ValueError("Strict unseen exclusion list is empty")
    return set(names)


def audit_rows(rows: Iterable[dict[str, Any]], excluded: set[str]) -> dict[str, Any]:
    hits: Counter[str] = Counter()
    total = 0
    for row in rows:
        total += 1
        for key in ("positive", "negatives", "all_gold", "candidates", "tool_name"):
            values = row.get(key) or []
            if isinstance(values, str):
                values = [values]
            for name in values:
                if name in excluded:
                    hits[key] += 1
    return {"n_rows": total, "excluded_occurrences": dict(hits), "passed": not hits}


def assert_no_exposure(rows: Iterable[dict[str, Any]], excluded: set[str]) -> None:
    result = audit_rows(rows, excluded)
    if not result["passed"]:
        raise ValueError(f"Strict unseen training exposure: {result['excluded_occurrences']}")


def filter_training_rows(
    rows: Iterable[dict[str, Any]], pool_names: Iterable[str], excluded: set[str],
    n_negatives: int = 4, seed: int = 42,
) -> list[dict[str, Any]]:
    allowed = sorted(set(pool_names) - excluded)
    allowed_set = set(allowed)
    result: list[dict[str, Any]] = []
    for row in rows:
        gold = set(row.get("all_gold") or ([row["positive"]] if row.get("positive") else []))
        if gold & excluded:
            raise ValueError(f"Held-out tool is a training positive: {row.get('sample_id')}")
        cleaned = {**row, "candidates": [n for n in row.get("candidates", []) if n not in excluded]}
        negatives = list(dict.fromkeys(n for n in row.get("negatives", []) if n in allowed_set and n not in gold))
        if row.get("positive"):
            if len(negatives) < n_negatives:
                fallback = [n for n in allowed if n not in gold and n not in negatives]
                random.Random(f"{seed}:{row.get('sample_id')}:{row['positive']}").shuffle(fallback)
                negatives += fallback[:n_negatives - len(negatives)]
            negatives = negatives[:n_negatives]
            if len(negatives) != n_negatives:
                raise ValueError("Insufficient allowed negatives")
        cleaned["negatives"] = negatives
        result.append(cleaned)
    assert_no_exposure(result, excluded)
    return result
