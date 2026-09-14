"""Generate Phase 7 Stress Test dataset with nested random distractors.

This script implements the exact protocol used by Method 2:
1. Samples 200 queries from `data/custom_vi/test_seen.jsonl` (100 positive + 100 negative, seed=42).
2. Pools 4,461 real tool schemas from `benchmark_vi/tool_pool.json` and `custom_vi/tools.json`.
3. Constructs nested haystacks for N in [3, 10, 50, 100, 500, 1000].
   - Nested guarantee: Prefix property ensures Haystack(N=10) contains Haystack(N=3).
   - Gold guarantee: Ground-truth tool is always included.
4. Outputs:
   - `data/processed/stress_test/anchors.jsonl`
   - `data/processed/stress_test/augmented/random_N{3,10,50,100,500,1000}.jsonl`
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


def sample_queries(
    samples: Iterable[dict[str, Any]],
    n_queries: int = 200,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample stratified positive/negative queries."""
    samples = list(samples)
    if n_queries <= 0 or n_queries > len(samples):
        raise ValueError("n_queries must be positive and <= dataset length")
    positives = [s for s in samples if s.get("function_calls")]
    negatives = [s for s in samples if not s.get("function_calls")]
    half = n_queries // 2
    rng = random.Random(seed)
    picked_ids = set()
    for group, quota in ((positives, half), (negatives, n_queries - half)):
        quota = min(quota, len(group))
        picked_ids.update(s["id"] for s in rng.sample(group, quota))
    remaining = [s for s in samples if s["id"] not in picked_ids]
    if len(picked_ids) < n_queries:
        picked_ids.update(s["id"] for s in rng.sample(remaining, n_queries - len(picked_ids)))
    return [s for s in samples if s["id"] in picked_ids]


def load_tool_pool(
    benchmark_pool_path: Path,
    custom_tools_path: Path,
) -> dict[str, dict[str, Any]]:
    """Merge core tool pool and custom tools into a unified dictionary."""
    pool: dict[str, dict[str, Any]] = {}
    if benchmark_pool_path.is_file():
        items = json.loads(benchmark_pool_path.read_text(encoding="utf-8"))
        for item in items:
            name = item.get("name")
            if name:
                pool[name] = item

    if custom_tools_path.is_file():
        custom_items = json.loads(custom_tools_path.read_text(encoding="utf-8"))
        for item in custom_items:
            name = item.get("name")
            if name and name not in pool:
                pool[name] = item

    return pool


def build_stress_instances(
    anchors: list[dict[str, Any]],
    pool: dict[str, dict[str, Any]],
    n_values: Sequence[int] = (3, 10, 50, 100, 500, 1000),
    seed: int = 42,
) -> dict[int, list[dict[str, Any]]]:
    """Build augmented instances for each N with nested distractors."""
    pool_names = sorted(pool.keys())
    augmented_by_n: dict[int, list[dict[str, Any]]] = {n: [] for n in n_values}

    for sample in anchors:
        sample_schemas = {t["name"]: t for t in (sample.get("tools") or []) if t.get("name")}
        gold_names = list(
            dict.fromkeys(
                c["name"] for c in (sample.get("function_calls") or []) if c.get("name")
            )
        )

        # Distractor order: deterministic shuffle per sample
        excluded = set(gold_names)
        ordered_distractors = [n for n in pool_names if n not in excluded]
        random.Random(f"{seed}:{sample['id']}:random").shuffle(ordered_distractors)

        for n in n_values:
            truncated = len(gold_names) > n
            kept_gold = gold_names[:n] if truncated else gold_names
            n_needed = max(n - len(kept_gold), 0)
            distractors = ordered_distractors[:n_needed]

            # Collect tool schemas
            tools = [sample_schemas.get(name) or pool.get(name) for name in kept_gold]
            tools += [pool[name] for name in distractors]
            tools = [t for t in tools if t is not None]

            augmented_by_n[n].append({
                "id": sample["id"],
                "query": sample["query"],
                "function_calls": sample.get("function_calls", []),
                "tools": tools,
                "n_target": n,
                "gold_tool_names": gold_names,
            })

    return augmented_by_n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-seen",
        type=Path,
        default=Path("data/custom_vi/test_seen.jsonl"),
        help="Path to data/custom_vi/test_seen.jsonl",
    )
    parser.add_argument(
        "--benchmark-pool",
        type=Path,
        default=Path("data/benchmark_vi/tool_pool.json"),
        help="Path to data/benchmark_vi/tool_pool.json",
    )
    parser.add_argument(
        "--custom-tools",
        type=Path,
        default=Path("data/custom_vi/tools.json"),
        help="Path to data/custom_vi/tools.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stress_test"),
        help="Output root directory for stress test data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-queries", type=int, default=200)
    args = parser.parse_args()

    assert args.test_seen.is_file(), f"Not found: {args.test_seen}"
    assert args.benchmark_pool.is_file(), f"Not found: {args.benchmark_pool}"

    print(f"[*] Reading test_seen from {args.test_seen}...")
    with args.test_seen.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[*] Sampling {args.n_queries} stratified anchors (seed={args.seed})...")
    anchors = sample_queries(records, n_queries=args.n_queries, seed=args.seed)
    pos_count = sum(1 for a in anchors if a.get("function_calls"))
    neg_count = sum(1 for a in anchors if not a.get("function_calls"))
    print(f"    -> Selected {len(anchors)} anchors ({pos_count} positive, {neg_count} negative)")

    print("[*] Loading tool pool...")
    pool = load_tool_pool(args.benchmark_pool, args.custom_tools)
    print(f"    -> Unified tool pool: {len(pool)} tools")

    n_values = (3, 10, 50, 100, 500, 1000)
    print(f"[*] Generating nested haystacks for N = {n_values}...")
    augmented_by_n = build_stress_instances(anchors, pool, n_values=n_values, seed=args.seed)

    # Save anchors
    args.output_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir = args.output_dir / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    anchors_path = args.output_dir / "anchors.jsonl"
    with anchors_path.open("w", encoding="utf-8") as f:
        for a in anchors:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(f"[+] Saved anchors to: {anchors_path}")

    # Save augmented instances
    for n, items in augmented_by_n.items():
        out_file = augmented_dir / f"random_N{n}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        file_size_kb = out_file.stat().st_size / 1024
        print(f"    [+] Saved N={n:<4}: {len(items)} instances ({file_size_kb:.1f} KB) -> {out_file.name}")

    print("\n✅ Phase 7 Stress Test dataset generation complete!")


if __name__ == "__main__":
    main()
