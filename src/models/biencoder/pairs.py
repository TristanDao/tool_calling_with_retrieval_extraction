"""Sinh training pairs cho Bi-Encoder (§1.3 method2_plan).

Mỗi positive `(query, gold_tool)` được ghép 4 hard negative theo thứ tự ưu tiên:

1. **Candidate trong chính sample** — distractor do benchmark chọn sẵn.
2. **Cùng `feature_group`** — với CustomTools-VI, 10 nhóm × 4 tool nên tool cùng
   nhóm (`vi_search_restaurants` vs `vi_order_food`) là negative khó đúng nghĩa.
3. **BM25 top-k** trên toàn pool, loại positive — nguồn duy nhất cho glaive/xLAM
   vì phần lớn tool ở đó chưa có `feature_group` thật.

Sample no-call (`function_calls: []`) không sinh positive nhưng vẫn được ghi lại
với `positive: null` để hiệu chỉnh ngưỡng abstention `τ` ở §5.1.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.biencoder.tool_pool import load_tool_pool
from src.models.sources import (
    DEFAULT_SOURCES,
    DecontaminationIndex,
    SourceSpec,
    build_decontamination_index,
    iter_samples,
    load_or_build_decontamination,
    normalize_query_key,
    pairwise_overlap,
    write_jsonl,
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class PairsConfig:
    tool_pool_path: Path = Path("data/method2/tool_pool.json")
    output_dir: Path = Path("data/method2/biencoder")
    decontamination_path: Path = Path("data/method2/decontamination.json")
    stats_path: Path = Path("data/method2/biencoder/pairs_stats.json")
    n_hard_negatives: int = 4
    bm25_top_k: int = 20
    use_bm25: bool = True
    seed: int = 42
    limit_per_source: int | None = None
    progress_every: int = 5000
    #: True chỉ khi chạy như bước preprocessing tường minh (--rebuild-decontamination).
    allow_build_decontamination: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PairsConfig":
        defaults = cls()
        return cls(
            tool_pool_path=Path(raw.get("tool_pool_path", defaults.tool_pool_path)),
            output_dir=Path(raw.get("output_dir", defaults.output_dir)),
            decontamination_path=Path(
                raw.get("decontamination_path", defaults.decontamination_path)
            ),
            stats_path=Path(raw.get("stats_path", defaults.stats_path)),
            n_hard_negatives=int(raw.get("n_hard_negatives", defaults.n_hard_negatives)),
            bm25_top_k=int(raw.get("bm25_top_k", defaults.bm25_top_k)),
            use_bm25=bool(raw.get("use_bm25", defaults.use_bm25)),
            seed=int(raw.get("seed", defaults.seed)),
            limit_per_source=raw.get("limit_per_source"),
            progress_every=int(raw.get("progress_every", defaults.progress_every)),
        )


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Miner:
    """BM25 trên document text của tool pool. Lazy import `rank_bm25`."""

    def __init__(self, pool: dict[str, dict[str, Any]]) -> None:
        from rank_bm25 import BM25Okapi

        self.names = list(pool)
        corpus = [tokenize(pool[name]["doc_text"]) for name in self.names]
        self.index = BM25Okapi(corpus)

    def top_k(self, query: str, k: int) -> list[str]:
        scores = self.index.get_scores(tokenize(query))
        if k >= len(self.names):
            order = range(len(self.names))
        else:
            import numpy as np

            order = np.argpartition(-scores, k)[:k]
            order = sorted(order, key=lambda i: -scores[i])
        return [self.names[i] for i in order]


def build_pairs(
    config: PairsConfig,
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pool = load_tool_pool(config.tool_pool_path)
    by_group: dict[str, list[str]] = defaultdict(list)
    for name, tool in pool.items():
        by_group[tool.get("feature_group", "Khác")].append(name)

    miner = BM25Miner(pool) if config.use_bm25 else None
    rng = random.Random(config.seed)

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negative_sources: Counter = Counter()
    stats = {
        "n_positive_pairs": 0,
        "n_negative_samples": 0,
        "gold_not_in_pool": 0,
        "queries_per_split": Counter(),
        "positives_per_source": Counter(),
    }
    #: Kiểm tra tool unseen không lọt vào positive của split train (§1, Phase 1).
    positives_in_train: set[str] = set()
    queries_seen: dict[str, set[str]] = defaultdict(set)

    decontamination = load_or_build_decontamination(
        config.decontamination_path,
        specs,
        config.limit_per_source,
        allow_build=config.allow_build_decontamination,
    )

    processed = 0
    for sample in iter_samples(
        specs, limit_per_source=config.limit_per_source, decontamination=decontamination
    ):
        processed += 1
        if config.progress_every and processed % config.progress_every == 0:
            print(f"[pairs] {processed} sample, {stats['n_positive_pairs']} positive", flush=True)
        query = str(sample.get("query", "")).strip()
        if not query:
            continue
        split = sample["_split"]
        stats["queries_per_split"][split] += 1
        candidates = [t.get("name") for t in (sample.get("tools") or []) if t.get("name")]
        gold_names = [c["name"] for c in (sample.get("function_calls") or []) if c.get("name")]
        unique_gold = list(dict.fromkeys(gold_names))

        queries_seen[split].add(normalize_query_key(query))

        if not unique_gold:
            rows_by_split[split].append(
                {
                    "sample_id": sample.get("id"),
                    "query": query,
                    "positive": None,
                    "negatives": [],
                    "candidates": candidates,
                    "source": sample.get("source") or sample["_source_key"],
                    "source_key": sample["_source_key"],
                    "split": split,
                    "tool_split": sample["_tool_split"],
                }
            )
            stats["n_negative_samples"] += 1
            continue

        for gold in unique_gold:
            if gold not in pool:
                stats["gold_not_in_pool"] += 1
                continue
            negatives, origins = _pick_negatives(
                gold=gold,
                all_gold=set(unique_gold),
                candidates=candidates,
                pool=pool,
                by_group=by_group,
                miner=miner,
                query=query,
                config=config,
                rng=rng,
            )
            negative_sources.update(origins)
            rows_by_split[split].append(
                {
                    "sample_id": sample.get("id"),
                    "query": query,
                    "positive": gold,
                    "negatives": negatives,
                    "candidates": candidates,
                    "all_gold": unique_gold,
                    "source": sample.get("source") or sample["_source_key"],
                    "source_key": sample["_source_key"],
                    "split": split,
                    "tool_split": sample["_tool_split"],
                }
            )
            stats["n_positive_pairs"] += 1
            stats["positives_per_source"][sample["_source_key"]] += 1
            if split == "train":
                positives_in_train.add(gold)

    stats["decontamination"] = decontamination.report()
    stats["split_overlap_after"] = pairwise_overlap(queries_seen)
    stats["unique_queries_per_split"] = {k: len(v) for k, v in sorted(queries_seen.items())}

    unseen_tools = {
        name
        for name, tool in pool.items()
        if "custom_vi" in tool.get("sources", [])
        and "train" not in tool.get("gold_in_splits", [])
    }
    stats["negative_sources"] = dict(negative_sources)
    stats["queries_per_split"] = dict(stats["queries_per_split"])
    stats["positives_per_source"] = dict(stats["positives_per_source"])
    stats["n_unseen_tools_in_pool"] = len(unseen_tools)
    stats["unseen_tools_leaked_into_train_positives"] = sorted(
        unseen_tools & positives_in_train
    )
    return rows_by_split, stats


def _pick_negatives(
    gold: str,
    all_gold: set[str],
    candidates: list[str],
    pool: dict[str, dict[str, Any]],
    by_group: dict[str, list[str]],
    miner: BM25Miner | None,
    query: str,
    config: PairsConfig,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    picked: list[str] = []
    origins: list[str] = []

    def _add(names: list[str], origin: str) -> None:
        for name in names:
            if len(picked) >= config.n_hard_negatives:
                return
            if name in all_gold or name in picked or name not in pool:
                continue
            picked.append(name)
            origins.append(origin)

    _add([c for c in candidates if c not in all_gold], "in_sample_candidate")

    group = pool[gold].get("feature_group", "Khác")
    if group != "Khác":
        same_group = [n for n in by_group[group] if n not in all_gold]
        rng.shuffle(same_group)
        _add(same_group, "feature_group")

    if miner is not None and len(picked) < config.n_hard_negatives:
        # Bỏ top-1 BM25: thường chính là gold hoặc biến thể của nó.
        _add(miner.top_k(query, config.bm25_top_k)[1:], "bm25")

    if len(picked) < config.n_hard_negatives:
        pool_names = list(pool)
        _add(rng.sample(pool_names, min(len(pool_names), 50)), "random")

    return picked, origins


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bi-Encoder training pairs")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/biencoder.yaml"))
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument(
        "--rebuild-decontamination",
        action="store_true",
        help="Build lại decontamination index nếu thiếu (bước preprocessing)",
    )
    args = parser.parse_args()

    raw: dict[str, Any] = {}
    if args.config.exists():
        import yaml

        raw = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("pairs", {})
    config = PairsConfig.from_dict(raw)
    if args.limit_per_source:
        config.limit_per_source = args.limit_per_source
    if args.no_bm25:
        config.use_bm25 = False
    if args.rebuild_decontamination:
        config.allow_build_decontamination = True

    rows_by_split, stats = build_pairs(config)

    for split, rows in rows_by_split.items():
        path = config.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        print(f"[pairs] {split}: {len(rows)} dòng → {path}")

    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[pairs] positive={stats['n_positive_pairs']} negative={stats['n_negative_samples']}")

    decontamination = stats["decontamination"]
    print(
        f"[pairs] decontamination: {decontamination['n_overlapping_queries']} query trùng split "
        f"→ loại {decontamination['rows_dropped_total']} sample "
        f"({decontamination['rows_dropped_by_transition']})"
    )

    overlap = stats["split_overlap_after"]
    print(f"[pairs] overlap sau decontamination: {overlap}")
    dirty = {pair: n for pair, n in overlap.items() if n}
    if dirty:
        raise SystemExit(f"[pairs] LỖI: còn query dùng chung giữa các split: {dirty}")
    print("[pairs] OK: train ∩ val = train ∩ test = val ∩ test = 0 theo normalized query")

    leaked = stats["unseen_tools_leaked_into_train_positives"]
    if leaked:
        raise SystemExit(f"[pairs] LỖI: tool unseen lọt vào positive của train: {leaked}")
    print("[pairs] OK: không có tool unseen nào làm positive trong split train")


if __name__ == "__main__":
    main()
