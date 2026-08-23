"""Đánh giá Bi-Encoder + hiệu chỉnh ngưỡng (§Phase 2.5 method2_plan).

Metric truy hồi (khớp định nghĩa trong `docs/evaluation.md` §Retrieval):

- **Micro Recall@K** — trên toàn bộ unique gold tool của từng sample.
- **Full Recall@K** — tỉ lệ sample có *tất cả* gold tool nằm trong top-K.
- **MRR** — reciprocal rank của gold tool xuất hiện sớm nhất.
- **NDCG@10**.

Hai chế độ `scope`:

- `candidates` — chỉ xếp hạng trong candidate pool của sample (đúng thiết lập
  benchmark; gate §Phase 2 đo ở đây).
- `pool` — xếp hạng trên toàn bộ ~4,4k tool (thiết lập stress test).

Hiệu chỉnh ngưỡng **chỉ được chạy trên val**; `thresholds.json` phải freeze
trước khi chạy test, nếu tune trên test thì kết quả vô hiệu.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.models.biencoder.retrieve import RetrievalThresholds, ToolRetriever, select_tools
from src.models.sources import load_jsonl

DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass
class SampleRanking:
    """Ranking đã tính sẵn cho một sample — dùng lại khi quét lưới ngưỡng."""

    sample_id: str
    query: str
    gold: list[str]
    ranked: list[tuple[str, float]]
    source: str
    source_key: str
    tool_split: str

    @property
    def is_negative(self) -> bool:
        return not self.gold


def group_rows_by_sample(path: str | Path) -> list[dict[str, Any]]:
    """Pairs file có 1 dòng/positive; gộp lại thành 1 dòng/sample."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in load_jsonl(path):
        key = str(row.get("sample_id") or row["query"])
        if key not in grouped:
            grouped[key] = {
                "sample_id": key,
                "query": row["query"],
                "gold": [],
                "candidates": row.get("candidates") or [],
                "source": row.get("source", ""),
                "source_key": row.get("source_key", ""),
                "tool_split": row.get("tool_split", "mixed"),
            }
            order.append(key)
        if row.get("positive"):
            grouped[key]["gold"] = list(row.get("all_gold") or [row["positive"]])
    return [grouped[k] for k in order]


def rank_samples(
    retriever: ToolRetriever,
    samples: Sequence[dict[str, Any]],
    scope: str = "candidates",
    batch_size: int = 64,
) -> list[SampleRanking]:
    queries = [s["query"] for s in samples]
    embeddings = retriever.encode_queries(queries, batch_size=batch_size)
    rankings: list[SampleRanking] = []
    for i, sample in enumerate(samples):
        candidates = sample.get("candidates") if scope == "candidates" else None
        if scope == "candidates" and not candidates:
            candidates = None
        ranked = retriever.score(embeddings[i], candidates)
        rankings.append(
            SampleRanking(
                sample_id=sample["sample_id"],
                query=sample["query"],
                gold=list(sample.get("gold") or []),
                ranked=ranked,
                source=sample.get("source", ""),
                source_key=sample.get("source_key", ""),
                tool_split=sample.get("tool_split", "mixed"),
            )
        )
    return rankings


# ------------------------------------------------------------------ metrics


def retrieval_metrics(
    rankings: Sequence[SampleRanking],
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    positives = [r for r in rankings if not r.is_negative]
    if not positives:
        return {"n_positive_samples": 0}

    hits_at_k: dict[int, int] = defaultdict(int)
    gold_total = 0
    full_hits: dict[int, int] = defaultdict(int)
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []

    for ranking in positives:
        names = [name for name, _ in ranking.ranked]
        gold = set(ranking.gold)
        gold_total += len(gold)
        positions = {name: i for i, name in enumerate(names)}

        first_rank = min((positions[g] for g in gold if g in positions), default=None)
        reciprocal_ranks.append(1.0 / (first_rank + 1) if first_rank is not None else 0.0)

        for k in k_values:
            top_k = set(names[:k])
            hits_at_k[k] += len(gold & top_k)
            full_hits[k] += int(gold <= top_k)

        dcg = sum(
            1.0 / math.log2(positions[g] + 2) for g in gold if positions.get(g, 99) < 10
        )
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 10)))
        ndcg_scores.append(dcg / idcg if idcg else 0.0)

    n = len(positives)
    metrics: dict[str, Any] = {
        "n_positive_samples": n,
        "n_gold_tools": gold_total,
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "ndcg@10": round(sum(ndcg_scores) / n, 4),
    }
    for k in k_values:
        metrics[f"micro_recall@{k}"] = round(hits_at_k[k] / gold_total, 4) if gold_total else 0.0
        metrics[f"full_recall@{k}"] = round(full_hits[k] / n, 4)
    return metrics


def metrics_by_slice(
    rankings: Sequence[SampleRanking],
    field: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    buckets: dict[str, list[SampleRanking]] = defaultdict(list)
    for ranking in rankings:
        buckets[getattr(ranking, field)].append(ranking)
    return {key: retrieval_metrics(rows, k_values) for key, rows in sorted(buckets.items())}


# -------------------------------------------------------------- calibration


def calibrate_tau(
    rankings: Sequence[SampleRanking],
    grid: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Chọn `τ` maximize Macro-F1 giữa hai lớp {call, no_tool_call}."""
    best_tau, best_score, best_metrics = grid[0], -1.0, {}
    top_scores = [(r.ranked[0][1] if r.ranked else -1.0, r.is_negative) for r in rankings]

    for tau in grid:
        tp = fp = fn = tn = 0
        for score, is_negative in top_scores:
            predicted_call = score >= tau
            if is_negative:
                fp += int(predicted_call)
                tn += int(not predicted_call)
            else:
                tp += int(predicted_call)
                fn += int(not predicted_call)
        macro, detail = _macro_f1(tp, fp, fn, tn)
        if macro > best_score:
            best_tau, best_score, best_metrics = tau, macro, detail
    return best_tau, best_metrics


def _macro_f1(tp: int, fp: int, fn: int, tn: int) -> tuple[float, dict[str, float]]:
    def f1(p_num: int, p_den: int, r_num: int, r_den: int) -> float:
        precision = p_num / p_den if p_den else 0.0
        recall = r_num / r_den if r_den else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    f1_call = f1(tp, tp + fp, tp, tp + fn)
    f1_no_call = f1(tn, tn + fn, tn, tn + fp)
    macro = (f1_call + f1_no_call) / 2
    return macro, {
        "macro_f1": round(macro, 4),
        "f1_call": round(f1_call, 4),
        "f1_no_call": round(f1_no_call, 4),
        "negative_recall": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "call_recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
    }


def calibrate_call_selection(
    rankings: Sequence[SampleRanking],
    tau: float,
    k_max: int,
    absolute_grid: Sequence[float],
    gap_grid: Sequence[float],
) -> dict[str, Any]:
    """So chiến lược ngưỡng tuyệt đối với gap-based (Q4 §9)."""
    positives = [r for r in rankings if not r.is_negative and r.ranked]

    best_absolute = _best_strategy(
        positives, tau, k_max, absolute_grid, strategy="absolute"
    )
    best_gap = _best_strategy(positives, tau, k_max, gap_grid, strategy="gap")
    winner = "absolute" if best_absolute["f1"] >= best_gap["f1"] else "gap"
    return {"absolute": best_absolute, "gap": best_gap, "winner": winner}


def _best_strategy(
    positives: Sequence[SampleRanking],
    tau: float,
    k_max: int,
    grid: Sequence[float],
    strategy: str,
) -> dict[str, Any]:
    best = {"value": grid[0] if grid else 0.0, "f1": -1.0, "tool_set_accuracy": 0.0}
    for value in grid:
        thresholds = RetrievalThresholds(
            tau=tau,
            tau_call=value if strategy == "absolute" else 1.0,
            gap_delta=value if strategy == "gap" else 0.0,
            k_max=k_max,
            strategy=strategy,
        )
        tp = fp = fn = exact = 0
        for ranking in positives:
            selected = select_tools(ranking.ranked, thresholds)
            gold = set(ranking.gold)
            predicted = set(selected)
            tp += len(gold & predicted)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
            exact += int(gold == predicted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best = {
                "value": round(value, 4),
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "tool_set_accuracy": round(exact / len(positives), 4) if positives else 0.0,
            }
    return best


def _grid(spec: dict[str, Any]) -> list[float]:
    start = float(spec.get("start", 0.0))
    stop = float(spec.get("stop", 1.0))
    step = float(spec.get("step", 0.01))
    n = int(round((stop - start) / step)) + 1
    return [round(start + i * step, 6) for i in range(max(n, 1))]


def calibrate_thresholds(
    retriever: ToolRetriever,
    val_path: str | Path,
    config: dict[str, Any],
    scope: str = "candidates",
) -> RetrievalThresholds:
    samples = group_rows_by_sample(val_path)
    rankings = rank_samples(retriever, samples, scope=scope)

    tau, tau_metrics = calibrate_tau(rankings, _grid(config.get("tau_grid", {})))
    k_max = int(config.get("k_max", 3))
    selection = calibrate_call_selection(
        rankings,
        tau=tau,
        k_max=k_max,
        absolute_grid=_grid(config.get("tau_call_grid", {})),
        gap_grid=_grid(config.get("gap_delta_grid", {"start": 0.0, "stop": 0.3, "step": 0.01})),
    )
    strategy = str(config.get("strategy", "absolute"))
    if strategy == "auto":
        strategy = selection["winner"]

    return RetrievalThresholds(
        tau=tau,
        tau_call=selection["absolute"]["value"],
        gap_delta=selection["gap"]["value"],
        k_max=k_max,
        strategy=strategy,
        calibrated_on=str(val_path),
        metrics={
            "abstention": tau_metrics,
            "call_selection": selection,
            "retrieval": retrieval_metrics(rankings),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate / calibrate Bi-Encoder")
    parser.add_argument("command", choices=["evaluate", "calibrate"])
    parser.add_argument("--config", type=Path, default=Path("configs/method2/biencoder.yaml"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--scope", choices=["candidates", "pool"], default="candidates")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import yaml

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    index_cfg = raw.get("index", {})
    retriever = ToolRetriever.from_paths(
        model_path=args.model,
        embeddings_path=index_cfg.get("embeddings_path", "data/method2/index/tool_embeddings.npy"),
        tool_ids_path=index_cfg.get("tool_ids_path", "data/method2/index/tool_ids.json"),
        max_seq_length=int(raw.get("model", {}).get("max_seq_length", 192)),
    )

    if args.command == "calibrate":
        thresholds = calibrate_thresholds(
            retriever, args.pairs, raw.get("thresholds", {}), scope=args.scope
        )
        output = args.output or Path(
            raw.get("thresholds", {}).get(
                "output_path", "artifacts/method2/biencoder/run01/thresholds.json"
            )
        )
        thresholds.save(output)
        print(json.dumps(thresholds.__dict__, ensure_ascii=False, indent=2))
        return

    samples = group_rows_by_sample(args.pairs)
    rankings = rank_samples(retriever, samples, scope=args.scope)
    report = {
        "scope": args.scope,
        "overall": retrieval_metrics(rankings, raw.get("evaluate", {}).get("k_values", DEFAULT_K_VALUES)),
        "by_source_key": metrics_by_slice(rankings, "source_key"),
        "by_tool_split": metrics_by_slice(rankings, "tool_split"),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
