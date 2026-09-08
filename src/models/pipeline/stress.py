"""Phase 7 — Stress test: Method 2 khi số lượng tool tăng dần.

Điểm cần chứng minh (§Phase 7 `method2_plan`, §10.2 `experimental_plan`):
latency của Method 2 gần như **phẳng** theo N, trong khi SLM và API tăng tuyến
tính vì cả N tool phải nằm trong context. Bi-Encoder chỉ tốn thêm một phép nhân
`1×d · d×N`; Cross-Encoder không đổi vì chỉ chạy trên tool ĐÃ chọn.

Ba quyết định thiết kế — chúng quyết định đường cong đọc được hay không:

1. **Haystack lồng nhau.** Mỗi query có đúng MỘT thứ tự distractor cố định; N
   chỉ là độ dài prefix, nên haystack N=10 chứa trọn haystack N=3. Bốc lại ngẫu
   nhiên cho từng N sẽ trộn hai biến "nhiều distractor hơn" và "distractor khác
   đi"; dao động thu được không quy được về N.

2. **Gold luôn nằm trong haystack**: `haystack = gold ∪ prefix(distractor)`, dài
   đúng N. Nếu gold bị bốc rơi thì accuracy tụt vì không có đáp án, chứ không
   phải vì retrieval khó lên.

3. **Schema gold lấy từ chính sample, schema distractor lấy từ pool.** Pool gộp
   các biến thể trùng tên (`n_variants`) nên schema pool có thể lệch với schema
   mà argument gold được viết theo. Lấy nhầm là tự chế thêm lỗi extraction
   không liên quan gì tới N.

Về `t_retrieve`: `ToolRetriever.score` gom hàng từ index dùng chung 4.464 tool
bằng một list-comprehension theo tên, nên phần này tăng tuyến tính theo N vì lý
do Python thuần tuý — hệ thống thật sẽ dựng index đúng N tool và chỉ còn một
phép matmul. Số `t_retrieve` ở đây là **chặn trên**; khẳng định "phẳng" đọc ở
`t_cross_encode` (thành phần chi phối) và ở `total`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.models.pipeline.method2 import (
    MODE_PIPELINE,
    Method2Config,
    Method2Pipeline,
    StageTimings,
    summarize_latency,
)
from src.models.sources import load_jsonl, write_jsonl

DISTRACTOR_RANDOM = "random"
DISTRACTOR_SAME_DOMAIN = "same_domain"

#: Nhãn mặc định của tool chưa được phân loại. 4.424/4.464 tool đang ở đây, nên
#: `same_domain` chỉ có nghĩa với 40 tool CustomTools-VI (10 nhóm × 4 tool).
UNLABELED_GROUP = "Khác"


@dataclass
class StressConfig:
    n_values: tuple[int, ...] = (3, 10, 50, 100, 500, 1000)
    distractor_modes: tuple[str, ...] = (DISTRACTOR_RANDOM,)
    n_queries: int = 200
    seed: int = 42
    tool_pool_path: Path = Path("data/method2/tool_pool.json")
    gold_path: Path = Path("data/custom_vi/v1/test_seen.jsonl")
    output_dir: Path = Path("results/method2/stress")
    allow_mixed_domain: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StressConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        defaults = cls()
        return cls(
            n_values=tuple(int(n) for n in raw.get("n_values", defaults.n_values)),
            distractor_modes=tuple(
                str(m) for m in raw.get("distractor_modes", defaults.distractor_modes)
            ),
            n_queries=int(raw.get("n_queries", defaults.n_queries)),
            seed=int(raw.get("seed", defaults.seed)),
            tool_pool_path=Path(raw.get("tool_pool_path", defaults.tool_pool_path)),
            gold_path=Path(raw.get("gold_path", defaults.gold_path)),
            output_dir=Path(raw.get("output_dir", defaults.output_dir)),
            allow_mixed_domain=bool(raw.get("allow_mixed_domain", False)),
        )


@dataclass
class Haystack:
    """Tập tool trình cho pipeline ở một mức N, kèm số liệu để kiểm chứng."""

    tools: list[dict[str, Any]]
    gold_names: list[str]
    n_distractors: int
    #: Tỉ lệ distractor thật sự cùng `feature_group` với gold. Chỉ có nghĩa ở
    #: mode `same_domain`; `None` ở mode `random`.
    purity: float | None = None
    truncated: bool = False


# --------------------------------------------------------------- lấy mẫu query


def sample_queries(
    samples: Iterable[dict[str, Any]],
    n_queries: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Bốc `n_queries` sample, **phân tầng theo positive/negative**.

    Bốc phẳng thì tỉ lệ negative dao động theo seed, mà negative chính là mẫu số
    của negative recall — số đó cần một mẫu số ổn định để so được giữa các N.
    Trả về theo đúng thứ tự xuất hiện trong file để mọi run đọc giống nhau.

    `list()` ngay đầu là bắt buộc, không phải phòng xa: `sources.load_jsonl`
    là generator, mà hàm này duyệt `samples` ba lượt — nhận thẳng generator
    thì lượt hai trở đi rỗng và stress test chạy trên 0 query mà không báo lỗi.
    """
    samples = list(samples)
    if n_queries <= 0 or n_queries > len(samples):
        raise ValueError("n_queries must be positive and no larger than the dataset")
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


# ------------------------------------------------------------- dựng haystack


def reference_group(sample: dict[str, Any], groups: dict[str, str]) -> str | None:
    """`feature_group` để đo `same_domain`.

    Sample positive lấy theo tool gold. Sample negative không có gold, nhưng nó
    được dựng ra để đối chọi với một nhóm cụ thể, nên lấy nhóm phổ biến nhất
    trong candidate pool của chính nó. `Khác` bị loại vì nó không phải một nhóm
    mà là chỗ chứa những tool chưa được phân loại.
    """
    calls = sample.get("function_calls") or []
    names = [c["name"] for c in calls if c.get("name")]
    if not names:
        names = [t["name"] for t in (sample.get("tools") or []) if t.get("name")]
    labels = [
        groups[name]
        for name in names
        if groups.get(name) and groups[name] != UNLABELED_GROUP
    ]
    if not labels:
        return None
    return Counter(labels).most_common(1)[0][0]


def distractor_order(
    sample: dict[str, Any],
    gold_names: Sequence[str],
    pool_names: Sequence[str],
    groups: dict[str, str],
    mode: str,
    seed: int,
) -> list[str]:
    """Thứ tự distractor cố định cho một query — prefix của nó là haystack.

    `same_domain` = xếp tool cùng nhóm lên trước rồi mới tới phần còn lại. Sort
    ổn định nên thứ tự ngẫu nhiên bên trong mỗi rổ được giữ nguyên, và tính lồng
    nhau giữa các N vẫn đúng. Cách này cũng tự xử lý việc mỗi nhóm chỉ có 4 tool:
    hết tool cùng nhóm thì phần dư được lấp bằng tool ngẫu nhiên, và `purity`
    ghi lại chính xác mức độ suy biến đó thay vì im lặng.
    """
    excluded = set(gold_names)
    ordered = [n for n in pool_names if n not in excluded]
    random.Random(f"{seed}:{sample['id']}:{mode}").shuffle(ordered)
    if mode == DISTRACTOR_SAME_DOMAIN:
        group = reference_group(sample, groups)
        if group is not None:
            ordered.sort(key=lambda name: 0 if groups.get(name) == group else 1)
    return ordered


def build_haystack(
    sample: dict[str, Any],
    n: int,
    ordered_distractors: Sequence[str],
    pool: dict[str, dict[str, Any]],
    groups: dict[str, str],
    mode: str,
) -> Haystack:
    sample_schemas = {t["name"]: t for t in (sample.get("tools") or []) if t.get("name")}
    gold_names = list(
        dict.fromkeys(
            c["name"] for c in (sample.get("function_calls") or []) if c.get("name")
        )
    )
    truncated = len(gold_names) > n
    kept_gold = gold_names[:n] if truncated else gold_names
    distractors = list(ordered_distractors[: max(n - len(kept_gold), 0)])

    tools = [sample_schemas.get(name) or pool[name] for name in kept_gold]
    tools += [pool[name] for name in distractors]

    purity = None
    if mode == DISTRACTOR_SAME_DOMAIN and distractors:
        group = reference_group(sample, groups)
        if group is not None:
            same = sum(1 for name in distractors if groups.get(name) == group)
            purity = same / len(distractors)
    return Haystack(
        tools=tools,
        gold_names=gold_names,
        n_distractors=len(distractors),
        purity=purity,
        truncated=truncated,
    )


# ------------------------------------------------------------------- chạy test


def _metrics(
    samples: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Đo bằng đúng module của `src/evaluation` — không tự tính lại metric.

    Gold record dựng từ sample GỐC (10 tool của nó), không phải từ haystack:
    phía gold phải đứng yên khi N đổi, nếu không thì normalizer đọc schema khác
    nhau ở mỗi N và ArgEM mất tính so sánh.
    """
    from src.evaluation.config import NormalizationConfig
    from src.evaluation.detection_metrics import compute_detection_metrics
    from src.evaluation.extraction_metrics import compute_extraction_metrics
    from src.evaluation.io import align_predictions, parse_gold_record, parse_prediction_record
    from src.evaluation.metric_utils import safe_divide
    from src.evaluation.normalization import ArgumentNormalizer
    from src.evaluation.retrieval_metrics import compute_retrieval_metrics
    from src.evaluation.selection_metrics import compute_selection_metrics
    from src.evaluation.end_to_end_metrics import compute_end_to_end_metrics
    from src.evaluation.schema_validation import aggregate_schema_validity, validate_prediction_schema

    gold = [parse_gold_record(s) for s in samples]
    aligned = align_predictions(gold, [parse_prediction_record(p) for p in predictions])

    selection = compute_selection_metrics(gold, aligned)
    retrieval = compute_retrieval_metrics(gold, aligned, ks=(1, 3, 5))
    detection = compute_detection_metrics(gold, aligned)
    extraction, _ = compute_extraction_metrics(
        gold, aligned, ArgumentNormalizer(NormalizationConfig())
    )
    end_to_end, _ = compute_end_to_end_metrics(gold, aligned, ArgumentNormalizer(NormalizationConfig()))
    strict, _ = compute_end_to_end_metrics(gold, aligned, ArgumentNormalizer(NormalizationConfig.strict()))
    schema = aggregate_schema_validity([validate_prediction_schema(g, p) for g, p in zip(gold, aligned, strict=True)])
    return {
        "n_fcem_positive": end_to_end["n_fcem_positive"],
        "strict_arga": strict["n_fcem_positive"],
        "overall_success": end_to_end["overall_success"],
        "json_validity": schema["prediction_parse_validity"],
        "schema_validity": schema["call_schema_validity"],
        "cost_usd_per_1k": None,
        "tool_set_accuracy": selection["tool_set_accuracy_positive"],
        "recall_at_1": retrieval["recall_at_1"],
        "recall_at_3": retrieval["recall_at_3"],
        "mrr": retrieval["mrr"],
        "negative_recall": safe_divide(
            detection["true_negative"], detection["negative_count"]
        ),
        "arg_em_given_correct_tool": extraction["normalized_arg_em_given_correct_tool"],
        "positive_count": selection["positive_count"],
        "negative_count": detection["negative_count"],
    }


def run_stress(
    pipeline: Method2Pipeline,
    config: StressConfig,
    samples: Sequence[dict[str, Any]] | None = None,
    pool: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from src.models.biencoder.tool_pool import load_tool_pool

    if samples is None:
        samples = sample_queries(
            load_jsonl(config.gold_path), config.n_queries, config.seed
        )
    if pool is None:
        pool = load_tool_pool(config.tool_pool_path)
    groups = {name: (tool.get("feature_group") or UNLABELED_GROUP) for name, tool in pool.items()}
    pool_names = sorted(pool)
    if not samples or len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Stress anchors must be nonempty and have unique IDs")
    if any(n <= 0 or n > len(pool) for n in config.n_values):
        raise ValueError("N must be positive and no larger than the tool pool")
    if any(mode not in (DISTRACTOR_RANDOM, DISTRACTOR_SAME_DOMAIN) for mode in config.distractor_modes):
        raise ValueError("Unknown distractor mode")
    for sample in samples:
        gold_names = {c["name"] for c in sample.get("function_calls", [])}
        if not gold_names <= set(pool) or len(gold_names) > min(config.n_values):
            raise ValueError("Every anchor's gold tools must fit every haystack")
        if DISTRACTOR_SAME_DOMAIN in config.distractor_modes and not config.allow_mixed_domain:
            group = reference_group(sample, groups)
            same = sum(name not in gold_names and groups[name] == group for name in pool)
            if group is None or same < max(config.n_values) - len(gold_names):
                raise ValueError("Pure same_domain infeasible; enrich groups or explicitly label a mixed-domain extension")
    write_jsonl(config.output_dir / "anchors.jsonl", samples)

    # Stress test đo retrieval TRONG haystack đã dựng; ép `candidates` bất kể
    # config, vì `pool` sẽ xếp hạng trên toàn bộ 4.464 tool và N mất tác dụng.
    previous_scope = pipeline.retrieval_scope
    pipeline.retrieval_scope = "candidates"

    rows: list[dict[str, Any]] = []
    try:
        for mode in config.distractor_modes:
            orders = {
                s["id"]: distractor_order(
                    s,
                    [c["name"] for c in (s.get("function_calls") or []) if c.get("name")],
                    pool_names,
                    groups,
                    mode,
                    config.seed,
                )
                for s in samples
            }
            for n in config.n_values:
                predictions: list[dict[str, Any]] = []
                raw_predictions: list[dict[str, Any]] = []
                presented: list[dict[str, Any]] = []
                timings: list[StageTimings] = []
                purities: list[float] = []
                truncated = 0
                for sample in samples:
                    haystack = build_haystack(
                        sample, n, orders[sample["id"]], pool, groups, mode
                    )
                    truncated += int(haystack.truncated)
                    if haystack.purity is not None:
                        purities.append(haystack.purity)
                    outcome = pipeline.run_sample(
                        {**sample, "tools": haystack.tools}, MODE_PIPELINE
                    )
                    predictions.append(outcome["prediction"])
                    raw_predictions.append(outcome.get("raw", {}))
                    presented.append({**sample, "tools": haystack.tools})
                    timings.append(outcome["timings"])

                write_jsonl(
                    config.output_dir / f"predictions_{mode}_n{n}.jsonl", predictions
                )
                write_jsonl(config.output_dir / f"raw_{mode}_n{n}.jsonl", raw_predictions)
                rows.append(
                    {
                        "mode": mode,
                        "distractor_label": "same_domain_first_mixed" if mode == DISTRACTOR_SAME_DOMAIN and config.allow_mixed_domain else mode,
                        "n": n,
                        "n_queries": len(samples),
                        "truncated_gold": truncated,
                        "same_domain_purity": (
                            round(sum(purities) / len(purities), 4) if purities else None
                        ),
                        "latency": summarize_latency(timings),
                        **_metrics(presented, predictions),
                    }
                )
    finally:
        pipeline.retrieval_scope = previous_scope

    report = {
        "config": {
            "n_values": list(config.n_values),
            "distractor_modes": list(config.distractor_modes),
            "n_queries": config.n_queries,
            "seed": config.seed,
            "gold_path": str(config.gold_path),
            "tool_pool_path": str(config.tool_pool_path),
            "pool_size": len(pool),
        },
        "rows": rows,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "stress_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.output_dir / "stress_summary.md").write_text(
        summary_markdown(report), encoding="utf-8"
    )
    return report


# ----------------------------------------------------------------- báo cáo


def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stress test Method 2 — scaling theo số lượng tool",
        "",
        f"Pool: {report['config']['pool_size']} tool · "
        f"{report['config']['n_queries']} query · seed {report['config']['seed']}",
        "",
        "| Mode | N | Tool Set Acc | R@1 | Neg Recall | ArgEM\\|tool | N-FCEM positive | Strict ArgA | "
        "t_embed p50 | t_retrieve p50 | t_cross p50 | total p50 | total p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        latency = row.get("latency") or {}

        def p(stage: str, key: str = "p50_ms") -> str:
            return _cell((latency.get(stage) or {}).get(key), 2)

        lines.append(
            "| {mode} | {n} | {acc} | {r1} | {neg} | {argem} | {fcem} | {strict} | "
            "{embed} | {retrieve} | {cross} | {total} | {p95} |".format(
                mode=row.get("distractor_label", row["mode"]),
                n=row["n"],
                acc=_cell(row["tool_set_accuracy"]),
                r1=_cell(row["recall_at_1"]),
                neg=_cell(row["negative_recall"]),
                argem=_cell(row["arg_em_given_correct_tool"]),
                fcem=_cell(row.get("n_fcem_positive")),
                strict=_cell(row.get("strict_arga")),
                embed=p("t_query_embed"),
                retrieve=p("t_retrieve"),
                cross=p("t_cross_encode"),
                total=p("total"),
                p95=p("total", "p95_ms"),
            )
        )
    impure = [
        row for row in report["rows"] if row.get("same_domain_purity") is not None and row["same_domain_purity"] < 1.0
    ]
    if impure:
        lines += [
            "",
            "`same_domain_first_mixed` có thêm distractor ngoài nhóm khi không đủ tool cùng nhóm. "
            "Đây là mixed distractors; tỉ lệ distractor thật sự cùng nhóm:",
            "",
            "| N | purity |",
            "|---:|---:|",
        ]
        lines += [f"| {row['n']} | {_cell(row['same_domain_purity'])} |" for row in impure]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 — stress test Method 2")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/stress.yaml"))
    parser.add_argument(
        "--pipeline-config", type=Path, default=Path("configs/method2/pipeline.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=None,
        help="Ghi đè n_values, dùng để chạy thử nhanh trước khi tốn quota.",
    )
    args = parser.parse_args()

    config = StressConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.n_queries:
        config.n_queries = args.n_queries
    if args.n_values:
        config.n_values = tuple(args.n_values)

    pipeline = Method2Pipeline.from_config(Method2Config.from_yaml(args.pipeline_config))
    report = run_stress(pipeline, config)
    print(summary_markdown(report))


if __name__ == "__main__":
    main()
