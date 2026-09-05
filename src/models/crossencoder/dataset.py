"""Dataset + pair builder cho Cross-Encoder (§1.4, §1.5 method2_plan).

Hai phần:

- `build_pairs()` — với mỗi gold call, sinh một dòng cho **mỗi parameter** của
  tool đó, gồm cả parameter không nằm trong `arguments` (nhãn `has_value=0`).
  Khi bật `should_call` (ablation §6.1), sinh thêm hàng cấp **tool**
  `(query, tool)` với nhãn nhị phân — xem `_should_call_row`.
  Nhãn ghi ra đĩa dùng **char span**, không phụ thuộc tokenizer, nên đổi backbone
  ở ablation §6.5 không phải sinh lại dữ liệu.
- `CrossEncoderDataset` — đọc file đó, tokenize theo backbone thật và quy char
  span về token index tại thời điểm train.

Sample nào không sinh được nhãn (non-verbatim, enum lạ, array/object) bị `SKIP`
và được đếm theo lý do vào `label_stats.json`. §1.5 quy định: **%SKIP > 30% thì
dừng lại** và xem lại quyết định Q2 (fuzzy span alignment).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_SHOULD_CALL,
    UNSUPPORTED_TYPES,
    CollatorConfig,
    CrossEncoderCollator,
    iter_parameters,
)
from src.models.crossencoder.label_generator import (
    LabelGenerator,
    LabelGeneratorConfig,
    SkipLabel,
)
from src.models.sources import (
    DEFAULT_SOURCES,
    SourceSpec,
    iter_samples,
    load_jsonl,
    load_or_build_decontamination,
    normalize_query_key,
    pairwise_overlap,
    write_jsonl,
)


@dataclass
class ShouldCallConfig:
    """Hàng cấp tool cho head `should_call` (ablation §6.1).

    **Negative chỉ lấy từ sample no-call.** Có thể sinh thêm negative kiểu
    "query cần tool, nhưng không phải tool NÀY" từ candidate sai của sample
    positive — cố tình không làm. Head này thay đúng một thứ: ngưỡng cosine τ,
    thứ chỉ trả lời "query có cần gọi tool không". Trộn thêm negative kiểu chọn
    sai tool là đổi luôn câu hỏi mà head phải học, và ablation không còn so được
    một-đổi-một với baseline. Việc chọn đúng tool đã do Bi-Encoder đảm nhiệm.

    Cân bằng lớp là bắt buộc chứ không phải tinh chỉnh: train có 59.130 sample
    positive so với 16.199 no-call. Để nguyên tỉ lệ đó thì head học được cách
    luôn trả 1 và vẫn đạt loss thấp — hỏng đúng lớp mà §7.10e chỉ ra là nút thắt.
    """

    enabled: bool = False
    #: Số tool ghép với mỗi sample no-call. Ở train trung bình chỉ có 2,11
    #: candidate/sample no-call nên 1 là gần như "mỗi sample một dòng".
    tools_per_negative: int = 1
    #: Tỉ lệ nhãn 1 sau khi cân bằng. 0.5 = downsample positive về bằng negative.
    positive_rate: float = 0.5
    seed: int = 42

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ShouldCallConfig":
        defaults = cls()
        return cls(
            enabled=bool(raw.get("enabled", defaults.enabled)),
            tools_per_negative=int(
                raw.get("tools_per_negative", defaults.tools_per_negative)
            ),
            positive_rate=float(raw.get("positive_rate", defaults.positive_rate)),
            seed=int(raw.get("seed", defaults.seed)),
        )


def _should_call_row(
    sample: dict[str, Any],
    tool: dict[str, Any],
    label: int,
    split: str,
    source: str,
) -> dict[str, Any]:
    """Một hàng `(query, tool)`.

    Dùng đúng khuôn hàng cấp parameter để `CrossEncoderDataset` không phải biết
    gì về loại hàng này: `param` mang `routing_type="should_call"` nên
    `build_schema_question` tự chuyển sang question cấp tool, còn `has_value=0`
    khiến bước căn span bị bỏ qua.
    """
    return {
        "sample_id": sample.get("id"),
        "query": str(sample.get("query", "")).strip(),
        "tool_name": tool.get("name"),
        "param": {
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "param_names": [p["name"] for p in iter_parameters(tool)],
            "routing_type": SCHEMA_TYPE_SHOULD_CALL,
        },
        "labels": {
            "has_value": 0,
            "should_call": label,
            "schema_type": SCHEMA_TYPE_SHOULD_CALL,
        },
        "source": source,
        "source_key": sample["_source_key"],
        "split": split,
        "tool_split": sample["_tool_split"],
    }


def balance_should_call(
    positives: dict[str, list[dict[str, Any]]],
    negatives: dict[str, list[dict[str, Any]]],
    config: ShouldCallConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Downsample lớp trội về đúng `positive_rate`, xác định theo seed."""
    balanced: dict[str, list[dict[str, Any]]] = {}
    report: dict[str, Any] = {}
    rate = min(max(config.positive_rate, 0.0), 1.0)
    for split in sorted(set(positives) | set(negatives)):
        pos, neg = positives.get(split, []), negatives.get(split, [])
        # Giữ trọn lớp thiếu, cắt lớp thừa — không vứt bớt lớp hiếm chỉ để
        # chiều một tỉ lệ.
        n_pos, n_neg = len(pos), len(neg)
        if pos and neg and 0.0 < rate < 1.0:
            if n_pos / (n_pos + n_neg) > rate:
                n_pos = min(n_pos, int(round(n_neg * rate / (1.0 - rate))))
            else:
                n_neg = min(n_neg, int(round(n_pos * (1.0 - rate) / rate)))
        rng = random.Random(f"{config.seed}:{split}")
        rows = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
        rng.shuffle(rows)
        balanced[split] = rows
        report[split] = {
            "positive": n_pos,
            "negative": n_neg,
            "positive_available": len(pos),
            "negative_available": len(neg),
            "positive_rate": round(n_pos / (n_pos + n_neg), 4) if rows else None,
        }
    return balanced, report


@dataclass
class CrossEncoderPairsConfig:
    output_dir: Path = Path("data/method2/crossencoder")
    stats_path: Path = Path("data/method2/label_stats.json")
    decontamination_path: Path = Path("data/method2/decontamination.json")
    #: Ngưỡng cảnh báo %SKIP (§1.5) — vượt ngưỡng thì dừng và xem lại Q2.
    skip_rate_gate: float = 0.30
    require_boolean_cue: bool = False
    limit_per_source: int | None = None
    allow_build_decontamination: bool = False
    should_call: "ShouldCallConfig" = field(default_factory=lambda: ShouldCallConfig())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CrossEncoderPairsConfig":
        defaults = cls()
        return cls(
            output_dir=Path(raw.get("output_dir", defaults.output_dir)),
            stats_path=Path(raw.get("stats_path", defaults.stats_path)),
            decontamination_path=Path(
                raw.get("decontamination_path", defaults.decontamination_path)
            ),
            skip_rate_gate=float(raw.get("skip_rate_gate", defaults.skip_rate_gate)),
            require_boolean_cue=bool(raw.get("require_boolean_cue", False)),
            limit_per_source=raw.get("limit_per_source"),
            should_call=ShouldCallConfig.from_dict(raw.get("should_call") or {}),
        )


def build_pairs(
    config: CrossEncoderPairsConfig,
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    generator = LabelGenerator(
        LabelGeneratorConfig(
            tokenizer_name="",  # tokenizer-free: chỉ sinh char span
            require_boolean_cue=config.require_boolean_cue,
        )
    )

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kept_by_type: Counter = Counter()
    skip_by_reason: Counter = Counter()
    skip_by_reason_type: Counter = Counter()
    skip_by_source: Counter = Counter()
    total_by_source: Counter = Counter()
    has_value_counter: Counter = Counter()
    n_calls = 0
    n_calls_without_schema = 0
    queries_seen: dict[str, set[str]] = defaultdict(set)

    # Dùng chung index với Bi-Encoder: hai stage phải chia split y hệt nhau,
    # nếu không val của stage này lại là test của stage kia.
    decontamination = load_or_build_decontamination(
        config.decontamination_path,
        specs,
        config.limit_per_source,
        allow_build=config.allow_build_decontamination,
    )

    should_call_positive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    should_call_negative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sc_rng = random.Random(config.should_call.seed)

    for sample in iter_samples(
        specs, limit_per_source=config.limit_per_source, decontamination=decontamination
    ):
        query = str(sample.get("query", "")).strip()
        calls = sample.get("function_calls") or []
        if not query:
            continue
        schemas = {t["name"]: t for t in (sample.get("tools") or []) if t.get("name")}
        split = sample["_split"]
        source = sample.get("source") or sample["_source_key"]

        if config.should_call.enabled:
            if calls:
                gold = [c.get("name") for c in calls if c.get("name") in schemas]
                if gold:
                    should_call_positive[split].append(
                        _should_call_row(sample, schemas[gold[0]], 1, split, source)
                    )
            elif schemas:
                picked = sc_rng.sample(
                    sorted(schemas),
                    min(config.should_call.tools_per_negative, len(schemas)),
                )
                should_call_negative[split].extend(
                    _should_call_row(sample, schemas[name], 0, split, source)
                    for name in picked
                )

        if not calls:
            continue
        queries_seen[split].add(normalize_query_key(query))

        for call in calls:
            tool_name = call.get("name")
            schema = schemas.get(tool_name)
            if schema is None:
                n_calls_without_schema += 1
                continue
            n_calls += 1
            arguments = call.get("arguments") or {}
            for param in iter_parameters(schema):
                total_by_source[source] += 1
                gold_value = arguments.get(param["name"])
                label = generator.generate(query, param, gold_value, param["required"])

                if isinstance(label, SkipLabel):
                    skip_by_reason[label.reason] += 1
                    skip_by_reason_type[f"{param['routing_type']}:{label.reason}"] += 1
                    skip_by_source[source] += 1
                    continue

                kept_by_type[param["routing_type"]] += 1
                has_value_counter[label["has_value"]] += 1
                rows_by_split[split].append(
                    {
                        "sample_id": sample.get("id"),
                        "query": query,
                        "tool_name": tool_name,
                        "param": _serialize_param(param),
                        "labels": label,
                        "source": source,
                        "source_key": sample["_source_key"],
                        "split": split,
                        "tool_split": sample["_tool_split"],
                    }
                )

    total_pairs = sum(total_by_source.values())
    total_skipped = sum(skip_by_reason.values())
    stats = {
        "n_gold_calls": n_calls,
        "n_calls_without_schema": n_calls_without_schema,
        "n_pairs_total": total_pairs,
        "n_pairs_kept": total_pairs - total_skipped,
        "n_pairs_skipped": total_skipped,
        "skip_rate": round(total_skipped / total_pairs, 4) if total_pairs else 0.0,
        "skip_by_reason": dict(skip_by_reason.most_common()),
        "skip_by_type_reason": dict(skip_by_reason_type.most_common()),
        "skip_rate_by_source": {
            src: round(skip_by_source[src] / total, 4)
            for src, total in total_by_source.items()
            if total
        },
        "kept_by_routing_type": dict(kept_by_type.most_common()),
        "has_value_distribution": {
            "positive": has_value_counter[1],
            "negative": has_value_counter[0],
            "negative_rate": round(
                has_value_counter[0] / max(sum(has_value_counter.values()), 1), 4
            ),
        },
        "unsupported_type_coverage": {
            "n_pairs": skip_by_reason.get("unsupported_type", 0),
            "rate": round(
                skip_by_reason.get("unsupported_type", 0) / total_pairs, 4
            )
            if total_pairs
            else 0.0,
        },
        "rows_per_split": {split: len(rows) for split, rows in rows_by_split.items()},
        "decontamination": decontamination.report(),
        "split_overlap_after": pairwise_overlap(queries_seen),
        "unique_queries_per_split": {k: len(v) for k, v in sorted(queries_seen.items())},
    }
    if config.should_call.enabled:
        balanced, sc_report = balance_should_call(
            should_call_positive, should_call_negative, config.should_call
        )
        for split, rows in balanced.items():
            rows_by_split[split].extend(rows)
        stats["should_call"] = sc_report
        stats["rows_per_split"] = {
            split: len(rows) for split, rows in rows_by_split.items()
        }

    return rows_by_split, stats


def _serialize_param(param: dict[str, Any]) -> dict[str, Any]:
    """Giữ đúng những field mà schema question và normalizer cần."""
    keys = ("name", "description", "type", "enum", "format", "required", "routing_type", "value_type")
    return {k: param[k] for k in keys if k in param}


class CrossEncoderDataset(Dataset):
    """Đọc pair file, tokenize, quy char span → token index theo backbone hiện tại."""

    def __init__(
        self,
        path: str | Path,
        collator: CrossEncoderCollator,
        max_length: int = 256,
    ) -> None:
        self.collator = collator
        self.max_length = max_length
        self.aligner = LabelGenerator(
            LabelGeneratorConfig(tokenizer_name="", max_length=max_length),
            tokenizer=collator.tokenizer,
        )
        self.rows: list[dict[str, Any]] = []
        self.n_dropped_unalignable = 0
        for row in load_jsonl(path):
            prepared = self._prepare(row)
            if prepared is None:
                # Char span nằm ngoài cửa sổ `max_length` của backbone này —
                # không thể sinh nhãn token hợp lệ nên bỏ, và đếm lại để báo cáo.
                self.n_dropped_unalignable += 1
                continue
            self.rows.append(prepared)

    def _prepare(self, row: dict[str, Any]) -> dict[str, Any] | None:
        labels = dict(row["labels"])
        if labels["has_value"] == 1 and "char_start" in labels:
            # PHẢI dùng ngân sách thật của collator, không phải `max_length`.
            # Collator cắt query xuống `max_length - n_question - n_special`;
            # căn span theo max_length thì nhãn trỏ ra ngoài chuỗi và bị
            # `_collate_labels._clip` kẹp về vị trí sai — hỏng nhãn, không báo.
            budget = self.collator.query_token_budget(row["param"])
            if budget < 8:
                return None
            # aligner nhận `max_length` của cả chuỗi và tự trừ 2, nên cộng bù.
            token_span = self.aligner.align_char_span_to_tokens(
                row["query"], labels["char_start"], labels["char_end"], budget + 2
            )
            if token_span is None:
                return None
            labels["span_start"], labels["span_end"] = token_span
        row = dict(row)
        row["labels"] = labels
        return row

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return self.collator.encode_one(row["query"], row["param"], row["labels"])


def make_dataset(
    path: str | Path,
    tokenizer_name: str = "xlm-roberta-base",
    max_length: int = 256,
    padding: str = "longest",
) -> tuple[CrossEncoderDataset, CrossEncoderCollator]:
    collator = CrossEncoderCollator(
        CollatorConfig(tokenizer_name=tokenizer_name, max_length=max_length, padding=padding)
    )
    return CrossEncoderDataset(path, collator, max_length=max_length), collator


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cross-Encoder training pairs")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/crossencoder.yaml"))
    parser.add_argument("--limit-per-source", type=int, default=None)
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
    config = CrossEncoderPairsConfig.from_dict(raw)
    if args.limit_per_source:
        config.limit_per_source = args.limit_per_source
    if args.rebuild_decontamination:
        config.allow_build_decontamination = True

    rows_by_split, stats = build_pairs(config)

    for split, rows in rows_by_split.items():
        path = config.output_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        print(f"[ce-pairs] {split}: {len(rows)} cặp → {path}")

    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[ce-pairs] tổng {stats['n_pairs_total']} cặp, SKIP {stats['skip_rate']:.1%}")
    for reason, count in list(stats["skip_by_reason"].items())[:6]:
        print(f"[ce-pairs]   {reason}: {count}")

    overlap = stats["split_overlap_after"]
    dirty = {pair: n for pair, n in overlap.items() if n}
    if dirty:
        raise SystemExit(f"[ce-pairs] LỖI: còn query dùng chung giữa các split: {dirty}")
    print(f"[ce-pairs] OK: overlap giữa các split = 0 ({overlap})")

    if stats["skip_rate"] > config.skip_rate_gate:
        print(
            f"[ce-pairs] CẢNH BÁO: SKIP {stats['skip_rate']:.1%} > ngưỡng "
            f"{config.skip_rate_gate:.0%} — xem lại Q2 (fuzzy span alignment) trước khi train"
        )


if __name__ == "__main__":
    main()
