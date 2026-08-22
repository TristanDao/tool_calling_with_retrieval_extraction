"""Dataset + pair builder cho Cross-Encoder (§1.4, §1.5 method2_plan).

Hai phần:

- `build_pairs()` — với mỗi gold call, sinh một dòng cho **mỗi parameter** của
  tool đó, gồm cả parameter không nằm trong `arguments` (nhãn `has_value=0`).
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.models.crossencoder.data_collator import (
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
from src.models.sources import DEFAULT_SOURCES, SourceSpec, iter_samples, load_jsonl, write_jsonl


@dataclass
class CrossEncoderPairsConfig:
    output_dir: Path = Path("data/method2/crossencoder")
    stats_path: Path = Path("data/method2/label_stats.json")
    #: Ngưỡng cảnh báo %SKIP (§1.5) — vượt ngưỡng thì dừng và xem lại Q2.
    skip_rate_gate: float = 0.30
    require_boolean_cue: bool = False
    limit_per_source: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CrossEncoderPairsConfig":
        defaults = cls()
        return cls(
            output_dir=Path(raw.get("output_dir", defaults.output_dir)),
            stats_path=Path(raw.get("stats_path", defaults.stats_path)),
            skip_rate_gate=float(raw.get("skip_rate_gate", defaults.skip_rate_gate)),
            require_boolean_cue=bool(raw.get("require_boolean_cue", False)),
            limit_per_source=raw.get("limit_per_source"),
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

    for sample in iter_samples(specs, limit_per_source=config.limit_per_source):
        query = str(sample.get("query", "")).strip()
        calls = sample.get("function_calls") or []
        if not query or not calls:
            continue
        schemas = {t["name"]: t for t in (sample.get("tools") or []) if t.get("name")}
        split = sample["_split"]
        source = sample.get("source") or sample["_source_key"]

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
            token_span = self.aligner.align_char_span_to_tokens(
                row["query"], labels["char_start"], labels["char_end"], self.max_length
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
    args = parser.parse_args()

    raw: dict[str, Any] = {}
    if args.config.exists():
        import yaml

        raw = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("pairs", {})
    config = CrossEncoderPairsConfig.from_dict(raw)
    if args.limit_per_source:
        config.limit_per_source = args.limit_per_source

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
    if stats["skip_rate"] > config.skip_rate_gate:
        print(
            f"[ce-pairs] CẢNH BÁO: SKIP {stats['skip_rate']:.1%} > ngưỡng "
            f"{config.skip_rate_gate:.0%} — xem lại Q2 (fuzzy span alignment) trước khi train"
        )


if __name__ == "__main__":
    main()
