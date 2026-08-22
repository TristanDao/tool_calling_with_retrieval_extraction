"""Canonicalize + dedupe tool → tool pool thống nhất (§1.1 method2_plan).

Vấn đề: glaive có 768 tên tool nhưng ~10,500 signature khác nhau — cùng một
`tool_name` có hàng chục biến thể description/parameters do dịch khác nhau giữa
các sample. Nếu không gộp, cùng một tool vừa là positive vừa là negative trong
một batch MNRL → loss học nhiễu nặng.

Quy tắc gộp, cho mỗi `name`:

- `description`: biến thể **xuất hiện nhiều nhất** (tie → chuỗi dài hơn).
- `parameters`: **union theo key**; mỗi key giữ spec của biến thể phổ biến nhất.
- `required`: key nào required ở **đa số** biến thể thì mới giữ required.
- `feature_group`: giá trị khác `Khác` phổ biến nhất, nếu không có thì `Khác`.

Ghi ra `data/method2/tool_pool.json` kèm `n_variants` để log số biến thể đã gộp.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models.crossencoder.data_collator import (
    UNSUPPORTED_TYPES,
    iter_parameters,
    resolve_param_type,
)
from src.models.sources import DEFAULT_SOURCES, SourceSpec, iter_samples

UNKNOWN_FEATURE_GROUP = "Khác"


@dataclass
class ToolPoolConfig:
    output_path: Path = Path("data/method2/tool_pool.json")
    stats_path: Path = Path("data/method2/tool_pool_stats.json")
    #: Thêm tên parameter vào document text (ablation §6.3).
    include_param_names_in_doc: bool = True
    limit_per_source: int | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ToolPoolConfig":
        return cls(
            output_path=Path(raw.get("output_path", cls.output_path)),
            stats_path=Path(raw.get("stats_path", cls.stats_path)),
            include_param_names_in_doc=bool(
                raw.get("include_param_names_in_doc", True)
            ),
            limit_per_source=raw.get("limit_per_source"),
        )


def _signature(tool: dict[str, Any]) -> str:
    """Chữ ký để đếm số biến thể của một tool."""
    params = tool.get("parameters") or {}
    return json.dumps(
        [tool.get("description", ""), params], ensure_ascii=False, sort_keys=True
    )


def _most_common(values: list[Any], key=None) -> Any:
    counter = Counter(json.dumps(v, ensure_ascii=False, sort_keys=True) for v in values)
    best = max(counter.items(), key=lambda kv: (kv[1], len(kv[0])))
    return json.loads(best[0])


def build_tool_pool(
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    config: ToolPoolConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = config or ToolPoolConfig()

    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    signatures: dict[str, set[str]] = defaultdict(set)
    sources_of: dict[str, set[str]] = defaultdict(set)
    #: Tool từng là gold trong split train — dùng để assert zero-shot ở §1.
    gold_in_split: dict[str, set[str]] = defaultdict(set)

    for sample in iter_samples(specs, limit_per_source=config.limit_per_source):
        source = str(sample.get("source") or sample["_source_key"])
        for tool in sample.get("tools", []) or []:
            name = tool.get("name")
            if not name:
                continue
            variants[name].append(tool)
            signatures[name].add(_signature(tool))
            sources_of[name].add(source)
        for call in sample.get("function_calls", []) or []:
            if call.get("name"):
                gold_in_split[call["name"]].add(sample["_split"])

    pool: list[dict[str, Any]] = []
    for name, tools in sorted(variants.items()):
        merged = _merge_variants(name, tools)
        merged["n_variants"] = len(signatures[name])
        merged["sources"] = sorted(sources_of[name])
        merged["gold_in_splits"] = sorted(gold_in_split.get(name, set()))
        merged["doc_text"] = build_document_text(
            merged, include_param_names=config.include_param_names_in_doc
        )
        pool.append(merged)

    stats = _pool_stats(pool, signatures)
    return pool, stats


def _merge_variants(name: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    description = _most_common([t.get("description", "") or "" for t in tools])

    groups = [
        t.get("feature_group")
        for t in tools
        if t.get("feature_group") and t["feature_group"] != UNKNOWN_FEATURE_GROUP
    ]
    feature_group = Counter(groups).most_common(1)[0][0] if groups else UNKNOWN_FEATURE_GROUP

    specs_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    required_counts: Counter = Counter()
    for tool in tools:
        for param in iter_parameters(tool):
            spec = {k: v for k, v in param.items() if k not in ("name", "required", "routing_type", "value_type")}
            specs_by_key[param["name"]].append(spec)
            if param["required"]:
                required_counts[param["name"]] += 1

    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, specs in specs_by_key.items():
        properties[key] = _most_common(specs)
        if required_counts[key] * 2 >= len(tools):
            required.append(key)

    return {
        "name": name,
        "description": description,
        "feature_group": feature_group,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": sorted(required),
        },
    }


def build_document_text(tool: dict[str, Any], include_param_names: bool = True) -> str:
    """Text được Bi-Encoder encode cho một tool.

    Ablation §6.3 so `name+desc` với `name+desc+param_names`.
    """
    parts = [str(tool.get("name", "")), str(tool.get("description", "") or "")]
    if include_param_names:
        names = [p["name"] for p in iter_parameters(tool)]
        if names:
            parts.append("Tham số: " + ", ".join(names))
    return ". ".join(part for part in parts if part)


def _pool_stats(pool: list[dict[str, Any]], signatures: dict[str, set[str]]) -> dict[str, Any]:
    total_variants = sum(len(s) for s in signatures.values())
    type_counter: Counter = Counter()
    unsupported_tools = 0
    for tool in pool:
        has_unsupported = False
        for param in iter_parameters(tool):
            routing = resolve_param_type(param)
            type_counter[routing] += 1
            has_unsupported |= routing in UNSUPPORTED_TYPES
        unsupported_tools += int(has_unsupported)

    merged_most = sorted(
        ((name, len(sigs)) for name, sigs in signatures.items()),
        key=lambda kv: -kv[1],
    )[:10]

    return {
        "n_tools": len(pool),
        "n_signatures_before_merge": total_variants,
        "n_tools_with_unsupported_param": unsupported_tools,
        "param_type_distribution": dict(type_counter.most_common()),
        "feature_group_distribution": dict(
            Counter(t["feature_group"] for t in pool).most_common()
        ),
        "source_distribution": dict(
            Counter(src for t in pool for src in t["sources"]).most_common()
        ),
        "top_merged_tools": [{"name": n, "n_variants": v} for n, v in merged_most],
    }


def load_tool_pool(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return {tool["name"]: tool for tool in json.load(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical tool pool for Method 2")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/tool_pool.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit-per-source", type=int, default=None)
    args = parser.parse_args()

    raw: dict[str, Any] = {}
    if args.config.exists():
        import yaml

        raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = ToolPoolConfig.from_dict(raw)
    if args.output:
        config.output_path = args.output
    if args.limit_per_source:
        config.limit_per_source = args.limit_per_source

    pool, stats = build_tool_pool(config=config)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    config.stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[tool_pool] {stats['n_tools']} tool, gộp từ {stats['n_signatures_before_merge']} biến thể")
    print(f"[tool_pool] → {config.output_path}")


if __name__ == "__main__":
    main()
