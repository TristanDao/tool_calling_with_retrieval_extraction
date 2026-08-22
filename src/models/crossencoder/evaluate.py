"""Metric theo từng head của Cross-Encoder + gate §Phase 3 method2_plan.

Đây là metric **chẩn đoán ở mức thành phần**, không thay thế evaluator chung.
End-to-end (N-FCEM, ArgEM, Overall Success) đi qua `src/evaluation` trên file
`predictions.jsonl` theo prediction contract — xem `docs/evaluation.md`.

Gate để qua Phase 4, đo ở chế độ **oracle retrieval** trên custom val:

| Metric | Ngưỡng |
|---|---|
| `has_value` F1 | ≥ 0.90 |
| Span EM | ≥ 0.80 |
| Enum accuracy | ≥ 0.90 |
| Boolean accuracy | ≥ 0.85 |
| Argument EM (per-call) | ≥ 0.70 |
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.models.crossencoder.data_collator import (
    SCHEMA_TYPE_BOOLEAN,
    SCHEMA_TYPE_ENUM,
    SCHEMA_TYPE_NUMBER,
    SCHEMA_TYPE_STRING,
)


@dataclass
class ComponentMetrics:
    """Tích luỹ theo batch — dùng cả trong vòng train lẫn khi eval riêng."""

    has_value_threshold: float = 0.5
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, outputs: dict[str, torch.Tensor], labels: dict[str, Any]) -> None:
        probs = torch.sigmoid(outputs["has_value"]).detach().float().cpu()
        gold_has = labels["has_value"].detach().cpu()
        predicted = (probs >= self.has_value_threshold).long()

        self.counts["hv_tp"] += int(((predicted == 1) & (gold_has == 1)).sum())
        self.counts["hv_fp"] += int(((predicted == 1) & (gold_has == 0)).sum())
        self.counts["hv_fn"] += int(((predicted == 0) & (gold_has == 1)).sum())
        self.counts["hv_tn"] += int(((predicted == 0) & (gold_has == 0)).sum())

        schema_types = labels.get("schema_type") or []
        start_pred = outputs["span_start"].detach().float().cpu().argmax(dim=-1)
        end_pred = outputs["span_end"].detach().float().cpu().argmax(dim=-1)
        enum_pred = outputs["enum_logits"].detach().float().cpu().argmax(dim=-1)
        bool_pred = outputs["boolean_logits"].detach().float().cpu().argmax(dim=-1)

        for i, schema_type in enumerate(schema_types):
            if gold_has[i] != 1:
                continue
            if schema_type in (SCHEMA_TYPE_STRING, SCHEMA_TYPE_NUMBER):
                self.counts["span_total"] += 1
                exact = (
                    int(start_pred[i]) == int(labels["span_start"][i])
                    and int(end_pred[i]) == int(labels["span_end"][i])
                )
                self.counts["span_exact"] += int(exact)
                self.counts["span_start_correct"] += int(
                    int(start_pred[i]) == int(labels["span_start"][i])
                )
            elif schema_type == SCHEMA_TYPE_ENUM:
                self.counts["enum_total"] += 1
                self.counts["enum_correct"] += int(
                    int(enum_pred[i]) == int(labels["enum_label"][i])
                )
            elif schema_type == SCHEMA_TYPE_BOOLEAN:
                self.counts["bool_total"] += 1
                self.counts["bool_correct"] += int(
                    int(bool_pred[i]) == int(labels["boolean_label"][i])
                )

    def compute(self) -> dict[str, float]:
        c = self.counts
        precision = _ratio(c["hv_tp"], c["hv_tp"] + c["hv_fp"])
        recall = _ratio(c["hv_tp"], c["hv_tp"] + c["hv_fn"])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "has_value_precision": round(precision, 4),
            "has_value_recall": round(recall, 4),
            "has_value_f1": round(f1, 4),
            "has_value_accuracy": round(
                _ratio(c["hv_tp"] + c["hv_tn"], sum(c[k] for k in ("hv_tp", "hv_tn", "hv_fp", "hv_fn"))), 4
            ),
            "span_em": round(_ratio(c["span_exact"], c["span_total"]), 4),
            "span_start_accuracy": round(_ratio(c["span_start_correct"], c["span_total"]), 4),
            "enum_accuracy": round(_ratio(c["enum_correct"], c["enum_total"]), 4),
            "boolean_accuracy": round(_ratio(c["bool_correct"], c["bool_total"]), 4),
            "n_span": c["span_total"],
            "n_enum": c["enum_total"],
            "n_boolean": c["bool_total"],
        }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def check_gates(metrics: dict[str, float], gates: dict[str, float]) -> dict[str, Any]:
    """So metric với ngưỡng gate; trả `passed` và danh sách chưa đạt."""
    mapping = {
        "has_value_f1": "has_value_f1",
        "span_em": "span_em",
        "enum_accuracy": "enum_accuracy",
        "boolean_accuracy": "boolean_accuracy",
        "argument_em": "argument_em",
    }
    failures = []
    for gate_key, metric_key in mapping.items():
        if gate_key not in gates or metric_key not in metrics:
            continue
        if metrics[metric_key] < gates[gate_key]:
            failures.append(
                {"metric": metric_key, "value": metrics[metric_key], "required": gates[gate_key]}
            )
    return {"passed": not failures, "failures": failures}


@torch.no_grad()
def evaluate_file(
    model_path: str,
    pairs_path: str | Path,
    max_length: int = 256,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    from src.models.crossencoder.data_collator import CollatorConfig, CrossEncoderCollator
    from src.models.crossencoder.dataset import CrossEncoderDataset
    from src.models.crossencoder.inference import load_cross_encoder

    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_cross_encoder(model_path, device_obj)
    collator = CrossEncoderCollator(
        CollatorConfig(tokenizer_name=model_path, max_length=max_length), tokenizer=tokenizer
    )
    dataset = CrossEncoderDataset(pairs_path, collator, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    from src.models.crossencoder.train import forward_batch, move_batch

    model.eval()
    overall = ComponentMetrics()
    by_source: dict[str, ComponentMetrics] = defaultdict(ComponentMetrics)
    row_index = 0
    for batch in loader:
        sources = [
            dataset.rows[i]["source"]
            for i in range(row_index, min(row_index + batch_size, len(dataset.rows)))
        ]
        row_index += batch_size
        moved = move_batch(batch, device_obj)
        outputs = forward_batch(model, moved)
        cpu_labels = {
            k: (v.cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in moved["labels"].items()
        }
        overall.update(outputs, cpu_labels)
        for source in set(sources):
            by_source[source].update(outputs, cpu_labels)

    return {
        "overall": overall.compute(),
        "by_source": {source: m.compute() for source, m in sorted(by_source.items())},
        "n_pairs": len(dataset),
        "n_dropped_unalignable": dataset.n_dropped_unalignable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder heads")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/crossencoder.yaml"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import yaml

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    report = evaluate_file(
        args.model,
        args.pairs,
        max_length=int(raw.get("data", {}).get("max_length", 256)),
        batch_size=int(raw.get("train", {}).get("eval_batch_size", 16)),
    )
    report["gates"] = check_gates(report["overall"], raw.get("gates", {}))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
