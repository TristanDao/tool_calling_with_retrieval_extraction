"""Re-score preserved predictions and export auditable tables and error review cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.config import EvaluationConfig
from src.evaluation.error_analysis import classify_errors
from src.evaluation.evaluator import evaluate
from src.models.sources import load_jsonl, sha256_file, write_jsonl

DATASETS = {"benchmark": "data/benchmark_vi/test.jsonl",
            "custom_seen": "data/custom_vi/v1/test_seen.jsonl",
            "custom_unseen": "data/custom_vi/v1/test_unseen.jsonl"}


def review(source: Path, output: Path, bootstrap: int = 1000,
           predictions_root: Path | None = None,
           protocol: str = "legacy_negative_exposure_rescored") -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    rows = []
    for tag, gold_relative in DATASETS.items():
        pred_dir = (predictions_root or source / "results/method2/predictions") / tag
        stage = output / tag
        provenance = {}
        for name in ("predictions.jsonl", "oracle_predictions.jsonl"):
            path = pred_dir / name
            predictions = list(load_jsonl(path))
            for pred in predictions:
                telemetry = pred.get("telemetry", {})
                for key in ("cost_usd", "input_tokens", "output_tokens"):
                    if telemetry.get(key) == 0 and pred.get("metadata", {}).get("model") == "method_2":
                        telemetry.pop(key, None)
                pred.setdefault("metadata", {})["cost_basis"] = "unavailable_gpu_cost"
            write_jsonl(stage / name, predictions)
            provenance[name] = {"path": str(path), "sha256": sha256_file(path)}
        report, per_sample = evaluate(source / gold_relative, stage / "predictions.jsonl", stage,
                                      EvaluationConfig(bootstrap_samples=bootstrap,
                                                       slice_fields=(*EvaluationConfig().slice_fields, "metadata.tool_split")),
                                      stage / "oracle_predictions.jsonl")
        gold = {s["id"]: s for s in load_jsonl(source / gold_relative)}
        raw = {s["id"]: s for s in load_jsonl(pred_dir / "raw_predictions_pipeline.jsonl")}
        events = [event for pred in load_jsonl(stage / "predictions.jsonl")
                  for event in classify_errors(gold[pred["id"]], pred, raw.get(pred["id"], {}))]
        write_jsonl(stage / "errors_reclassified.jsonl", events)
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            buckets[event["error_class"]].append(event)
        review_cases = []
        for label, bucket in sorted(buckets.items()):
            picked = random.Random(f"42:{tag}:{label}").sample(bucket, min(10, len(bucket)))
            for event in picked:
                review_cases.append({**event, "query": gold[event["id"]]["query"],
                                     "human_label": None, "review_note": None})
        write_jsonl(stage / "human_review_queue.jsonl", review_cases)
        m = report["metrics"]
        row = {"dataset": tag, **report["dataset"],
               "tool_set_accuracy_positive": m["selection"]["tool_set_accuracy_positive"],
               "strict_arga": m["strict_end_to_end"]["n_fcem_positive"],
               "normalized_arga": m["end_to_end"]["n_fcem_positive"],
               "oracle_arga": m["oracle_end_to_end"]["n_fcem_positive"],
               "oracle_pipeline_gap_positive": m["oracle_pipeline_gap_positive"],
               "argument_f1": m["extraction"]["argument_pair"]["f1"],
               "schema_validity": m["schema_validity"]["call_schema_validity"],
               "p50_ms": m["efficiency"]["latency_ms"]["p50"],
               "p95_ms": m["efficiency"]["latency_ms"]["p95"],
               "error_events": dict(Counter(e["error_class"] for e in events)),
               "samples_with_error_events": len({e['id'] for e in events}),
               "failed_samples_evaluator": sum(not r["overall_success"] for r in per_sample),
               "ci": m["confidence_intervals"]["n_fcem_positive"]}
        rows.append(row)
        (stage / "source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"{tag}: {row['sample_count']} samples; strict={row['strict_arga']:.4f}; normalized={row['normalized_arga']:.4f}", flush=True)
    result = {"protocol": protocol, "rows": rows,
              "cost": "unavailable", "normalizer_ablation": "disabled_inference_run" if protocol.endswith("without_normalizer") else "not_measured",
              "notes": ["Test predictions unchanged except removal of placeholder cost/tokens.",
                        "Strict scoring is not inference normalizer ablation.",
                        "Heuristic W/T/P/I labels require human review."]}
    (output / "tables.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Method 2 — report", "", f"Protocol: {protocol}; candidate-scope. Cost unavailable.", "",
          "| Dataset | N | Strict ArgA | Normalized ArgA | Oracle ArgA | Gap | Arg F1 | P50 ms | P95 ms |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        values = [r[k] * 100 for k in ("strict_arga", "normalized_arga", "oracle_arga", "oracle_pipeline_gap_positive", "argument_f1")]
        md.append(f"| {r['dataset']} | {r['sample_count']} | " + " | ".join(f"{v:.2f}%" for v in values) + f" | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} |")
    md += ["", "All ArgA rates use positive samples. Oracle gap uses the same positive denominator.",
           "Strict and normalized score the same postprocessed predictions; they do not measure inference ablation.",
           "Review human_review_queue.jsonl before attributing W/T/P/I to linguistic causes.", ""]
    (output / "report.md").write_text("\n".join(md), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output_train/output_evaluation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/method2_completion/legacy_review"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--protocol", choices=["legacy_negative_exposure_rescored", "strict_unseen", "strict_unseen_without_normalizer"], default="legacy_negative_exposure_rescored")
    args = parser.parse_args()
    review(args.source, args.output, args.bootstrap, args.predictions_root, args.protocol)


if __name__ == "__main__":
    main()
