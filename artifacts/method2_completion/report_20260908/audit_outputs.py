"""Audit saved Method 2 runs and collect reproducible report evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TRAIN = ROOT / "output_train"
RUNS = {
    "legacy_bi": "output_biencoder", "legacy_ce": "output_crossencoder",
    "legacy_eval": "output_evaluation", "validation": "output_method2-completion-01-validation",
    "round1": "output_method2-completion-02-round1", "round2": "output_method2-completion-03-round2",
    "repair": "output_method2-ce-repair-01b", "evaluation": "output_method2-completion-04-evaluation",
    "stress": "output_method2-completion-05-stress", "ablation": "output_method2-completion-06-normalizer-ablation",
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    result: dict[str, Any] = {"date": "2026-09-08", "root": str(TRAIN), "inventory": {}, "notebooks": {}}
    for tag, name in RUNS.items():
        root = TRAIN / name
        files = [p for p in root.rglob("*") if p.is_file()]
        result["inventory"][tag] = {"folder": name, "files": len(files), "bytes": sum(p.stat().st_size for p in files)}
    for path in TRAIN.glob("*.ipynb"):
        notebook = read(path)
        errors = [o for c in notebook["cells"] for o in c.get("outputs", []) if o.get("output_type") == "error"]
        result["notebooks"][path.name] = {"sha256": sha(path), "error_count": len(errors), "executed_cells": sum(c.get("execution_count") is not None for c in notebook["cells"])}
    aroot = TRAIN / RUNS["ablation"]
    eroot = TRAIN / RUNS["evaluation"]
    marker_path = aroot / "results/method2/completion/normalizer_ablation_completed.json"
    marker = read(marker_path)
    prior = read(eroot / "results/method2/completion/evaluation_completed.json")
    manifest = read(aroot / "completion_manifest.json")
    for name, expected in manifest["files"].items():
        assert sha(aroot / name) == expected, name
    for name, expected in marker["checkpoint_hashes"].items():
        assert sha(aroot / name) == expected, name
    assert marker["checkpoint_hashes"] == prior["checkpoint_hashes"]
    result["ablation_verification"] = {"bundle_files": len(manifest["files"]), "artifact_files": len(marker["checkpoint_hashes"]), "same_artifacts_as_04": True, "marker": marker, "bundle_sha256": sha(aroot / "completion_manifest.json")}
    result["tables"] = {}
    result["paired"] = {}
    result["detailed"] = {}
    result["dataset_stats"] = {}
    result["error_examples"] = []
    for tag, stage in [("evaluation", "evaluation"), ("ablation", "normalizer_ablation")]:
        root = TRAIN / RUNS[tag]
        base = root / "results/method2/completion" / stage
        result["tables"][tag] = read(base / "report/tables.json")
        result["detailed"][tag] = {}
        for dataset, gold_path in {"benchmark": "data/benchmark_vi/test.jsonl", "custom_seen": "data/custom_vi/v1/test_seen.jsonl", "custom_unseen": "data/custom_vi/v1/test_unseen.jsonl"}.items():
            gold = records(root / gold_path)
            gold_ids = {s["id"] for s in gold}
            result["detailed"][tag][dataset] = read(base / "report" / dataset / "report.json")
            for file in ["predictions.jsonl", "oracle_predictions.jsonl", "raw_predictions_pipeline.jsonl", "raw_predictions_oracle.jsonl"]:
                rows = records(base / dataset / file)
                assert len(rows) == len({r["id"] for r in rows}) == len(gold_ids)
                assert {r["id"] for r in rows} == gold_ids
            if tag == "ablation":
                before = {r["id"]: r for r in records(eroot / "results/method2/completion/evaluation" / dataset / "predictions.jsonl")}
                after = records(base / dataset / "predictions.jsonl")
                result["paired"][dataset] = {
                    "count": len(after),
                    "changed_calls": sum(r["function_calls"] != before[r["id"]]["function_calls"] for r in after),
                    "changed_tool_sequences": sum([c["name"] for c in r["function_calls"]] != [c["name"] for c in before[r["id"]]["function_calls"]] for r in after),
                    "changed_rankings": sum(r["ranked_tools"] != before[r["id"]]["ranked_tools"] for r in after),
                }
            else:
                parameter_types: Counter[str] = Counter()
                for sample in gold:
                    schemas = {t["name"]: t for t in sample["tools"]}
                    for call in sample["function_calls"]:
                        props = schemas.get(call["name"], {}).get("parameters", {}).get("properties", {})
                        for key in call["arguments"]:
                            schema = props.get(key, {})
                            parameter_types["enum" if "enum" in schema else str(schema.get("type", "unknown"))] += 1
                result["dataset_stats"][dataset] = {"samples": len(gold), "positive": sum(bool(s["function_calls"]) for s in gold), "calls": sum(len(s["function_calls"]) for s in gold), "multi_call": sum(len(s["function_calls"]) > 1 for s in gold), "repeated_tool": sum(len(s["function_calls"]) > len({c["name"] for c in s["function_calls"]}) for s in gold), "argument_types": dict(parameter_types)}
                queue = records(base / "report" / dataset / "human_review_queue.jsonl")
                seen = set()
                for example in queue:
                    key = example["error_class"]
                    if key not in seen and key in ["W", "T", "P", "I"]:
                        result["error_examples"].append({"dataset": dataset, **example})
                        seen.add(key)
                result["dataset_stats"][dataset]["review_queue_count"] = len(queue)
                result["dataset_stats"][dataset]["human_reviewed"] = sum(bool(s.get("reviewed") or s.get("human_label")) for s in queue)
    result["tables"]["legacy"] = read(ROOT / "artifacts/method2_completion/legacy_review/tables.json")
    result["train_reports"] = {}
    for tag in ["legacy_bi", "legacy_ce", "round1", "round2", "repair"]:
        for path in (TRAIN / RUNS[tag] / "artifacts/method2").rglob("train_report.json"):
            result["train_reports"][f"{tag}/{path.parent.name}"] = read(path)
    rroot = TRAIN / RUNS["repair"]
    result["repair_comparison"] = read(rroot / "results/method2/ce_repair/comparison.json")
    result["gates"] = {tag: read(rroot / "results/method2/ce_repair" / tag / "validation_gate.json") for tag in ["baseline", "candidate"]}
    result["repair_audit"] = read(rroot / "ce_repair_audit.json")
    result["strict_audit"] = read(aroot / "data/method2/excluded_tools.json")
    result["retrieval"] = {scope: read(aroot / f"results/method2/completion/retrieval_val_{scope}.json") for scope in ["candidates", "pool"]}
    result["thresholds"] = read(aroot / "artifacts/method2/biencoder/strict_round2/thresholds.json")
    result["stress"] = read(TRAIN / RUNS["stress"] / "results/method2/completion/stress/stress_report.json")
    result["split_sizes"] = {}
    for parent in ["data/benchmark_vi", "data/custom_vi/v1"]:
        for p in (aroot / parent).glob("*.jsonl"):
            rs = records(p)
            result["split_sizes"][f"{parent}/{p.name}"] = {"n": len(rs), "positive": sum(bool(x.get("function_calls")) for x in rs)}
    excluded = set(result["strict_audit"]["excluded_tools"])
    exposures = {}
    for path in [aroot / "data/method2/biencoder/train.jsonl", TRAIN / RUNS["round2"] / "data/method2/biencoder/train_mined.jsonl"]:
        rs = records(path)
        serialized = [json.dumps(x, ensure_ascii=False) for x in rs]
        exposures[path.name] = {"rows": len(rs), "heldout_identifier_occurrences": sum(sum(f'"{tool}"' in x for tool in excluded) for x in serialized)}
    result["strict_training_scan"] = exposures
    (OUT / "evidence.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verified": result["ablation_verification"]["artifact_files"], "paired": result["paired"], "datasets": result["dataset_stats"], "training_scan": exposures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
