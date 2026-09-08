"""Prepare an additive, hash-verified CE supervision repair without changing BI artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.biencoder.strict_unseen import assert_no_exposure, load_excluded
from src.models.crossencoder.supervision import annotated_pairs
from src.models.sources import load_jsonl, normalize_query_key, sha256_file, write_jsonl


def prepare(source: Path, destination: Path) -> dict[str, Any]:
    import yaml

    if destination.exists():
        raise FileExistsError(f"Preserve existing output: {destination}")
    parent = json.loads((source / "completion_manifest.json").read_text(encoding="utf-8"))
    for name, digest in parent["files"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"Source bundle modified: {name}")
    custom_train = Path("data/custom_vi/v1/train.jsonl")
    data_manifest = json.loads((source / "data/method2/manifest.json").read_text(encoding="utf-8"))
    if sha256_file(custom_train) != data_manifest["sources"]["custom_train"]["sha256"]:
        raise ValueError("Local custom train must match the original training source")
    destination.mkdir(parents=True)
    report: dict[str, Any] = {"parent_completion_sha256": sha256_file(source / "completion_manifest.json"),
                              "custom_train_sha256": sha256_file(custom_train), "splits": {}}
    excluded = load_excluded(source / "data/method2/excluded_tools.json")
    split_queries: dict[str, set[str]] = {}
    for split, master_paths in [("train", [custom_train]),
                                 ("val", [source / f"data/custom_vi/v1/{s}.jsonl" for s in ("val_seen", "val_unseen")])]:
        old = list(load_jsonl(source / f"data/method2/crossencoder/{split}.jsonl"))
        old_custom = [row for row in old if row["source"] == "custom_vi"]
        retained_ids = {row["sample_id"] for row in old_custom}
        old_keys = {(r["sample_id"], r["tool_name"], r["param"]["name"]) for r in old_custom}
        rows = [row for row in old if row["source"] != "custom_vi"]
        repaired = []
        changed_spans = added = 0
        old_lookup = {(r["sample_id"], r["tool_name"], r["param"]["name"]): r for r in old_custom}
        for path in master_paths:
            for sample in load_jsonl(path):
                if sample["id"] not in retained_ids:
                    continue
                source_key = "custom_train" if split == "train" else "custom_" + sample["metadata"]["split"]
                repaired.extend(annotated_pairs(sample, split, source_key))
        if {row["sample_id"] for row in repaired} != retained_ids:
            raise ValueError(f"Retained sample IDs changed in {split}")
        for row in repaired:
            key = (row["sample_id"], row["tool_name"], row["param"]["name"])
            added += key not in old_keys
            if key in old_lookup:
                old_labels = old_lookup[key]["labels"]
                changed_spans += any(row["labels"].get(k) != old_labels.get(k) for k in ("char_start", "char_end"))
        rows.extend(repaired)
        if split == "train":
            assert_no_exposure(rows, excluded)
        split_queries[split] = {normalize_query_key(r["query"]) for r in rows}
        write_jsonl(destination / f"data/method2/ce_repair/{split}.jsonl", rows)
        report["splits"][split] = {"old_custom_pairs": len(old_custom), "new_custom_pairs": len(repaired),
                                   "added_pairs": added, "corrected_span_boundaries": changed_spans,
                                   "retained_sample_ids": len(retained_ids),
                                   "noncustom_pairs_unchanged": len(rows) - len(repaired),
                                   "surface_canonical_mismatch_by_type": dict(Counter(
                                       r["param"]["value_type"] for r in repaired
                                       if r.get("surface_normalizes_to_gold") is False))}
    if split_queries["train"] & split_queries["val"]:
        raise ValueError("Training and validation query overlap")
    test_queries = {normalize_query_key(s["query"]) for split in ("seen", "unseen")
                    for s in load_jsonl(source / f"data/custom_vi/v1/test_{split}.jsonl")}
    if split_queries["train"] & test_queries:
        raise ValueError("Training and test query overlap")
    config = yaml.safe_load((source / "configs/method2/crossencoder.yaml").read_text(encoding="utf-8"))
    config["data"].update({"train_path": "data/method2/ce_repair/train.jsonl", "val_path": "data/method2/ce_repair/val.jsonl"})
    config["train"].update({"output_dir": "artifacts/method2/crossencoder/repair_v1", "resume_from": None})
    config_path = destination / "configs/method2/ce_repair_v1.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for name in ["scripts/method2/run_ce_repair.py", "src/models/crossencoder/supervision.py", "docs/method2_ce_repair.md"]:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, path)
    (destination / "ce_repair_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    files = {p.relative_to(destination).as_posix(): sha256_file(p) for p in destination.rglob("*") if p.is_file()}
    if set(files) & set(parent["files"]):
        raise ValueError("The repair addon must not overwrite frozen bundle files")
    (destination / "ce_repair_manifest.json").write_text(json.dumps({"parent_completion_sha256": report["parent_completion_sha256"], "files": files}, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output_train/output_method2-completion-03-round2"))
    parser.add_argument("--destination", type=Path, default=Path("artifacts/method2_completion/ce_repair_v1"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), ensure_ascii=False, indent=2))
