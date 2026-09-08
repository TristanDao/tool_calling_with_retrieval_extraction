"""Build an isolated strict-unseen dataset and portable Kaggle run configurations."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.biencoder.strict_unseen import assert_no_exposure, audit_rows, filter_training_rows
from src.models.sources import load_jsonl, sha256_file, write_jsonl


@dataclass
class PreparationConfig:
    source: Path = Path("output_train/output_evaluation")
    destination: Path = Path("artifacts/method2_completion/kaggle_bundle")
    seed: int = 42
    n_negatives: int = 4


def prepare(config: PreparationConfig) -> dict[str, Any]:
    import yaml

    source = config.source.resolve()
    destination = config.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Use a new destination; preserving {destination}")
    excluded = set()
    for split in ("val_unseen", "test_unseen"):
        for sample in load_jsonl(source / f"data/custom_vi/v1/{split}.jsonl"):
            excluded.update(call["name"] for call in sample["function_calls"])
    pool_path = source / "data/method2/tool_pool.json"
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    if isinstance(pool, list):
        pool = {tool["name"]: tool for tool in pool}
    cleaned = filter_training_rows(load_jsonl(source / "data/method2/biencoder/train.jsonl"),
                                   pool, excluded, config.n_negatives, config.seed)
    assert_no_exposure(load_jsonl(source / "data/method2/crossencoder/train.jsonl"), excluded)
    destination.mkdir(parents=True)
    for relative in ("src", "scripts/method2", "configs/method2", "configs/eval"):
        shutil.copytree(relative, destination / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "notebooks").mkdir()
    for notebook in Path("notebooks").glob("method2_completion_*.ipynb"):
        shutil.copy2(notebook, destination / "notebooks" / notebook.name)
    (destination / "docs").mkdir()
    shutil.copy2("docs/method2_completion.md", destination / "docs/method2_completion.md")
    shutil.copytree(source / "data", destination / "data",
                    ignore=shutil.ignore_patterns("index", "train_mined.jsonl"))
    write_jsonl(destination / "data/method2/biencoder/train.jsonl", cleaned)
    exclusion = {"protocol": "strict_unseen", "excluded_tools": sorted(excluded),
                 "seed": config.seed, "includes_validation_unseen": True,
                 "source_train_sha256": sha256_file(source / "data/method2/biencoder/train.jsonl"),
                 "before": audit_rows(load_jsonl(source / "data/method2/biencoder/train.jsonl"), excluded),
                 "after": audit_rows(cleaned, excluded)}
    (destination / "data/method2/excluded_tools.json").write_text(
        json.dumps(exclusion, ensure_ascii=False, indent=2), encoding="utf-8")
    base = yaml.safe_load(Path("configs/method2/biencoder.yaml").read_text(encoding="utf-8"))
    for round_number in (1, 2):
        cfg = copy.deepcopy(base)
        run = f"artifacts/method2/biencoder/strict_round{round_number}"
        cfg["pairs"]["excluded_tools_path"] = "data/method2/excluded_tools.json"
        cfg["pairs"]["n_hard_negatives"] = config.n_negatives
        cfg["train"].update({"train_path": "data/method2/biencoder/" +
                             ("train.jsonl" if round_number == 1 else "train_mined.jsonl"),
                             "output_dir": run, "resume_from": None, "seed": config.seed})
        cfg["thresholds"].update({"strategy": "auto", "output_path": f"{run}/thresholds.json"})
        cfg["mining"].update({"output_path": "data/method2/biencoder/train_mined.jsonl",
                              "n_negatives": config.n_negatives})
        (destination / f"configs/method2/strict_round{round_number}.yaml").write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    pipeline = yaml.safe_load(Path("configs/method2/pipeline.yaml").read_text(encoding="utf-8"))
    pipeline["models"].update({"biencoder": "artifacts/method2/biencoder/strict_round2/final",
                                "thresholds": "artifacts/method2/biencoder/strict_round2/thresholds.json",
                                "crossencoder": "artifacts/method2/crossencoder/run01/final"})
    (destination / "configs/method2/completion_pipeline.yaml").write_text(
        yaml.safe_dump(pipeline, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ce = yaml.safe_load(Path("configs/method2/crossencoder.yaml").read_text(encoding="utf-8"))
    ce["train"].update({"output_dir": "artifacts/method2/crossencoder/custom4", "resume_from": None})
    for stage in ce["curriculum"]["stages"]:
        if stage["name"] == "finetune":
            stage["epochs"] = 4
    (destination / "configs/method2/ce_custom4.yaml").write_text(
        yaml.safe_dump(ce, allow_unicode=True, sort_keys=False), encoding="utf-8")
    group_counts = Counter(tool.get("feature_group", "Khác") for tool in pool.values())
    feasibility = {"pool_groups": dict(group_counts), "groups_with_1000_tools": [g for g,n in group_counts.items() if g != "Khác" and n >= 1000],
                   "pure_same_domain_full_plan_feasible": all(group_counts.get(tool.get("feature_group"), 0) >= 1000
                                                             for tool in pool.values() if tool.get("feature_group") != "Khác"),
                   "reason": "Verify per-anchor same-group capacity; unlabeled Khác is not a domain.",
                   "random_instances": 1200, "same_domain_status": "pending_group_enrichment_and_capacity_check"}
    (destination / "same_domain_feasibility.json").write_text(
        json.dumps(feasibility, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = destination / "data/method2/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parent_manifest_sha256"] = sha256_file(source / "data/method2/manifest.json")
    manifest["protocol"] = "strict_unseen"
    manifest["preparation_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    for path in (destination / "data/method2").rglob("*"):
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(destination).as_posix()
            old_key = next((k for k in manifest["derived"] if k.replace("\\", "/") == relative), relative)
            manifest["derived"][old_key] = {"present": True, "bytes": path.stat().st_size,
                                             "sha256": sha256_file(path)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = {p.relative_to(destination).as_posix(): sha256_file(p)
             for p in destination.rglob("*") if p.is_file()}
    package = {"protocol": "strict_unseen", "source": str(source), "files": files,
               "note": "Source hashes identify uncommitted edits; commit alone is insufficient."}
    (destination / "completion_manifest.json").write_text(
        json.dumps(package, indent=2), encoding="utf-8")
    return exclusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PreparationConfig.source)
    parser.add_argument("--destination", type=Path, default=PreparationConfig.destination)
    args = parser.parse_args()
    print(json.dumps(prepare(PreparationConfig(args.source, args.destination)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
