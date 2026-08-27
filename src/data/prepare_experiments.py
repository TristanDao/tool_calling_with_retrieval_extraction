"""Materialize reproducible training and evaluation data for Method 1 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.data.convert_to_instruction import convert_file


ROOT = Path("data")
DEFAULT_OUTPUT = ROOT / "experiments"
SEED = 42


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_group(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _build_core_splits(
    english: list[dict[str, Any]], vietnamese: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    vi_by_id = {row["id"]: row for row in vietnamese}
    if len(vi_by_id) != len(vietnamese):
        raise ValueError("Vietnamese core data contains duplicate IDs")

    groups: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for row in english:
        if row["id"] not in vi_by_id:
            raise ValueError(f"Missing Vietnamese pair for {row['id']}")
        key = (row["source"], not bool(row.get("function_calls")))
        groups.setdefault(key, []).append(row)

    en_splits = {name: [] for name in ("train", "val", "test")}
    vi_splits = {name: [] for name in ("train", "val", "test")}
    for index, rows in enumerate(groups.values()):
        split = _split_group(rows, SEED + index)
        for name, selected in split.items():
            en_splits[name].extend(selected)
            vi_splits[name].extend(vi_by_id[row["id"]] for row in selected)
    return en_splits, vi_splits


def _take_quota(
    rows: list[dict[str, Any]], positive_by_source: dict[str, int], negative: int,
) -> list[dict[str, Any]]:
    positives = [row for row in rows if row.get("function_calls")]
    negatives = [row for row in rows if not row.get("function_calls")]
    selected: list[dict[str, Any]] = []
    source_seeds = {"glaive": 43, "xlam": 44}
    for source, count in positive_by_source.items():
        source_rows = [row for row in positives if row["source"] == source]
        if len(source_rows) < count:
            raise ValueError(f"Insufficient {source} positive data: need {count}, got {len(source_rows)}")
        random.Random(source_seeds[source]).shuffle(source_rows)
        selected.extend(source_rows[:count])
    if len(negatives) < negative:
        raise ValueError(f"Insufficient negative data: need {negative}, got {len(negatives)}")
    random.Random(SEED).shuffle(negatives)
    selected.extend(negatives[:negative])
    random.Random(SEED).shuffle(selected)
    return selected


def _write_experiment(
    name: str,
    output: Path,
    train_en: list[dict[str, Any]],
    train_vi: list[dict[str, Any]],
    eval_en: dict[str, list[dict[str, Any]]],
    eval_vi: dict[str, list[dict[str, Any]]],
    include_custom_train: bool,
) -> None:
    experiment = output / name
    experiment.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    custom_train = _read_jsonl(ROOT / "custom_vi/train.jsonl") if include_custom_train else []

    if train_en or train_vi:
        if train_en:
            path = experiment / "train_en.jsonl"
            _write_jsonl(path, train_en)
            files.append(path)
        if train_vi:
            path = experiment / "train_vi.jsonl"
            _write_jsonl(path, train_vi)
            files.append(path)
        merged = train_en + train_vi + custom_train
        path = experiment / "train.jsonl"
        _write_jsonl(path, merged)
        files.append(path)
        train_sources = [("en", "en", train_en), ("vi", "vi", train_vi)]
        if custom_train:
            custom_path = experiment / "train_custom_vi.jsonl"
            _write_jsonl(custom_path, custom_train)
            files.append(custom_path)
            train_sources.append(("vi", "custom_vi", custom_train))
        instruction_dir = experiment / "instruction"
        instruction_parts: list[Path] = []
        for language, label, rows in train_sources:
            if not rows:
                continue
            source_path = experiment / f"_train_{label}.jsonl"
            output_path = instruction_dir / f"train_{label}_chat.jsonl"
            _write_jsonl(source_path, rows)
            converted, skipped = convert_file(source_path, output_path, language=language)
            if skipped or converted != len(rows):
                raise ValueError(f"Instruction conversion mismatch for {name}/{language}")
            source_path.unlink()
            files.append(output_path)
            instruction_parts.append(output_path)
        combined = instruction_dir / "train_chat.jsonl"
        with combined.open("w", encoding="utf-8") as target:
            for part in instruction_parts:
                target.write(part.read_text(encoding="utf-8"))
        files.append(combined)

    for language, splits in (("en", eval_en), ("vi", eval_vi)):
        for split_name in ("val", "test"):
            path = experiment / f"{split_name}_{language}.jsonl"
            _write_jsonl(path, splits[split_name])
            files.append(path)
            if split_name == "val":
                instruction_path = experiment / "instruction" / f"val_{language}_chat.jsonl"
                convert_file(path, instruction_path, language=language)
                files.append(instruction_path)

    val_parts = [
        experiment / "instruction" / "val_en_chat.jsonl",
        experiment / "instruction" / "val_vi_chat.jsonl",
    ]
    val_combined = experiment / "instruction" / "val_chat.jsonl"
    with val_combined.open("w", encoding="utf-8") as target:
        for part in val_parts:
            target.write(part.read_text(encoding="utf-8"))
    files.append(val_combined)

    custom_files: list[str] = []
    custom_dir = ROOT / "custom_vi"
    for filename in ("val_seen.jsonl", "val_unseen.jsonl", "test_seen.jsonl", "test_unseen.jsonl"):
        source = custom_dir / filename
        destination = experiment / f"custom_{filename}"
        _write_jsonl(destination, _read_jsonl(source))
        files.append(destination)
        custom_files.append(str(destination))

    manifest = {
        "experiment": name,
        "seed": SEED,
        "custom_train_included": include_custom_train,
        "files": {str(path.relative_to(experiment)): _sha256(path) for path in files},
        "train_counts": {
            "en": len(train_en),
            "vi": len(train_vi),
            "custom_vi": len(custom_train),
            "total": len(train_en) + len(train_vi) + len(custom_train),
        },
        "evaluation_files": custom_files,
    }
    (experiment / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def prepare(output: Path = DEFAULT_OUTPUT) -> None:
    en_positive = _read_jsonl(ROOT / "normalized_en/glaive_normalized.jsonl")
    en_positive += _read_jsonl(ROOT / "normalized_en/xlam_normalized.jsonl")
    en_negative = _read_jsonl(ROOT / "normalized_en/glaive_negative.jsonl")
    vi_positive = _read_jsonl(ROOT / "translations/glaive_normalized_vi.jsonl")
    vi_positive += _read_jsonl(ROOT / "translations/xlam_normalized_vi.jsonl")
    vi_negative = _read_jsonl(ROOT / "translations/glaive_negative_vi.jsonl")

    en_splits, vi_splits = _build_core_splits(en_positive + en_negative, vi_positive + vi_negative)
    controlled_en = _take_quota(
        en_splits["train"], positive_by_source={"glaive": 23_000, "xlam": 31_000}, negative=6_000
    )
    vi_train_by_id = {row["id"]: row for row in vi_splits["train"]}
    controlled_vi = [
        vi_train_by_id[selected["id"]]
        for selected in controlled_en
    ]
    bilingual_en = _take_quota(
        en_splits["train"], positive_by_source={"glaive": 11_500, "xlam": 15_500}, negative=3_000
    )
    bilingual_ids = {row["id"] for row in bilingual_en}
    bilingual_vi = [row for row in vi_splits["train"] if row["id"] in bilingual_ids]

    for name, train_en, train_vi, custom in (
        ("e0", [], [], False),
        ("e1", controlled_en, [], False),
        ("e2", [], controlled_vi, False),
        ("e3", controlled_en, [], False),
        ("e4", bilingual_en, bilingual_vi, False),
        ("e5", bilingual_en, bilingual_vi, True),
    ):
        _write_experiment(name, output, train_en, train_vi, en_splits, vi_splits, custom)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.output)
    print(f"[prepare] experiments written to {args.output}")


if __name__ == "__main__":
    main()
