"""Materialize minimal, reproducible training artifacts for Method 1 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.convert_to_instruction import convert_rows

ROOT = Path("data")
DEFAULT_OUTPUT = ROOT / "experiments"
DEFAULT_BENCHMARK_ROOT = ROOT / "benchmark_core"
DEFAULT_ACTIVE_BENCHMARK = ROOT / "benchmark_vi"
DEFAULT_SEED = 42
DEFAULT_MODEL = "unsloth/Qwen3.5-4B"
ALTERNATIVE_MODEL = "unsloth/Qwen3.5-2B"
TRAINING_BUDGET: dict[str, Any] = {
    "max_seq_length": 4096,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "effective_batch_size": 16,
    "learning_rate": 5e-7,
    "warmup_ratio": 0.05,
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "load_in_4bit": True,
}


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    entries: list[tuple[dict[str, Any], str]]
    validation_paths: list[str]
    validation_rule: str
    custom_train_included: bool = False
    prerequisite: dict[str, Any] | None = None
    sampling: dict[str, Any] | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_hash(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(ids):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _split_rows(revision_dir: Path, language: str, split: str) -> list[dict[str, Any]]:
    return _read_jsonl(revision_dir / language / f"{split}.jsonl")


def _resolve_revision(
    benchmark_root: Path,
    active_benchmark: Path,
    revision: str | None,
) -> tuple[str, Path]:
    if revision:
        revision_dir = benchmark_root / revision
        if not revision_dir.exists():
            raise FileNotFoundError(revision_dir)
        return revision, revision_dir
    metadata_path = active_benchmark / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Active benchmark metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = metadata.get("revision")
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"Active metadata does not identify a revision: {metadata_path}")
    revision_dir = benchmark_root / selected
    if not revision_dir.exists():
        raise FileNotFoundError(revision_dir)
    return selected, revision_dir


def _take_quota(
    rows: list[dict[str, Any]],
    total: int,
    positive_by_source: dict[str, int],
    negative_target: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if total < 0:
        raise ValueError("total must be non-negative")
    positives = [row for row in rows if row.get("function_calls")]
    negatives = [row for row in rows if not row.get("function_calls")]
    if total > len(rows):
        raise ValueError(f"Insufficient data: need {total}, got {len(rows)}")
    negative = min(max(negative_target, 0), len(negatives), total)
    positive_total = total - negative
    selected: list[dict[str, Any]] = []
    selected_positive = 0
    available_by_source = Counter(
        row.get("source", "") for row in positives
    )
    for offset, (source, count) in enumerate(sorted(positive_by_source.items())):
        source_rows = [row for row in positives if row.get("source") == source]
        random.Random(seed + offset + 1).shuffle(source_rows)
        take = min(count, len(source_rows), positive_total - selected_positive)
        selected.extend(source_rows[:take])
        selected_positive += take
    requested_source_shortfall = selected_positive < min(
        positive_total, sum(positive_by_source.values())
    )
    if selected_positive < positive_total:
        selected_ids = {row.get("id") for row in selected}
        remaining = [
            row
            for row in positives
            if row.get("id") not in selected_ids
        ]
        random.Random(seed + 50).shuffle(remaining)
        selected.extend(remaining[: positive_total - selected_positive])
        selected_positive = len(selected)
    if len(selected) < positive_total:
        raise ValueError(
            f"Insufficient positive data: need {positive_total}, got {len(selected)}"
        )
    random.Random(seed + 100).shuffle(negatives)
    selected.extend(negatives[:negative])
    random.Random(seed + 200).shuffle(selected)
    selected_source_counts = Counter(
        row.get("source", "") for row in selected if row.get("function_calls")
    )
    if len({row.get("id") for row in selected}) != len(selected):
        raise ValueError("Quota selection contains duplicate sample IDs")
    return selected, {
        "requested_total": total,
        "requested_negative": negative_target,
        "requested_positive_by_source": dict(sorted(positive_by_source.items())),
        "selected_total": len(selected),
        "selected_negative": negative,
        "selected_positive": len(selected) - negative,
        "selected_positive_by_source": dict(sorted(selected_source_counts.items())),
        "available_total": len(rows),
        "available_negative": len(negatives),
        "available_positive_by_source": dict(sorted(available_by_source.items())),
        "fallback_used": negative != negative_target or requested_source_shortfall,
    }


def _with_language(row: dict[str, Any], language: str) -> dict[str, Any]:
    result = dict(row)
    result["language"] = language
    return result


def _training_counts(entries: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    languages = Counter(language for _row, language in entries)
    sources = Counter(row.get("source", "") for row, _language in entries)
    labels = Counter("positive" if row.get("function_calls") else "negative" for row, _language in entries)
    return {
        "total": len(entries),
        "language": dict(sorted(languages.items())),
        "source": dict(sorted(sources.items())),
        "label": dict(sorted(labels.items())),
    }


def _write_experiment(
    spec: ExperimentSpec,
    output: Path,
    revision: str,
    revision_dir: Path,
    seed: int,
    general_sft_checkpoint: str | None,
) -> None:
    experiment_dir = output / spec.name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    train_rows = [_with_language(row, language) for row, language in spec.entries]
    files: list[Path] = []
    if train_rows:
        train_path = experiment_dir / "train.jsonl"
        _write_jsonl(train_path, train_rows)
        files.append(train_path)
        instruction_path = experiment_dir / "instruction" / "train_chat.jsonl"
        converted, skipped = convert_rows(train_rows, instruction_path)
        if converted != len(train_rows) or skipped:
            raise ValueError(
                f"Native conversion mismatch for {spec.name}: converted={converted}, skipped={skipped}"
            )
        files.append(instruction_path)

    core_ids = {row["id"] for row, _language in spec.entries if row.get("source") != "custom_vi"}
    custom_ids = {row["id"] for row, _language in spec.entries if row.get("source") == "custom_vi"}
    manifest: dict[str, Any] = {
        "schema_version": "method1-experiment-v2",
        "experiment": spec.name,
        "benchmark_revision": revision,
        "benchmark_path": str(revision_dir),
        "custom_train_included": spec.custom_train_included,
        "checkpoint": {
            "primary": DEFAULT_MODEL,
            "alternative": ALTERNATIVE_MODEL,
            "template_source": "loaded from the exact checkpoint at training time",
        },
        "seed": seed,
        "target_epochs": 1.0,
        "training": {
            "counts": _training_counts(spec.entries),
            "core_unique_ids": len(core_ids),
            "custom_unique_ids": len(custom_ids),
            "core_ids_sha256": _ids_hash(core_ids),
            "custom_ids_sha256": _ids_hash(custom_ids),
            "sample_id_policy": "E1/E2 use the same core IDs; E4 pairs EN and VI by ID",
            "sampling": spec.sampling or {},
            "budget": TRAINING_BUDGET,
        },
        "benchmark_manifest_sha256": _sha256(revision_dir / "manifest.json"),
        "split_manifest_sha256": _sha256(revision_dir / "split_manifest.json"),
        "validation": {
            "paths": spec.validation_paths,
            "rule": spec.validation_rule,
        },
        "evaluation": {
            "core": [
                str(revision_dir / "en" / "test.jsonl"),
                str(revision_dir / "vi" / "test.jsonl"),
            ],
            "custom_vi": [
                "data/custom_vi/test_seen.jsonl",
                "data/custom_vi/test_unseen.jsonl",
            ],
            "local_copies": False,
        },
        "negative_response_policy": (
            "Use assistant_content/negative_response when present; otherwise use the "
            "deterministic language-specific fallback in convert_to_instruction.py."
        ),
        "files": {},
    }
    if spec.prerequisite is not None:
        manifest["general_sft_prerequisite"] = spec.prerequisite
        if general_sft_checkpoint:
            manifest["general_sft_prerequisite"]["checkpoint"] = general_sft_checkpoint
    manifest["files"] = {
        str(path.relative_to(experiment_dir)): _sha256(path)
        for path in sorted(files)
    }
    _write_json(experiment_dir / "manifest.json", manifest)


def _make_specs(
    en_train: list[dict[str, Any]],
    vi_train: list[dict[str, Any]],
    custom_train: list[dict[str, Any]],
    revision_dir: Path,
    include_e3: bool,
    general_sft_checkpoint: str | None,
    seed: int,
) -> list[ExperimentSpec]:
    en_val = str(revision_dir / "en" / "val.jsonl")
    vi_val = str(revision_dir / "vi" / "val.jsonl")
    controlled_en, controlled_sampling = _take_quota(
        en_train,
        total=60_000,
        positive_by_source={"glaive": 10_712, "xlam": 45_439},
        negative_target=3_849,
        seed=seed,
    )
    vi_by_id = {row["id"]: row for row in vi_train}
    controlled_vi = [vi_by_id[row["id"]] for row in controlled_en]
    bilingual_en, bilingual_sampling = _take_quota(
        en_train,
        total=30_000,
        positive_by_source={"glaive": 10_712, "xlam": 16_288},
        negative_target=3_000,
        seed=seed + 300,
    )
    bilingual_ids = {row["id"] for row in bilingual_en}
    bilingual_vi = [row for row in vi_train if row["id"] in bilingual_ids]
    if len(bilingual_vi) != len(bilingual_en):
        raise ValueError("Bilingual training subset is not paired")

    specs = [
        ExperimentSpec("e0", [], [], "No checkpoint selection; zero-shot inference."),
        ExperimentSpec(
            "e1",
            [(row, "en") for row in controlled_en],
            [en_val],
            "Select on core English validation.",
            sampling=controlled_sampling,
        ),
        ExperimentSpec(
            "e2",
            [(row, "vi") for row in controlled_vi],
            [vi_val],
            "Select on core Vietnamese validation.",
            sampling=controlled_sampling,
        ),
        ExperimentSpec(
            "e4",
            [(row, "en") for row in bilingual_en] + [(row, "vi") for row in bilingual_vi],
            [en_val, vi_val],
            "Select with a pre-registered macro rule over core EN and VI validation.",
            sampling=bilingual_sampling,
        ),
        ExperimentSpec(
            "e5",
            [(row, "en") for row in bilingual_en]
            + [(row, "vi") for row in bilingual_vi]
            + [(row, "vi") for row in custom_train],
            [en_val, vi_val, "data/custom_vi/val_seen.jsonl", "data/custom_vi/val_unseen.jsonl"],
            "Select with the registered core plus CustomTools validation rule; report each subset separately.",
            custom_train_included=True,
            sampling={
                "core": bilingual_sampling,
                "custom_total": len(custom_train),
            },
        ),
    ]
    if include_e3:
        if not general_sft_checkpoint:
            raise ValueError("E3 requires --general-sft-checkpoint")
        specs.insert(
            3,
            ExperimentSpec(
                "e3",
                [(row, "en") for row in controlled_en],
                [en_val],
                "Select on core English validation after independent general SFT.",
                prerequisite={"required": True, "stage": "general_sft"},
            ),
        )
    return specs


def prepare(
    output: Path = DEFAULT_OUTPUT,
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT,
    active_benchmark: Path = DEFAULT_ACTIVE_BENCHMARK,
    revision: str | None = None,
    seed: int = DEFAULT_SEED,
    include_e3: bool = False,
    general_sft_checkpoint: str | None = None,
    overwrite: bool = False,
) -> str:
    """Materialize E0, E1, E2, E4 and E5 from one frozen revision."""
    selected_revision, revision_dir = _resolve_revision(benchmark_root, active_benchmark, revision)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Experiment output already exists: {output}; use --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    split_manifest_path = revision_dir / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(split_manifest, dict):
        raise ValueError(f"Expected object in {split_manifest_path}")
    for split in ("train", "val", "test"):
        expected_ids = {
            sample_id for sample_id, assigned_split in split_manifest.items() if assigned_split == split
        }
        for language in ("en", "vi"):
            split_rows = _split_rows(revision_dir, language, split)
            actual_ids = [row.get("id") for row in split_rows]
            if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != expected_ids:
                raise ValueError(
                    f"{language}/{split} does not match {split_manifest_path}: "
                    f"rows={len(actual_ids)}, assigned={len(expected_ids)}"
                )

    en_train = _split_rows(revision_dir, "en", "train")
    vi_train = _split_rows(revision_dir, "vi", "train")
    en_ids = {row["id"] for row in en_train}
    vi_ids = {row["id"] for row in vi_train}
    if en_ids != vi_ids:
        raise ValueError(f"Frozen revision train IDs are not paired: en={len(en_ids)}, vi={len(vi_ids)}")
    custom_train = _read_jsonl(ROOT / "custom_vi/train.jsonl")
    custom_ids = [row.get("id") for row in custom_train]
    if len(set(custom_ids)) != len(custom_ids):
        raise ValueError("CustomTools training data contains duplicate IDs")
    specs = _make_specs(
        en_train,
        vi_train,
        custom_train,
        revision_dir,
        include_e3,
        general_sft_checkpoint,
        seed,
    )
    for spec in specs:
        _write_experiment(
            spec,
            output,
            selected_revision,
            revision_dir,
            seed,
            general_sft_checkpoint,
        )
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "method1-experiment-set-v2",
            "benchmark_revision": selected_revision,
            "experiments": [spec.name for spec in specs],
            "e3_enabled": include_e3,
            "seed": seed,
        },
    )
    return selected_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--active-benchmark", type=Path, default=DEFAULT_ACTIVE_BENCHMARK)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--include-e3", action="store_true")
    parser.add_argument("--general-sft-checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected_revision = prepare(
        output=args.output,
        benchmark_root=args.benchmark_root,
        active_benchmark=args.active_benchmark,
        revision=args.revision,
        seed=args.seed,
        include_e3=args.include_e3,
        general_sft_checkpoint=args.general_sft_checkpoint,
        overwrite=args.overwrite,
    )
    print(f"[prepare] revision={selected_revision} experiments written to {args.output}")


if __name__ == "__main__":
    main()
