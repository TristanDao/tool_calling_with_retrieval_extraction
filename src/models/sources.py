"""Khai báo nguồn dữ liệu cho Method 2 và loader dùng chung.

Method 2 đọc từ ba nơi, tất cả đã ở unified master structure
(`id`, `query`, `function_calls[]`, `tools[]`):

- `data/benchmark_vi/{train,val,test}.jsonl` — glaive + xLAM, positive.
- `data/translations/glaive_negative_vi.jsonl` — negative của glaive.
- `data/custom_vi/v1/{train,val_seen,val_unseen,test_seen,test_unseen}.jsonl`
  — CustomTools-VI, chứa cả positive lẫn negative.

Split logic của Method 2 (`train`/`val`/`test`) không trùng split của file:
`custom_vi` có sẵn seen/unseen, còn glaive_negative chưa được chia nên được
chia xác định theo hash của **query** (query ở đó lặp lại rất nhiều; chia theo
`id` sẽ khiến cùng một query nằm ở cả train lẫn test).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

BENCHMARK_DIR = Path("data/benchmark_vi")
CUSTOM_DIR = Path("data/custom_vi/v1")
GLAIVE_NEGATIVE_PATH = Path("data/translations/glaive_negative_vi.jsonl")

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"


@dataclass(frozen=True)
class SourceSpec:
    """Một file dữ liệu + vai trò của nó trong Method 2."""

    key: str
    path: Path
    split: str
    #: `seen` / `unseen` / `mixed` — dùng cho Bảng D (generalization).
    tool_split: str = "mixed"
    #: True khi file chỉ chứa sample no-call.
    negative_only: bool = False
    #: Chia xác định theo hash khi file chưa có split sẵn.
    hash_split: dict[str, float] = field(default_factory=dict)
    #: Field dùng làm khoá hash. `query` để mọi bản sao của cùng một query rơi
    #: vào cùng split — glaive_negative lặp lại query rất nhiều, chia theo `id`
    #: sẽ khiến cùng một query vừa ở train vừa ở test rồi bị dedupe loại sạch.
    hash_split_on: str = "id"


DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("benchmark_train", BENCHMARK_DIR / "train.jsonl", SPLIT_TRAIN),
    SourceSpec("benchmark_val", BENCHMARK_DIR / "val.jsonl", SPLIT_VAL),
    SourceSpec("benchmark_test", BENCHMARK_DIR / "test.jsonl", SPLIT_TEST),
    SourceSpec("custom_train", CUSTOM_DIR / "train.jsonl", SPLIT_TRAIN, tool_split="seen"),
    SourceSpec("custom_val_seen", CUSTOM_DIR / "val_seen.jsonl", SPLIT_VAL, tool_split="seen"),
    SourceSpec("custom_val_unseen", CUSTOM_DIR / "val_unseen.jsonl", SPLIT_VAL, tool_split="unseen"),
    SourceSpec("custom_test_seen", CUSTOM_DIR / "test_seen.jsonl", SPLIT_TEST, tool_split="seen"),
    SourceSpec("custom_test_unseen", CUSTOM_DIR / "test_unseen.jsonl", SPLIT_TEST, tool_split="unseen"),
    SourceSpec(
        "glaive_negative",
        GLAIVE_NEGATIVE_PATH,
        SPLIT_TRAIN,
        negative_only=True,
        hash_split={SPLIT_TRAIN: 0.8, SPLIT_VAL: 0.1, SPLIT_TEST: 0.1},
        hash_split_on="query",
    ),
)


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hash_bucket(key: str, ratios: dict[str, float]) -> str:
    """Gán split xác định theo hash — cùng `key` luôn ra cùng split."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    position = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for split, ratio in ratios.items():
        cumulative += ratio
        if position < cumulative:
            return split
    return list(ratios)[-1]


def iter_samples(
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    splits: set[str] | None = None,
    limit_per_source: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Duyệt sample của mọi nguồn, gắn `_source_key`, `_split`, `_tool_split`."""
    for spec in specs:
        if not spec.path.exists():
            continue
        for index, sample in enumerate(load_jsonl(spec.path)):
            if limit_per_source is not None and index >= limit_per_source:
                break
            split = (
                hash_bucket(str(sample.get(spec.hash_split_on, index)), spec.hash_split)
                if spec.hash_split
                else spec.split
            )
            if splits is not None and split not in splits:
                continue
            sample["_source_key"] = spec.key
            sample["_split"] = split
            sample["_tool_split"] = spec.tool_split
            sample["_negative_only"] = spec.negative_only
            yield sample


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES) -> dict[str, Any]:
    """Manifest SHA-256 để freeze snapshot dataset (§0.3 method2_plan)."""
    entries = {}
    for spec in specs:
        if not spec.path.exists():
            entries[spec.key] = {"path": str(spec.path), "present": False}
            continue
        entries[spec.key] = {
            "path": str(spec.path),
            "present": True,
            "bytes": spec.path.stat().st_size,
            "sha256": sha256_file(spec.path),
        }
    return entries


DERIVED_ARTIFACTS = (
    Path("data/method2/tool_pool.json"),
    Path("data/method2/biencoder/train.jsonl"),
    Path("data/method2/biencoder/val.jsonl"),
    Path("data/method2/biencoder/test.jsonl"),
    Path("data/method2/crossencoder/train.jsonl"),
    Path("data/method2/crossencoder/val.jsonl"),
    Path("data/method2/crossencoder/test.jsonl"),
)


def main() -> None:
    """Ghi `data/method2/manifest.json` — nguồn + artefact dẫn xuất + commit hash."""
    import argparse
    import subprocess
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Freeze dataset snapshot for Method 2")
    parser.add_argument("--output", type=Path, default=Path("data/method2/manifest.json"))
    args = parser.parse_args()

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None

    derived = {
        str(path): {
            "present": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
        }
        for path in DERIVED_ARTIFACTS
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "sources": build_manifest(),
        "derived": derived,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    n_present = sum(1 for e in manifest["sources"].values() if e.get("present"))
    print(f"[manifest] {n_present}/{len(manifest['sources'])} nguồn → {args.output}")


if __name__ == "__main__":
    main()
