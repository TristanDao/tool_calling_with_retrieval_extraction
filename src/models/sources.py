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
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

BENCHMARK_DIR = Path("data/benchmark_vi")
CUSTOM_DIR = Path("data/custom_vi/v1")
GLAIVE_NEGATIVE_PATH = Path("data/translations/glaive_negative_vi.jsonl")

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

#: Thứ tự ưu tiên khi một query xuất hiện ở nhiều split: giữ ở split cao nhất.
#: `test` đứng đầu để tập test **không bao giờ mất sample** — bốn method phải
#: được đánh giá trên đúng cùng một tập; mọi decontamination dồn về val/train.
SPLIT_PRECEDENCE: tuple[str, ...] = (SPLIT_TEST, SPLIT_VAL, SPLIT_TRAIN)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query_key(query: Any) -> str:
    """Khoá so khớp query giữa các split: NFC + gộp whitespace + casefold.

    Dedupe theo chuỗi thô bỏ sót các cặp chỉ khác hoa/thường hoặc khoảng trắng —
    đo trên dữ liệu thật còn sót 9 query train↔val/test kiểu đó.
    """
    text = unicodedata.normalize("NFC", str(query))
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


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


def assign_split(spec: SourceSpec, sample: dict[str, Any], index: int) -> str:
    return (
        hash_bucket(str(sample.get(spec.hash_split_on, index)), spec.hash_split)
        if spec.hash_split
        else spec.split
    )


def iter_samples(
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    splits: set[str] | None = None,
    limit_per_source: int | None = None,
    decontamination: "DecontaminationIndex | None" = None,
) -> Iterator[dict[str, Any]]:
    """Duyệt sample của mọi nguồn, gắn `_source_key`, `_split`, `_tool_split`.

    Truyền `decontamination` để bỏ qua sample có query trùng với split cao hơn
    (xem `DecontaminationIndex`). Sample bị bỏ được đếm vào `index.dropped`.
    """
    for spec in specs:
        if not spec.path.exists():
            continue
        for index, sample in enumerate(load_jsonl(spec.path)):
            if limit_per_source is not None and index >= limit_per_source:
                break
            split = assign_split(spec, sample, index)
            if decontamination is not None:
                effective = decontamination.effective_split.get(
                    normalize_query_key(sample.get("query", ""))
                )
                if effective is not None and effective != split:
                    decontamination.dropped[f"{split}->{effective}"] += 1
                    decontamination.dropped_by_source[spec.key] += 1
                    continue
            if splits is not None and split not in splits:
                continue
            sample["_source_key"] = spec.key
            sample["_split"] = split
            sample["_tool_split"] = spec.tool_split
            sample["_negative_only"] = spec.negative_only
            yield sample


# --------------------------------------------------------- decontamination


@dataclass
class DecontaminationIndex:
    """Ánh xạ normalized query → split được phép giữ.

    Benchmark gốc (`data/benchmark_vi`) chia split theo **sample** chứ không
    theo query, nên cùng một query nằm ở nhiều split. Đo trên dữ liệu thật,
    1,718 query bị trùng: `test∩train` 574, `train∩val` 572,
    `test∩train∩val` 542, `test∩val` 30 — tức `val ∩ test` = 30 + 542 = 572.

    Overlap `val ∩ test` là rủi ro phương pháp luận nặng nhất: dù không train
    trên query đó, việc chọn checkpoint/hyperparameter bằng val vẫn làm metric
    test lạc quan lên. Index này giải quyết bằng cách gán mỗi query đúng **một**
    split theo `SPLIT_PRECEDENCE`.

    **Không sửa benchmark gốc.** File `data/benchmark_vi/*` giữ nguyên để tái
    lập được; decontamination chỉ diễn ra ở tầng dataset của Method 2 và số
    sample bị loại được ghi lại đầy đủ.
    """

    effective_split: dict[str, str]
    stats: dict[str, Any] = field(default_factory=dict)
    dropped: Counter = field(default_factory=Counter)
    dropped_by_source: Counter = field(default_factory=Counter)

    def report(self) -> dict[str, Any]:
        return {
            **self.stats,
            "rows_dropped_by_transition": dict(self.dropped.most_common()),
            "rows_dropped_by_source": dict(self.dropped_by_source.most_common()),
            "rows_dropped_total": sum(self.dropped.values()),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"effective_split": self.effective_split, "stats": self.stats},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "DecontaminationIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(effective_split=data["effective_split"], stats=data.get("stats", {}))


def build_decontamination_index(
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    limit_per_source: int | None = None,
) -> DecontaminationIndex:
    """Quét một lượt toàn bộ nguồn, gán mỗi normalized query đúng một split."""
    splits_of_query: dict[str, set[str]] = defaultdict(set)
    for spec in specs:
        if not spec.path.exists():
            continue
        for index, sample in enumerate(load_jsonl(spec.path)):
            if limit_per_source is not None and index >= limit_per_source:
                break
            key = normalize_query_key(sample.get("query", ""))
            if key:
                splits_of_query[key].add(assign_split(spec, sample, index))

    effective: dict[str, str] = {}
    overlaps: Counter = Counter()
    unique_before: Counter = Counter()
    for key, found in splits_of_query.items():
        for split in found:
            unique_before[split] += 1
        for split in SPLIT_PRECEDENCE:
            if split in found:
                effective[key] = split
                break
        if len(found) > 1:
            overlaps["∩".join(sorted(found))] += 1

    unique_after = Counter(effective.values())
    stats = {
        "n_unique_queries": len(effective),
        "unique_queries_per_split_before": dict(unique_before),
        "unique_queries_per_split_after": dict(unique_after),
        "overlapping_queries": dict(overlaps.most_common()),
        "n_overlapping_queries": sum(overlaps.values()),
        "precedence": list(SPLIT_PRECEDENCE),
    }
    return DecontaminationIndex(effective_split=effective, stats=stats)


DECONTAMINATION_PATH = Path("data/method2/decontamination.json")


class MissingDecontaminationIndex(FileNotFoundError):
    """Index bắt buộc nhưng không có — dừng job thay vì âm thầm build lại."""


def load_decontamination(path: str | Path = DECONTAMINATION_PATH) -> DecontaminationIndex:
    """Nạp index, **fail-closed** nếu thiếu.

    Không tự build lại: rebuild là một bước preprocessing riêng
    (`python -m src.models.sources decontaminate`). Nếu experiment chính tự
    dựng lại index từ dữ liệu đang có trên máy, ta mất đúng thứ cần đảm bảo —
    bằng chứng rằng model được train trên đúng split đã kiểm định.

    Bi-Encoder, Cross-Encoder và Method 1 **phải** dùng chung một index; hai
    pipeline chia split khác nhau thì val của bên này là test của bên kia.
    """
    path = Path(path)
    if not path.exists():
        raise MissingDecontaminationIndex(
            f"Thiếu decontamination index: {path}\n"
            f"Đây là artifact bắt buộc, không được build lại trong job training.\n"
            f"Chạy bước preprocessing: python -m src.models.sources decontaminate"
        )
    return DecontaminationIndex.load(path)


def load_or_build_decontamination(
    path: str | Path = DECONTAMINATION_PATH,
    specs: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    limit_per_source: int | None = None,
    allow_build: bool = False,
) -> DecontaminationIndex:
    """Nạp index; chỉ build lại khi `allow_build=True` (bước preprocessing).

    `limit_per_source` là chế độ smoke test trên tập con nên luôn build tại chỗ
    và **không** ghi đè index thật.
    """
    path = Path(path)
    if limit_per_source is not None:
        return build_decontamination_index(specs, limit_per_source)
    if path.exists():
        return DecontaminationIndex.load(path)
    if not allow_build:
        raise MissingDecontaminationIndex(
            f"Thiếu decontamination index: {path}\n"
            f"Chạy: python -m src.models.sources decontaminate"
        )
    index = build_decontamination_index(specs)
    index.save(path)
    return index


def pairwise_overlap(queries_per_split: dict[str, set[str]]) -> dict[str, int]:
    """Số query chung giữa từng cặp split — dùng để assert sau khi build pairs."""
    result: dict[str, int] = {}
    names = sorted(queries_per_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            result[f"{left}∩{right}"] = len(
                queries_per_split[left] & queries_per_split[right]
            )
    return result


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
    Path("data/method2/decontamination.json"),
    Path("data/method2/biencoder/train.jsonl"),
    Path("data/method2/biencoder/val.jsonl"),
    Path("data/method2/biencoder/test.jsonl"),
    Path("data/method2/crossencoder/train.jsonl"),
    Path("data/method2/crossencoder/val.jsonl"),
    Path("data/method2/crossencoder/test.jsonl"),
)


def main() -> None:
    """Hai lệnh preprocessing: `manifest` và `decontaminate`.

    Rebuild index tách hẳn khỏi job training để experiment chính fail-closed.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Method 2 dataset preprocessing")
    parser.add_argument(
        "command",
        nargs="?",
        default="manifest",
        choices=["manifest", "decontaminate"],
        help="manifest = freeze SHA-256 snapshot; decontaminate = build lại index",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "decontaminate":
        _run_decontaminate(args.output or DECONTAMINATION_PATH)
        return
    _run_manifest(args.output or Path("data/method2/manifest.json"))


def _run_decontaminate(output: Path) -> None:
    index = build_decontamination_index()
    index.save(output)
    stats = index.stats
    print(f"[decontaminate] {stats['n_unique_queries']} query duy nhất → {output}")
    print(f"[decontaminate] query trùng split: {stats['n_overlapping_queries']}")
    for pair, count in stats["overlapping_queries"].items():
        print(f"[decontaminate]   {pair}: {count}")
    print(f"[decontaminate] unique query/split trước: {stats['unique_queries_per_split_before']}")
    print(f"[decontaminate] unique query/split sau:   {stats['unique_queries_per_split_after']}")
    print("[decontaminate] Chạy lại `python -m src.models.sources manifest` để cập nhật SHA-256.")


def _run_manifest(output: Path) -> None:
    import subprocess
    from datetime import datetime, timezone

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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    n_present = sum(1 for e in manifest["sources"].values() if e.get("present"))
    print(f"[manifest] {n_present}/{len(manifest['sources'])} nguồn → {output}")


if __name__ == "__main__":
    main()
