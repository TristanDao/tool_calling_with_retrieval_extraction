"""Atomic checkpoint for translation pipeline.

State is stored in a JSON file. Each save writes to a temp file then renames
atomically (POSIX guarantee) so a crash mid-write never corrupts the checkpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Checkpoint:
    dataset: str
    last_processed_index: int = 0
    total_success: int = 0
    total_failed: int = 0
    started_at: str = field(default_factory=_utc_now_iso)
    last_update: str = field(default_factory=_utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def advance(self, n_success: int, n_failed: int) -> None:
        self.last_processed_index += n_success + n_failed
        self.total_success += n_success
        self.total_failed += n_failed
        self.last_update = _utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            dataset=data.get("dataset", "unknown"),
            last_processed_index=int(data.get("last_processed_index", 0)),
            total_success=int(data.get("total_success", 0)),
            total_failed=int(data.get("total_failed", 0)),
            started_at=data.get("started_at", _utc_now_iso()),
            last_update=data.get("last_update", _utc_now_iso()),
            extra=data.get("extra", {}),
        )


def save_atomic(checkpoint: Checkpoint, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.to_dict()

    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load(path: str | Path, dataset: str = "unknown") -> Checkpoint:
    path = Path(path)
    if not path.exists():
        return Checkpoint(dataset=dataset)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return Checkpoint(dataset=dataset)
    if "dataset" not in data:
        data["dataset"] = dataset
    return Checkpoint.from_dict(data)
