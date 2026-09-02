"""Archive generated pilot artifacts before a benchmark rebuild."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("data")
DEFAULT_TARGETS = ("benchmark_vi", "experiments", "statistics")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path, target: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not target.exists():
        return entries
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def archive_generated(
    root: Path = DEFAULT_ROOT,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    archive_root: Path | None = None,
    manifest_path: Path | None = None,
    apply: bool = False,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Inventory and optionally move generated targets into an immutable archive."""
    root = root.resolve()
    if archive_root is None:
        archive_root = root / "legacy"
    else:
        archive_root = archive_root.resolve()
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = archive_root / f"pilot_{stamp}"
    entries: list[dict[str, Any]] = []
    existing_targets = [root / target for target in targets if (root / target).exists()]
    for target in existing_targets:
        entries.extend(_inventory(root, target))
    report: dict[str, Any] = {
        "schema_version": "cleanup-manifest-v1",
        "mode": "apply" if apply else "dry-run",
        "archive": str(archive_dir),
        "targets": list(targets),
        "reason": "Archive generated pilot artifacts before creating the frozen benchmark revision.",
        "policy": "Preserve raw, normalized, translation, and CustomTools source data.",
        "files": entries,
        "file_count": len(entries),
        "bytes": sum(int(entry["bytes"]) for entry in entries),
    }
    if not apply:
        return report

    if archive_dir.exists():
        raise FileExistsError(f"Archive already exists: {archive_dir}")
    archive_dir.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for target in existing_targets:
        destination = archive_dir / target.name
        shutil.move(str(target), str(destination))
        moved.append(str(destination.relative_to(root)))
    report["moved"] = moved
    if manifest_path is None:
        manifest_path = archive_dir / "cleanup_manifest.json"
    report["manifest_path"] = str(manifest_path)
    _write_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--apply", action="store_true", help="Move targets into the archive")
    args = parser.parse_args()
    report = archive_generated(
        root=args.root,
        archive_root=args.archive_root,
        manifest_path=args.manifest,
        apply=args.apply,
        timestamp=args.timestamp,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
