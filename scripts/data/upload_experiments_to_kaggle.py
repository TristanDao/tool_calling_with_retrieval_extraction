"""Upload materialized experiment data as a Kaggle Dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def upload_dataset(handle: str, source: Path, version_notes: str) -> None:
    """Upload ``source`` to the Kaggle Dataset identified by ``handle``."""
    if "/" not in handle or handle.count("/") != 1:
        raise ValueError("handle must have the form '<kaggle-username>/<dataset-slug>'")
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source}")

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Install the dependency first: pip install kagglehub") from exc

    kagglehub.dataset_upload(
        handle,
        str(source),
        version_notes=version_notes,
        ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "handle",
        help="Kaggle Dataset handle, for example username/tool-calling-vi-experiments",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/experiments"),
        help="Folder to upload (default: data/experiments)",
    )
    parser.add_argument(
        "--version-notes",
        default="Method 1 E0-E5 controlled-track data, seed 42",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without uploading")
    args = parser.parse_args()

    if "/" not in args.handle or args.handle.count("/") != 1:
        parser.error("handle must have the form '<kaggle-username>/<dataset-slug>'")
    if not args.source.is_dir():
        parser.error(f"Dataset directory does not exist: {args.source}")

    if args.dry_run:
        print(f"[kaggle] dry-run: would upload {args.source} to {args.handle}")
        return

    upload_dataset(args.handle, args.source, args.version_notes)
    print(f"[kaggle] uploaded {args.source} to {args.handle}")


if __name__ == "__main__":
    main()
