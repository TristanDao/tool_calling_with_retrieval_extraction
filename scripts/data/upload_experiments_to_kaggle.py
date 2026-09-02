"""Upload materialized experiment data as a Kaggle Dataset."""

from __future__ import annotations

import argparse
import shutil
import tempfile
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


def _copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _build_upload_bundle(
    experiments: Path,
    benchmark_revision: Path | None,
    custom_data: Path | None,
    bundle: Path,
) -> Path:
    """Combine train artifacts and shared evaluation data without changing inputs."""
    if not experiments.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiments}")
    if benchmark_revision is not None and not benchmark_revision.is_dir():
        raise FileNotFoundError(f"Benchmark revision does not exist: {benchmark_revision}")
    if custom_data is not None and not custom_data.is_dir():
        raise FileNotFoundError(f"CustomTools directory does not exist: {custom_data}")
    if bundle.exists():
        raise FileExistsError(f"Bundle directory already exists: {bundle}")
    bundle.mkdir(parents=True)
    _copy_tree_contents(experiments, bundle)
    if benchmark_revision is not None:
        destination = bundle / "benchmark_core" / benchmark_revision.name
        shutil.copytree(benchmark_revision, destination)
    if custom_data is not None:
        shutil.copytree(custom_data, bundle / "custom_vi")
    return bundle


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
        "--benchmark-revision",
        type=Path,
        default=None,
        help="Optional frozen revision to include under benchmark_core/<revision>",
    )
    parser.add_argument(
        "--custom-data",
        type=Path,
        default=None,
        help="Optional data/custom_vi directory to include for shared evaluation",
    )
    parser.add_argument(
        "--version-notes",
        default="Method 1 E0-E4 controlled-track data, seed 42",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without uploading")
    args = parser.parse_args()

    if "/" not in args.handle or args.handle.count("/") != 1:
        parser.error("handle must have the form '<kaggle-username>/<dataset-slug>'")
    if not args.source.is_dir():
        parser.error(f"Dataset directory does not exist: {args.source}")
    for option, path in (
        ("--benchmark-revision", args.benchmark_revision),
        ("--custom-data", args.custom_data),
    ):
        if path is not None and not path.is_dir():
            parser.error(f"{option} directory does not exist: {path}")

    if args.dry_run:
        extras = [str(path) for path in (args.benchmark_revision, args.custom_data) if path]
        suffix = f" plus {', '.join(extras)}" if extras else ""
        print(f"[kaggle] dry-run: would upload {args.source}{suffix} to {args.handle}")
        return

    if args.benchmark_revision is None and args.custom_data is None:
        upload_dataset(args.handle, args.source, args.version_notes)
    else:
        with tempfile.TemporaryDirectory(prefix="tool-calling-vi-kaggle-") as temporary:
            bundle = _build_upload_bundle(
                args.source,
                args.benchmark_revision,
                args.custom_data,
                Path(temporary) / "dataset",
            )
            upload_dataset(args.handle, bundle, args.version_notes)
    print(f"[kaggle] uploaded {args.source} to {args.handle}")


if __name__ == "__main__":
    main()
