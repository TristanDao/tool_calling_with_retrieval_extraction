"""Gom toàn bộ artifact audit của một run vào `run_manifest.json`.

Nguyên tắc: **train xong mà không audit được thì coi như chưa train**. Mỗi run
phải trả lời được "chạy trên code nào, dữ liệu nào, config nào, ra số gì" mà
không cần mở lại notebook.

Nội dung, khớp danh sách chốt trước khi chạy Kaggle:

| Mục | Lấy từ |
|---|---|
| commit SHA (kèm cờ dirty) | `git rev-parse` / `git status --porcelain` |
| config YAML thực tế | đọc nguyên văn + SHA-256 file config |
| fingerprint dataset & tool pool | `data/method2/manifest.json`, `tool_pool_stats.json` |
| train/val/test query counts + overlap | `pairs_stats.json` (`unique_queries_per_split`, `split_overlap_after`) |
| positive/negative pairs | `pairs_stats.json` |
| model / checkpoint | `train_report.json` |
| best epoch + metric chọn checkpoint | `train_report.json::checkpoint_selection` |
| VRAM peak, thời lượng train | `train_report.json` |
| Recall@1/@5/@10, MRR | báo cáo eval truyền vào `--report retrieval=...` |

Chạy:

```bash
python -m src.models.run_manifest \
    --run-dir artifacts/method2/biencoder/run02 \
    --config configs/method2/biencoder.yaml \
    --stage biencoder \
    --report retrieval=results/method2/metrics/biencoder_val.json
```
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models.sources import sha256_file

DEFAULT_DATA_REPORTS: dict[str, Path] = {
    "dataset_manifest": Path("data/method2/manifest.json"),
    "tool_pool_stats": Path("data/method2/tool_pool_stats.json"),
    "biencoder_pairs_stats": Path("data/method2/biencoder/pairs_stats.json"),
    "crossencoder_label_stats": Path("data/method2/label_stats.json"),
}

#: Metric truy hồi bắt buộc phải có trước khi chuyển sang Cross-Encoder.
REQUIRED_RETRIEVAL_METRICS = ("micro_recall@1", "micro_recall@5", "micro_recall@10", "mrr")


def git_state() -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    dirty = _run("git", "status", "--porcelain")
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": dirty.splitlines()[:20] if dirty else [],
    }


def resolve_git_state(manifest_path: Path) -> dict[str, Any]:
    """Commit của snapshot, lấy từ git nếu có, không thì từ `manifest.json`.

    Kaggle chạy từ dataset đã copy nên không có thư mục `.git`. Yêu cầu audit là
    **biết code nào sinh ra artefact này**, không phải là phải có repo git tại
    chỗ chạy — mà commit đó đã được ghi vào manifest lúc build ở local.
    `preflight.check_git` đã rơi về cùng nguồn này; thiếu ở đây thì
    `audit_complete.missing` báo `git_commit` và chặn nhầm một run hợp lệ.
    """
    state = git_state()
    if state.get("commit"):
        state["commit_source"] = "git"
        return state

    manifest = _read_json(manifest_path)
    commit = manifest.get("git_commit") if isinstance(manifest, dict) else None
    return {
        "commit": commit,
        "branch": None,
        "dirty": False,
        "dirty_files": [],
        "commit_source": "dataset manifest" if commit else None,
    }


def environment_state() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_total_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 1
            )
    except ImportError:
        info["torch"] = None
    try:
        import sentence_transformers

        info["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def _read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return {"present": False, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def split_audit(pairs_stats: Any) -> dict[str, Any]:
    """Trích phần split/leakage và tự kết luận đạt hay không."""
    if not isinstance(pairs_stats, dict) or "split_overlap_after" not in pairs_stats:
        return {"available": False}

    overlap = pairs_stats["split_overlap_after"]
    decontamination = pairs_stats.get("decontamination", {})
    failures = {pair: n for pair, n in overlap.items() if n}
    return {
        "available": True,
        "unique_queries_per_split": pairs_stats.get("unique_queries_per_split"),
        "overlap_after_decontamination": overlap,
        "overlap_clean": not failures,
        "overlap_failures": failures,
        "queries_overlapping_before": decontamination.get("n_overlapping_queries"),
        "overlapping_pairs_before": decontamination.get("overlapping_queries"),
        "rows_dropped_total": decontamination.get("rows_dropped_total"),
        "rows_dropped_by_transition": decontamination.get("rows_dropped_by_transition"),
        "positive_pairs": pairs_stats.get("n_positive_pairs"),
        "negative_samples": pairs_stats.get("n_negative_samples"),
    }


def retrieval_gate(report: Any) -> dict[str, Any]:
    """Kiểm tra đã có đủ Recall@1/@5/@10 + MRR chưa (gate trước Cross-Encoder)."""
    metrics = report.get("overall") if isinstance(report, dict) else None
    if not isinstance(metrics, dict):
        return {"available": False, "missing": list(REQUIRED_RETRIEVAL_METRICS)}
    missing = [m for m in REQUIRED_RETRIEVAL_METRICS if m not in metrics]
    return {
        "available": not missing,
        "missing": missing,
        "metrics": {m: metrics.get(m) for m in REQUIRED_RETRIEVAL_METRICS},
    }


def build_run_manifest(
    run_dir: str | Path,
    config_path: str | Path,
    stage: str,
    reports: dict[str, Path] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config_path = Path(config_path)
    reports = reports or {}

    train_report = _read_json(run_dir / "train_report.json")
    pairs_stats = _read_json(DEFAULT_DATA_REPORTS["biencoder_pairs_stats"])

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "run_dir": str(run_dir),
        "git": resolve_git_state(DEFAULT_DATA_REPORTS["dataset_manifest"]),
        "environment": environment_state(),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path) if config_path.exists() else None,
            "resolved": (
                config_path.read_text(encoding="utf-8") if config_path.exists() else None
            ),
        },
        "data": {key: _read_json(path) for key, path in DEFAULT_DATA_REPORTS.items()},
        "split_audit": split_audit(pairs_stats),
        "train": {
            "final_checkpoint": train_report.get("final_checkpoint"),
            "duration_hours": train_report.get("training_duration_hours"),
            "peak_vram_mb": train_report.get("peak_vram_mb"),
            "n_train_pairs": train_report.get("n_train_pairs"),
            "checkpoint_selection": train_report.get("checkpoint_selection"),
            "final_metrics": train_report.get("final_metrics"),
        },
        "reports": {name: _read_json(path) for name, path in reports.items()},
    }

    if "retrieval" in manifest["reports"]:
        manifest["retrieval_gate"] = retrieval_gate(manifest["reports"]["retrieval"])

    manifest["audit_complete"] = _audit_complete(manifest)
    return manifest


def _audit_complete(manifest: dict[str, Any]) -> dict[str, Any]:
    """Liệt kê mục nào còn thiếu — để không train xong mới phát hiện thiếu."""
    checks = {
        "git_commit": bool(manifest["git"]["commit"]),
        "config_resolved": manifest["config"]["resolved"] is not None,
        "dataset_fingerprint": isinstance(manifest["data"]["dataset_manifest"], dict)
        and "sources" in manifest["data"]["dataset_manifest"],
        "tool_pool_fingerprint": isinstance(manifest["data"]["tool_pool_stats"], dict)
        and "n_tools" in manifest["data"]["tool_pool_stats"],
        "split_counts_and_overlap": manifest["split_audit"].get("available", False),
        "split_overlap_clean": manifest["split_audit"].get("overlap_clean", False),
        "pair_counts": manifest["split_audit"].get("positive_pairs") is not None,
        "checkpoint": bool(manifest["train"]["final_checkpoint"]),
        "checkpoint_selection": bool(manifest["train"]["checkpoint_selection"]),
        "peak_vram": manifest["train"]["peak_vram_mb"] is not None,
        "duration": manifest["train"]["duration_hours"] is not None,
        "retrieval_metrics": manifest.get("retrieval_gate", {}).get("available", False),
    }
    return {"checks": checks, "missing": sorted(k for k, ok in checks.items() if not ok)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build run manifest for a Method 2 run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", type=str, default="biencoder")
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Báo cáo bổ sung, vd retrieval=results/method2/metrics/biencoder_val.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reports: dict[str, Path] = {}
    for item in args.report:
        name, _, path = item.partition("=")
        if path:
            reports[name] = Path(path)

    manifest = build_run_manifest(args.run_dir, args.config, args.stage, reports)
    output = args.output or (args.run_dir / "run_manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    missing = manifest["audit_complete"]["missing"]
    print(f"[run_manifest] → {output}")
    print(f"[run_manifest] commit={manifest['git']['commit']} dirty={manifest['git']['dirty']}")
    audit = manifest["split_audit"]
    if audit.get("available"):
        print(f"[run_manifest] overlap sau decontamination: {audit['overlap_after_decontamination']}")
    if missing:
        print(f"[run_manifest] CHƯA ĐỦ: {missing}")
    else:
        print("[run_manifest] OK: đủ toàn bộ mục audit")


if __name__ == "__main__":
    main()
