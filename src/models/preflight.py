"""Cổng fail-closed trước mọi job training của Method 2 (Run 0).

Chuỗi kiểm tra, dừng ngay tại bước đầu tiên không đạt:

```
decontamination.json tồn tại
        ↓
SHA-256 == manifest.json
        ↓
overlap train/val/test == 0
        ↓
unseen positive leakage == 0
        ↓
package versions khớp bản đã pin
        ↓
CHO PHÉP TRAIN
```

Nguyên tắc: **không tự rebuild bất cứ thứ gì**. Nếu experiment chính tự dựng lại
index từ dữ liệu đang có trên máy thì ta mất đúng thứ cần đảm bảo — bằng chứng
rằng model được train trên đúng split đã kiểm định. Rebuild là một lệnh
preprocessing riêng: `python -m src.models.sources decontaminate`.

Exit code: 0 = pass, 1 = có check fail.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.sources import DECONTAMINATION_PATH, sha256_file

MANIFEST_PATH = Path("data/method2/manifest.json")
PAIRS_STATS_PATH = Path("data/method2/biencoder/pairs_stats.json")
LABEL_STATS_PATH = Path("data/method2/label_stats.json")
PINNED_VERSIONS_PATH = Path("configs/method2/pinned_versions.json")

#: Bốn artefact bắt buộc phải khớp SHA-256 với manifest trước khi train.
REQUIRED_ARTIFACTS = (
    "data/method2/decontamination.json",
    "data/method2/tool_pool.json",
    "data/method2/biencoder/train.jsonl",
    "data/method2/crossencoder/train.jsonl",
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = None

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: Any = None) -> Check:
        check = Check(name, passed, detail)
        self.checks.append(check)
        return check

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------- các check


def check_artifacts_match_manifest(
    report: PreflightReport,
    manifest_path: Path = MANIFEST_PATH,
    decontamination_path: Path = DECONTAMINATION_PATH,
) -> bool:
    """Tồn tại + SHA-256 khớp manifest. Đây là mắt xích fail-closed chính."""
    manifest = _read_json(manifest_path)
    if manifest is None:
        report.add("manifest tồn tại", False, f"thiếu {manifest_path}")
        return False
    report.add("manifest tồn tại", True, str(manifest_path))

    if not Path(decontamination_path).exists():
        report.add(
            "decontamination.json tồn tại",
            False,
            f"thiếu {decontamination_path} — chạy `python -m src.models.sources decontaminate` "
            f"như bước preprocessing riêng, KHÔNG rebuild trong job training",
        )
        return False
    report.add("decontamination.json tồn tại", True, str(decontamination_path))

    derived = manifest.get("derived") or {}
    normalized = {Path(k).as_posix(): v for k, v in derived.items()}
    all_ok = True
    for artifact in REQUIRED_ARTIFACTS:
        entry = normalized.get(artifact)
        path = Path(artifact)
        if entry is None or not entry.get("sha256"):
            report.add(f"SHA-256 {artifact}", False, "không có trong manifest")
            all_ok = False
            continue
        if not path.exists():
            report.add(f"SHA-256 {artifact}", False, "file không tồn tại")
            all_ok = False
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            report.add(
                f"SHA-256 {artifact}",
                False,
                f"lệch — manifest {entry['sha256'][:16]}… nhưng file {actual[:16]}…",
            )
            all_ok = False
        else:
            report.add(f"SHA-256 {artifact}", True, actual[:16] + "…")
    return all_ok


def check_split_integrity(
    report: PreflightReport,
    pairs_stats_path: Path = PAIRS_STATS_PATH,
    label_stats_path: Path = LABEL_STATS_PATH,
) -> bool:
    """Overlap ba cặp == 0, và không có tool unseen làm positive ở train."""
    ok = True
    for name, path in (("Bi-Encoder", pairs_stats_path), ("Cross-Encoder", label_stats_path)):
        stats = _read_json(path)
        if stats is None:
            report.add(f"overlap {name}", False, f"thiếu {path}")
            ok = False
            continue
        overlap = stats.get("split_overlap_after")
        if overlap is None:
            report.add(f"overlap {name}", False, "stats không có split_overlap_after")
            ok = False
            continue
        dirty = {pair: n for pair, n in overlap.items() if n}
        report.add(f"overlap {name} == 0", not dirty, dirty or overlap)
        ok &= not dirty

    pairs_stats = _read_json(pairs_stats_path) or {}
    leaked = pairs_stats.get("unseen_tools_leaked_into_train_positives")
    if leaked is None:
        report.add("unseen positive leakage == 0", False, "không có trong pairs_stats")
        return False
    report.add("unseen positive leakage == 0", not leaked, leaked or "0 tool")
    return ok and not leaked


def check_two_stages_agree(
    report: PreflightReport,
    pairs_stats_path: Path = PAIRS_STATS_PATH,
    label_stats_path: Path = LABEL_STATS_PATH,
) -> bool:
    """Bi-Encoder và Cross-Encoder phải chia split y hệt nhau."""
    bi = _read_json(pairs_stats_path) or {}
    cross = _read_json(label_stats_path) or {}
    bi_decon = (bi.get("decontamination") or {}).get("overlapping_queries")
    cross_decon = (cross.get("decontamination") or {}).get("overlapping_queries")
    if bi_decon is None or cross_decon is None:
        report.add("hai stage dùng chung index", False, "thiếu thống kê decontamination")
        return False
    same = bi_decon == cross_decon
    report.add(
        "hai stage dùng chung index",
        same,
        "khớp" if same else f"Bi {bi_decon} vs Cross {cross_decon}",
    )
    return same


def check_versions(
    report: PreflightReport,
    pinned_path: Path = PINNED_VERSIONS_PATH,
    strict: bool = True,
) -> bool:
    """Package quyết định API và tên metric phải đúng bản đã pin."""
    pinned = _read_json(pinned_path)
    if pinned is None:
        report.add("pinned_versions.json tồn tại", False, f"thiếu {pinned_path}")
        return False

    installed = installed_versions()
    ok = True
    for package, expected in (pinned.get("pinned") or {}).items():
        actual = installed.get(package)
        if actual is None:
            report.add(f"version {package}", False, "chưa cài")
            ok = False
            continue
        match = actual == expected
        # torch được ghi nhận chứ không ép: Kaggle cài sẵn bản CUDA riêng, ép cài
        # lại vừa chậm vừa dễ lệch CUDA runtime.
        if not match and not strict:
            report.add(f"version {package}", True, f"{actual} (pinned {expected}, không ép)")
            continue
        report.add(f"version {package}", match, f"{actual}" + ("" if match else f" ≠ {expected}"))
        ok &= match

    for package, actual in (pinned.get("recorded") or {}).items():
        report.add(
            f"version {package} (ghi nhận)", True, f"{installed.get(package)} (baseline {actual})"
        )
    return ok


def installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name, module_name in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence-transformers", "sentence_transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
    ):
        try:
            module = __import__(module_name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def check_gpu(report: PreflightReport, expect: str | None = None) -> bool:
    try:
        import torch
    except ImportError:
        report.add("GPU khả dụng", False, "chưa cài torch")
        return False
    if not torch.cuda.is_available():
        report.add("GPU khả dụng", False, "torch.cuda.is_available() = False")
        return False
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    report.add("GPU khả dụng", True, f"{name}, {total_gb:.1f} GB")
    if expect:
        match = expect.lower() in name.lower()
        report.add(f"GPU là {expect}", match, name)
        return match
    return True


def check_config(report: PreflightReport, config_path: Path) -> bool:
    if not config_path.exists():
        report.add("config tồn tại", False, f"thiếu {config_path}")
        return False
    try:
        import yaml

        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 — báo đúng lỗi parse cho người chạy
        report.add("config parse được", False, str(error))
        return False
    report.add("config parse được", True, f"{config_path} ({sha256_file(config_path)[:16]}…)")
    return True


def check_git(report: PreflightReport, allow_dirty: bool = True) -> bool:
    from src.models.run_manifest import git_state

    state = git_state()
    if not state["commit"]:
        report.add("commit SHA", False, "không đọc được git")
        return False
    report.add("commit SHA", True, f"{state['commit'][:12]} ({state['branch']})")
    if state["dirty"] and not allow_dirty:
        report.add("working tree sạch", False, f"{len(state['dirty_files'])} file thay đổi")
        return False
    report.add("working tree sạch", True, "sạch" if not state["dirty"] else "dirty (được phép)")
    return True


# ------------------------------------------------------------------- runner


def run_preflight(
    config_path: Path,
    require_gpu: str | None = None,
    strict_versions: bool = True,
    allow_dirty: bool = True,
) -> PreflightReport:
    report = PreflightReport()
    check_git(report, allow_dirty=allow_dirty)
    check_config(report, config_path)

    # Fail-closed: hỏng ở đây thì các check sau vô nghĩa.
    if not check_artifacts_match_manifest(report):
        report.add(
            "DỪNG",
            False,
            "artefact thiếu hoặc SHA-256 lệch — không được train, cũng không được rebuild tự động",
        )
        return report

    check_split_integrity(report)
    check_two_stages_agree(report)
    check_versions(report, strict=strict_versions)
    if require_gpu is not None:
        check_gpu(report, expect=require_gpu or None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed preflight cho Method 2")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/biencoder.yaml"))
    parser.add_argument(
        "--require-gpu",
        nargs="?",
        const="",
        default=None,
        metavar="TÊN",
        help="Bắt buộc có GPU; kèm tên (vd T4) để kiểm tra đúng loại",
    )
    parser.add_argument("--no-strict-versions", action="store_true")
    parser.add_argument("--require-clean-tree", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = run_preflight(
        args.config,
        require_gpu=args.require_gpu,
        strict_versions=not args.no_strict_versions,
        allow_dirty=not args.require_clean_tree,
    )
    for check in report.checks:
        print(check.line())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[preflight] → {args.output}")

    if report.passed:
        print("\n[preflight] TẤT CẢ ĐẠT — cho phép train")
        raise SystemExit(0)
    print(f"\n[preflight] KHÔNG ĐẠT: {report.failures}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
