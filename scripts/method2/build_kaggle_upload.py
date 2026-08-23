"""Dựng thư mục `kaggle_upload/` để đẩy lên Kaggle Dataset.

Ba dataset tách riêng vì vòng đời khác nhau: code đổi mỗi commit, data đổi khi
rebuild pairs, cache model gần như không đổi.

```
kaggle_upload/
├── toolcalling-vi-src/     src/ + configs/{method2,eval}/ + notebooks/
├── toolcalling-vi-data/    method2/ + custom_vi/v1/ + benchmark_vi/test.jsonl
└── hf-cache/               cache HuggingFace theo đúng layout HF_HUB_CACHE
```

Hai điểm dễ sai:

1. `data/method2/decontamination.json` nằm trong `.gitignore` nên **không** đi
   theo `git clone`. Thiếu nó thì Run 0 fail — đúng thiết kế, nhưng phải nhớ
   copy. Script này kiểm tra tường minh.
2. Cache HuggingFace có layout riêng: `hub/models--BAAI--bge-m3/...`, **không**
   phải `BAAI/bge-m3/`. Tầng `hub/` là bắt buộc để `HF_HOME` nhận ra —
   `snapshot_download(cache_dir=X)` đặt thẳng `models--*` vào `X` (đó là layout
   của `HF_HUB_CACHE`), nên script tự thêm `hub/`. Thiếu tầng này thì biến môi
   trường bị bỏ qua trong im lặng và model vẫn tải lại từ Hub mỗi session.

Sau khi dựng, script verify lại SHA-256 của mọi artefact đã copy với
`manifest.json` — bảo đảm cái upload lên đúng là cái đã kiểm định ở local.

Chạy:

```bash
python scripts/method2/build_kaggle_upload.py            # src + data
python scripts/method2/build_kaggle_upload.py --hf-cache # thêm ~3.4 GB model
```
"""

from __future__ import annotations

import argparse
import json
import operator
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

# Chạy trực tiếp bằng `python scripts/method2/...` nên phải tự thêm repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.sources import sha256_file  # noqa: E402

UPLOAD_ROOT = Path("kaggle_upload")

SRC_DATASET = "toolcalling-vi-src"
DATA_DATASET = "toolcalling-vi-data"
HF_DATASET = "toolcalling-vi-hf-cache"
WHEELS_DATASET = "toolcalling-vi-wheels"

#: (nguồn, đích tương đối trong dataset)
SRC_ITEMS: tuple[tuple[Path, str], ...] = (
    (Path("src"), "src"),
    (Path("configs/method2"), "configs/method2"),
    (Path("configs/eval"), "configs/eval"),
    (Path("notebooks"), "notebooks"),
)

DATA_ITEMS: tuple[tuple[Path, str], ...] = (
    (Path("data/method2"), "method2"),
    # Chỉ gold để đánh giá. `custom_vi/v1/train.jsonl` (38 MB) KHÔNG cần:
    # Method 2 train từ cặp đã sinh sẵn trong `data/method2/`, không đọc lại
    # dữ liệu thô.
    (Path("data/custom_vi/v1/test_seen.jsonl"), "custom_vi/v1/test_seen.jsonl"),
    (Path("data/custom_vi/v1/test_unseen.jsonl"), "custom_vi/v1/test_unseen.jsonl"),
    (Path("data/custom_vi/v1/val_seen.jsonl"), "custom_vi/v1/val_seen.jsonl"),
    (Path("data/custom_vi/v1/val_unseen.jsonl"), "custom_vi/v1/val_unseen.jsonl"),
    (Path("data/custom_vi/v1/tools.json"), "custom_vi/v1/tools.json"),
    (Path("data/benchmark_vi/test.jsonl"), "benchmark_vi/test.jsonl"),
)

#: Không mang lên Kaggle: cache Python và checkpoint HF cũ.
EXCLUDE_PATTERNS = ("__pycache__", ".pytest_cache", ".ipynb_checkpoints", "*.pyc")

HF_MODELS = ("BAAI/bge-m3", "xlm-roberta-base")

#: Package cần cài trên Kaggle. Wheel tải theo platform Linux để dùng được dù
#: máy dựng là Windows.
WHEEL_REQUIREMENTS = (
    "transformers==5.15.1",
    "sentence-transformers==6.0.0",
    "peft==0.20.0",
    "datasets==5.0.1",
    "jsonschema",
)
KAGGLE_PYTHON = "3.11"
LINUX_PLATFORMS = ("manylinux2014_x86_64", "manylinux_2_28_x86_64")

#: KHÔNG ship: nặng và Kaggle chắc chắn đã có bản dùng được. `torch` đặc biệt
#: nguy hiểm — thay bằng bản khác CUDA là hỏng GPU runtime của image.
ASSUME_PRESENT = {
    "torch", "numpy", "scipy", "scikit-learn", "sympy", "mpmath",
    "networkx", "jinja2", "markupsafe", "joblib", "threadpoolctl", "pandas",
}

#: Thiếu file này thì Run 0 fail — và nó không có trong git.
CRITICAL_FILES = ("method2/decontamination.json", "method2/manifest.json")


def _ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    for name in names:
        if name in EXCLUDE_PATTERNS or name.endswith(".pyc"):
            ignored.add(name)
    return ignored


def copy_items(items: Iterable[tuple[Path, str]], destination: Path) -> list[tuple[str, int]]:
    copied: list[tuple[str, int]] = []
    for source, relative in items:
        target = destination / relative
        if not source.exists():
            print(f"  THIẾU  {source}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target, ignore=_ignore)
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        else:
            shutil.copy2(source, target)
            size = target.stat().st_size
        copied.append((relative, size))
        print(f"  {size / 1024**2:8.2f} MB  {relative}")
    return copied


def verify_against_manifest(data_root: Path) -> list[str]:
    """SHA-256 của file đã copy phải khớp manifest — nếu không thì Run 0 sẽ fail."""
    manifest_path = data_root / "method2/manifest.json"
    if not manifest_path.exists():
        return ["thiếu method2/manifest.json"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, entry in (manifest.get("derived") or {}).items():
        if not entry.get("sha256"):
            continue
        # Khoá có thể dùng "\\" nếu manifest sinh trên Windows; đổi tường minh
        # vì Path.as_posix() không xử lý được backslash khi chạy trên Linux.
        # "data/method2/x.jsonl" trong repo → "method2/x.jsonl" trong dataset
        relative = str(name).replace("\\", "/").replace("data/", "", 1)
        staged = data_root / relative
        if not staged.exists():
            problems.append(f"thiếu {relative}")
            continue
        if sha256_file(staged) != entry["sha256"]:
            problems.append(f"SHA-256 lệch: {relative}")
    return problems


def download_hf_cache(destination: Path, models: Iterable[str] = HF_MODELS) -> None:
    """Tải model về đúng layout của `HF_HUB_CACHE`."""
    import os

    # Backend `xet` của huggingface_hub trả 404 trên xet-read-token ở một số
    # mạng/repo (đã gặp với xlm-roberta-base). Ép về đường tải HTTP thường.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    # `HF_HOME` tìm model trong `<HF_HOME>/hub/`, nên tải thẳng vào đó.
    hub = destination / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    for model in models:
        files = set(api.list_repo_files(model))
        # KHÔNG loại thẳng `pytorch_model.bin`: BAAI/bge-m3 chỉ có định dạng này,
        # loại đi thì cache tải xong mà không có trọng số nào — và lỗi chỉ lộ ra
        # lúc nạp model trên Kaggle, sau khi đã upload vài GB.
        has_safetensors = any(f.endswith(".safetensors") for f in files)
        ignore = ["*.h5", "*.ot", "*.msgpack", "*onnx*", "*.jpg", "*.png", "*.webp", ".DS_Store"]
        if has_safetensors:
            ignore.append("pytorch_model.bin")
        print(f"  tải {model} (safetensors={has_safetensors}) …", flush=True)
        snapshot_download(repo_id=model, cache_dir=str(hub), ignore_patterns=ignore)

    problems = verify_hf_cache(destination, models)
    for problem in problems:
        print(f"  LỖI: {problem}")
    size = sum(f.stat().st_size for f in destination.rglob("*") if f.is_file())
    print(f"  {size / 1024**3:.2f} GB tổng cache")
    if problems:
        raise SystemExit("Cache HuggingFace thiếu trọng số — đừng upload")


def verify_hf_cache(destination: Path, models: Iterable[str] = HF_MODELS) -> list[str]:
    """Mỗi model phải có ít nhất một file trọng số thật trong cache."""
    problems: list[str] = []
    for model in models:
        folder = destination / "hub" / ("models--" + model.replace("/", "--"))
        if not folder.exists():
            problems.append(f"{model}: không có thư mục cache")
            continue
        weights = [
            f
            for f in folder.rglob("*")
            if f.is_file() and f.suffix in (".safetensors", ".bin") and f.stat().st_size > 10**8
        ]
        if not weights:
            problems.append(f"{model}: không có file trọng số > 100 MB")
        else:
            total = sum(f.stat().st_size for f in weights) / 1024**3
            print(f"  {model}: {len(weights)} file trọng số, {total:.2f} GB")
    return problems


def build_wheelhouse(destination: Path) -> None:
    """Tải wheel cho Linux để cài offline trên Kaggle.

    Kaggle notebook có thể không bật internet, mà `transformers 5.x` cần
    `huggingface_hub 1.x` và `tokenizers 0.22.x` — bản có sẵn trong image gần
    như chắc chắn cũ hơn, nên `import transformers` sẽ vỡ nếu chỉ cài 3 wheel
    chính bằng `--no-deps`.

    Nhắm platform tường minh vì máy dựng là Windows: không có `--platform` thì
    pip tải wheel `win_amd64`, đem lên Linux là vô dụng.
    """
    import subprocess

    wheels = destination / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    wheels.mkdir(parents=True)
    staging = wheels.parent / ".download"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    command = [
        sys.executable, "-m", "pip", "download",
        "--only-binary=:all:",
        "--python-version", KAGGLE_PYTHON,
        "--implementation", "cp",
        "-d", str(staging),
    ]
    for platform in LINUX_PLATFORMS:
        command += ["--platform", platform]
    command += list(WHEEL_REQUIREMENTS)

    print(f"  tải closure cho Linux/py{KAGGLE_PYTHON} …", flush=True)
    subprocess.run(command, check=True, capture_output=True, text=True)

    kept = 0
    for whl in sorted(staging.glob("*.whl")):
        if _wheel_package(whl.name) in ASSUME_PRESENT:
            continue
        shutil.copy2(whl, wheels / whl.name)
        kept += 1
    shutil.rmtree(staging, ignore_errors=True)

    problems = verify_wheelhouse(wheels)
    for problem in problems:
        print(f"  LỖI: {problem}")
    size = sum(f.stat().st_size for f in wheels.glob("*.whl"))
    print(f"  {kept} wheel, {size / 1024**2:.1f} MB")
    if problems:
        raise SystemExit("Wheelhouse không dùng được — đừng upload")


def _wheel_package(filename: str) -> str:
    return re.split(r"-\d", filename, 1)[0].lower().replace("_", "-")


def _dependency_needed(requirement: str) -> bool:
    """Bỏ dep chỉ dành cho `extra` hoặc cho Python khác bản của Kaggle."""
    if ";" not in requirement:
        return True
    marker = requirement.split(";", 1)[1]
    if "extra ==" in marker:
        return False
    match = re.search(r'python_version\s*([<>=!]+)\s*["\']([\d.]+)["\']', marker)
    if not match:
        return True
    ops = {
        "<": operator.lt, "<=": operator.le, ">": operator.gt,
        ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
    }
    current = tuple(int(x) for x in KAGGLE_PYTHON.split("."))
    target = tuple(int(x) for x in match.group(2).split("."))
    return ops[match.group(1)](current, target)


def verify_wheelhouse(wheels: Path) -> list[str]:
    """Wheel phải chạy được trên Linux, và closure phải đóng.

    Thiếu một dep là `pip install --no-index` fail giữa chừng trên Kaggle, sau
    khi đã tốn công attach dataset và khởi động session.
    """
    import zipfile

    files = sorted(wheels.glob("*.whl"))
    if not files:
        return ["wheelhouse rỗng"]

    problems = [
        f"{f.name}: không phải wheel Linux"
        for f in files
        if "none-any" not in f.name and "manylinux" not in f.name
    ]
    have = {_wheel_package(f.name) for f in files}
    for wheel in files:
        with zipfile.ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            metadata = archive.read(name).decode("utf-8", "replace")
        for line in metadata.splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            requirement = line.split(":", 1)[1].strip()
            if not _dependency_needed(requirement):
                continue
            dep = re.split(r"[<>=!~;\[\s(]", requirement, 1)[0].lower().replace("_", "-")
            if dep not in have and dep not in ASSUME_PRESENT:
                problems.append(f"thiếu dependency {dep} (cần bởi {_wheel_package(wheel.name)})")
    return sorted(set(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng kaggle_upload/")
    parser.add_argument("--root", type=Path, default=UPLOAD_ROOT)
    parser.add_argument(
        "--hf-cache", action="store_true", help="Tải luôn model HuggingFace (~3.2 GB)"
    )
    parser.add_argument(
        "--wheels", action="store_true", help="Tải wheel Linux để cài offline (~79 MB)"
    )
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    src_root = args.root / SRC_DATASET
    data_root = args.root / DATA_DATASET

    print(f"=== {SRC_DATASET} ===")
    src_files = copy_items(SRC_ITEMS, src_root)
    print(f"=== {DATA_DATASET} ===")
    data_files = copy_items(DATA_ITEMS, data_root)

    print("=== kiểm tra file bắt buộc ===")
    missing = [name for name in CRITICAL_FILES if not (data_root / name).exists()]
    for name in CRITICAL_FILES:
        mark = "THIẾU" if name in missing else "OK   "
        print(f"  [{mark}] {name}")

    problems = [] if args.skip_verify else verify_against_manifest(data_root)
    if problems:
        print("=== SHA-256 KHÔNG KHỚP MANIFEST ===")
        for problem in problems:
            print(f"  {problem}")
    elif not args.skip_verify:
        print("=== SHA-256 khớp manifest: OK ===")

    if args.hf_cache:
        print(f"=== {HF_DATASET} ===")
        download_hf_cache(args.root / HF_DATASET)
    else:
        print(f"=== {HF_DATASET}: bỏ qua (dùng --hf-cache để tải) ===")

    if args.wheels:
        print(f"=== {WHEELS_DATASET} ===")
        build_wheelhouse(args.root / WHEELS_DATASET)
    else:
        print(f"=== {WHEELS_DATASET}: bỏ qua (dùng --wheels để tải) ===")

    total = sum(size for _, size in src_files + data_files)
    print(f"\nTổng src + data: {total / 1024**2:.1f} MB → {args.root}")

    if missing or problems:
        raise SystemExit("Chưa sẵn sàng upload — xem các dòng THIẾU / lệch ở trên")
    print("Sẵn sàng upload.")


if __name__ == "__main__":
    main()
