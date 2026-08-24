"""Đo throughput Bi-Encoder theo lưới cấu hình, mỗi cấu hình vài step.

Vì sao cần: smoke run đầu tiên trên Kaggle cho **475 s/step** — 100 step mất
13.2 giờ và một epoch mất 49 giờ, trong khi plan dự toán 50-70 phút/epoch.
Lệch ~45×, nên phải tìm cấu hình dùng được trước khi tiêu quota T4.

Vì sao mỗi step lại đắt đến vậy: `CachedMultipleNegativesRankingLoss` không
phải một forward/backward bình thường. Effective batch 256, mỗi sample có
anchor + positive + 4 negative → 1,536 lượt encode. Chia mini_batch 8 thành 192
chunk, và GradCache chạy **hai** pha (forward no-grad để cache, rồi
forward+backward tính lại) → ~384 lần gọi model mỗi step. Mỗi lần chỉ 8×192 =
1,536 token, quá nhỏ để lấp đầy T4 nên phần lớn thời gian là overhead. Cộng
thêm DataParallel scatter/gather giữa 2 GPU thì nhân lên tiếp.

Ba biến cần thử, theo thứ tự tác động giảm dần:

1. **Số GPU** — `--single-gpu`. sentence-transformers tự bọc DataParallel khi
   thấy nhiều GPU; với GradCache gọi model hàng trăm lần/step thì phí đồng bộ
   cộng dồn rất nhanh, mà plan vốn thiết kế cho 1×T4.
2. **mini_batch_size** — 8 → 16 → 32 giảm số lần gọi model xuống 1/2, 1/4,
   trong khi effective batch (tức số in-batch negative của MNRL) không đổi.
3. **effective batch** — chỉ giảm khi hai cái trên không đủ, vì nó **đổi chất
   lượng** chứ không chỉ tốc độ: MNRL mạnh lên theo số negative trong batch.

`gradient_checkpointing` để cuối: tắt được thì nhanh thêm, nhưng phải còn VRAM.

Chạy:

```bash
python scripts/method2/benchmark_biencoder.py --steps 5
python scripts/method2/benchmark_biencoder.py --steps 5 --only A,B,C
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class Case:
    name: str
    single_gpu: bool
    batch_size: int
    mini_batch_size: int
    grad_checkpointing: bool
    note: str


#: Mỗi case đổi đúng MỘT biến so với case trước, để quy trách nhiệm được.
GRID: tuple[Case, ...] = (
    Case("A", True, 256, 8, True, "gốc, nhưng ép 1 GPU — tách ảnh hưởng DataParallel"),
    Case("B", True, 256, 16, True, "nửa số lần gọi model, giữ nguyên chất lượng"),
    Case("C", True, 256, 32, True, "1/4 số lần gọi model"),
    Case("D", True, 128, 32, True, "giảm effective batch — ĐỔI chất lượng, chỉ khi cần"),
    Case("E", True, 256, 32, False, "tắt grad checkpointing, cần còn VRAM"),
)


def run_case(case: Case, steps: int, config: Path, output_root: Path) -> dict:
    output_dir = output_root / f"bench_{case.name}"
    shutil.rmtree(output_dir, ignore_errors=True)

    command = [
        sys.executable, "-m", "src.models.biencoder.train", "train",
        "--config", str(config),
        "--output-dir", str(output_dir),
        "--smoke", str(steps),
        "--batch-size", str(case.batch_size),
        "--mini-batch-size", str(case.mini_batch_size),
        # Eval trên corpus 4,4k tool làm nhiễu số đo mà không liên quan tốc độ train.
        "--no-eval",
        "--grad-checkpointing" if case.grad_checkpointing else "--no-grad-checkpointing",
    ]
    if case.single_gpu:
        command.append("--single-gpu")

    print(f"\n=== {case.name}: {case.note} ===", flush=True)
    print("   " + " ".join(command[2:]), flush=True)
    result = subprocess.run(command, capture_output=True, text=True)

    report_path = output_dir / "train_report.json"
    if result.returncode != 0 or not report_path.exists():
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-4:]
        return {"case": case, "ok": False, "error": " | ".join(tail) or "không rõ"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    observed = report["observed"]
    return {
        "case": case,
        "ok": True,
        "sec_per_step": observed.get("sec_per_step"),
        "peak_vram_mb": report.get("peak_vram_mb"),
        "effective_batch": observed.get("effective_batch_size"),
        "batch_ok": observed.get("effective_batch_matches_config"),
        "hours_per_epoch": (
            round(observed["estimated_sec_per_epoch"] / 3600, 2)
            if observed.get("estimated_sec_per_epoch")
            else None
        ),
        "setup_overhead_sec": observed.get("setup_overhead_sec"),
    }


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'':2} {'batch':>6} {'mini':>5} {'ckpt':>5} {'s/step':>9} {'VRAM MB':>9} {'h/epoch':>8}")
    print("-" * 78)
    for row in results:
        case = row["case"]
        ckpt = "on" if case.grad_checkpointing else "off"
        if not row["ok"]:
            print(f"{case.name:2} {case.batch_size:>6} {case.mini_batch_size:>5} {ckpt:>5}   LỖI: {row['error'][:40]}")
            continue
        print(
            f"{case.name:2} {case.batch_size:>6} {case.mini_batch_size:>5} {ckpt:>5} "
            f"{row['sec_per_step'] or 0:>9.1f} {row['peak_vram_mb'] or 0:>9.0f} "
            f"{row['hours_per_epoch'] or 0:>8.1f}"
        )
    print("-" * 78)

    good = [r for r in results if r["ok"] and r["sec_per_step"]]
    if not good:
        print("Không case nào chạy được.")
        return

    best = min(good, key=lambda r: r["sec_per_step"])
    baseline = next((r for r in good if r["case"].name == "A"), None)
    speedup = (
        f", nhanh hơn A {baseline['sec_per_step'] / best['sec_per_step']:.1f}×"
        if baseline and baseline is not best
        else ""
    )
    print(f"Nhanh nhất: {best['case'].name} — {best['sec_per_step']:.1f} s/step{speedup}")
    print(f"            {best['hours_per_epoch']:.1f} h/epoch, peak {best['peak_vram_mb']:.0f} MB")

    if best["case"].batch_size != 256:
        print(
            "LƯU Ý: case này giảm effective batch nên ĐỔI chất lượng, không chỉ tốc độ —\n"
            "       MNRL mạnh lên theo số in-batch negative. Ghi rõ vào báo cáo."
        )
    for row in good:
        if row["batch_ok"] is False:
            print(f"CẢNH BÁO {row['case'].name}: effective batch thực tế = {row['effective_batch']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cấu hình Bi-Encoder")
    parser.add_argument("--steps", type=int, default=5, help="Số step mỗi case (mặc định 5)")
    parser.add_argument("--config", type=Path, default=Path("configs/method2/biencoder.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/method2/biencoder"))
    parser.add_argument("--only", type=str, default=None, help="Chỉ chạy case nào, vd A,B,C")
    parser.add_argument("--output", type=Path, default=None, help="Ghi kết quả ra JSON")
    args = parser.parse_args()

    wanted = {n.strip().upper() for n in args.only.split(",")} if args.only else None
    cases = [c for c in GRID if wanted is None or c.name in wanted]

    results = [run_case(c, args.steps, args.config, args.output_root) for c in cases]
    print_table(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                [{**r, "case": r["case"].__dict__} for r in results], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        print(f"\n→ {args.output}")


if __name__ == "__main__":
    main()
