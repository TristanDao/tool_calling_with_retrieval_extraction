"""Phát lại bước chọn tool từ predictions đã có — KHÔNG cần GPU, không cần model.

Vì sao cần: `raw_predictions_pipeline.jsonl` đã lưu `ranked_tools` (top-20 kèm
score cosine) cho từng sample. Đổi ngưỡng hay đổi chiến lược chọn call chỉ là
tính lại trên đúng bộ score đó — chạy được offline trong vài giây.

Dùng để trả lời "sửa `strategy` thì Tool Set Accuracy lên bao nhiêu" TRƯỚC khi
tiêu giờ GPU chạy lại cả pipeline.

```bash
python scripts/method2/replay_selection.py \
    --gold data/custom_vi/v1/test_seen.jsonl \
    --raw results/method2/predictions/custom_seen/raw_predictions_pipeline.jsonl \
    --thresholds artifacts/method2/biencoder/run02/thresholds.json
```

Lưu ý: đây là công cụ **chẩn đoán**. Ngưỡng dùng cho kết quả chính thức vẫn
phải hiệu chỉnh trên val rồi freeze (§5.1) — không được chọn ngưỡng bằng cách
quét trên test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.biencoder.evaluate import replay_selection  # noqa: E402
from src.models.biencoder.retrieve import RetrievalThresholds  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fmt(value: float | None, width: int, digits: int = 4) -> str:
    return f"{value:{width}.{digits}f}" if value is not None else f"{'—':>{width}}"


def _row(label: str, result: dict[str, Any]) -> str:
    return (
        f"{label:28s} "
        f"{_fmt(result['tool_set_accuracy_positive'], 9)} "
        f"{_fmt(result['mean_selected_positive'], 14, 2)} "
        f"{_fmt(result['negative_recall'], 12)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Quét thêm vài giá trị gap_delta/tau_call để thấy hình dạng đường cong",
    )
    args = parser.parse_args()

    gold = load_jsonl(args.gold)
    raw = load_jsonl(args.raw)
    saved = json.loads(args.thresholds.read_text(encoding="utf-8"))
    base = RetrievalThresholds.load(args.thresholds)
    winner = ((saved.get("metrics") or {}).get("call_selection") or {}).get("winner")

    print(f"gold {len(gold)} sample · raw {len(raw)} sample")
    print(f"thresholds: strategy={base.strategy!r} winner-đã-tính={winner!r}")
    if winner and winner != base.strategy:
        print("  ^ LỆCH — file ghi một đằng, calibration chọn một nẻo")
    print()

    header = f"{'cấu hình':28s} {'toolset acc':>9s} {'#tool chọn TB':>14s} {'neg recall':>12s}"
    print(header)
    print("-" * len(header))

    variants: list[tuple[str, RetrievalThresholds]] = []
    for strategy in ("absolute", "gap"):
        thresholds = RetrievalThresholds(
            tau=base.tau,
            tau_call=base.tau_call,
            gap_delta=base.gap_delta,
            k_max=base.k_max,
            strategy=strategy,
        )
        mark = "  <- đang dùng" if strategy == base.strategy else ""
        variants.append((f"{strategy}{mark}", thresholds))

    if args.sweep:
        for delta in (0.05, 0.10, 0.15, 0.25, 0.30):
            variants.append(
                (
                    f"gap delta={delta}",
                    RetrievalThresholds(
                        tau=base.tau, gap_delta=delta, k_max=base.k_max, strategy="gap"
                    ),
                )
            )
        for k in (1, 2):
            variants.append(
                (
                    f"gap k_max={k}",
                    RetrievalThresholds(
                        tau=base.tau,
                        gap_delta=base.gap_delta,
                        k_max=k,
                        strategy="gap",
                    ),
                )
            )

    for label, thresholds in variants:
        print(_row(label, replay_selection(gold, raw, thresholds)))

    print()
    print("Ngưỡng chính thức vẫn phải freeze từ val — bảng này chỉ để chẩn đoán.")


if __name__ == "__main__":
    main()
