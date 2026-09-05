"""Hiệu chỉnh ngưỡng `should_call` trên val — offline, không cần GPU.

Pipeline ghi `metadata.should_call_prob` vào từng prediction, nên quét ngưỡng
chỉ là đọc lại một file. Cùng cái lợi đã cứu §7.10b: chạy GPU một lần trên val,
rồi dò ngưỡng bao nhiêu lần cũng được mà không tốn thêm quota.

Mục tiêu tối ưu **Macro-F1 giữa {call, no_tool_call}** — cùng hàm mục tiêu mà
`biencoder.evaluate.calibrate_tau` dùng cho τ. Ablation §6.1 so hai cơ chế
abstention với nhau, nên chúng phải được chọn ngưỡng theo cùng một tiêu chí; đổi
tiêu chí là so hai thứ khác nhau.

Quy trình (giống hệt τ): chạy val → chốt ngưỡng → **FREEZE** → mới chạy test.

    python scripts/method2/calibrate_should_call.py \\
      --gold data/custom_vi/v1/val_seen.jsonl \\
      --predictions results/method2/predictions/val_seen/predictions.jsonl \\
      --output artifacts/method2/crossencoder/run02/should_call_threshold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

# Cùng công thức macro-F1 với calibrate_tau — import chứ không chép lại, để hai
# nhánh ablation không thể lệch nhau về sau.
from src.models.biencoder.evaluate import _macro_f1
from src.models.sources import load_jsonl


def collect_probs(
    gold_path: Path, predictions_path: Path
) -> tuple[list[tuple[float, bool]], dict[str, Any]]:
    """`[(prob, is_negative)]` + báo cáo độ phủ."""
    is_negative = {
        row["id"]: not (row.get("function_calls") or []) for row in load_jsonl(gold_path)
    }
    pairs: list[tuple[float, bool]] = []
    n_missing = 0
    for row in load_jsonl(predictions_path):
        prob = (row.get("metadata") or {}).get("should_call_prob")
        if prob is None or row.get("id") not in is_negative:
            n_missing += 1
            continue
        pairs.append((float(prob), is_negative[row["id"]]))
    report = {
        "n_scored": len(pairs),
        "n_missing_prob": n_missing,
        "n_negative": sum(1 for _, neg in pairs if neg),
        "n_positive": sum(1 for _, neg in pairs if not neg),
    }
    return pairs, report


def sweep(
    pairs: Sequence[tuple[float, bool]], grid: Sequence[float]
) -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    best_threshold, best_score, best_detail = grid[0], -1.0, {}
    curve: list[dict[str, Any]] = []
    for threshold in grid:
        tp = fp = fn = tn = 0
        for prob, negative in pairs:
            predicted_call = prob >= threshold
            if negative:
                fp += int(predicted_call)
                tn += int(not predicted_call)
            else:
                tp += int(predicted_call)
                fn += int(not predicted_call)
        macro, detail = _macro_f1(tp, fp, fn, tn)
        curve.append({"threshold": round(threshold, 4), "macro_f1": round(macro, 6), **detail})
        if macro > best_score:
            best_threshold, best_score, best_detail = threshold, macro, detail
    return best_threshold, best_detail, curve


def _grid(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hiệu chỉnh ngưỡng should_call trên val")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.01)
    args = parser.parse_args()

    pairs, report = collect_probs(args.gold, args.predictions)
    if not pairs:
        raise SystemExit(
            "[should_call] không có `metadata.should_call_prob` nào — chạy pipeline "
            "với abstention='should_call' trước đã."
        )
    if not report["n_negative"]:
        raise SystemExit(
            "[should_call] tập val không có sample no-call: không hiệu chỉnh được "
            "ngưỡng abstention trên đó."
        )

    threshold, detail, curve = sweep(pairs, _grid(0.0, 1.0, args.step))
    payload = {
        "should_call_threshold": threshold,
        "calibrated_on": str(args.gold),
        "objective": "macro_f1",
        "metrics": detail,
        "coverage": report,
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[should_call] {report['n_positive']} positive / {report['n_negative']} negative")
    print(f"[should_call] ngưỡng = {threshold:.2f}  macro_f1 = {detail.get('macro_f1', 0):.4f}")
    print(f"[should_call] → {args.output}  (FREEZE trước khi chạy test)")


if __name__ == "__main__":
    main()
