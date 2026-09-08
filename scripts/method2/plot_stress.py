"""Vẽ 2 biểu đồ Phase 7 từ `stress_report.json`: accuracy-vs-N và latency-vs-N.

Tách khỏi `src/models/pipeline/stress.py` vì hai việc có vòng đời khác nhau:
chạy stress test cần GPU (Kaggle), còn vẽ lại hình thì chỉ cần file JSON đã tải
về — chỉnh nhãn, đổi màu, xuất lại cho báo cáo không nên phải đụng tới quota.

Trục x log theo §Phase 7 `method2_plan`. Trục y của biểu đồ latency để **tuyến
tính**: điều cần nhìn thấy là đường nằm ngang, mà log trục y sẽ bóp phẳng cả
những đường thật sự tăng tuyến tính, tức là làm hỏng đúng thứ cần chứng minh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ACCURACY_SERIES: tuple[tuple[str, str], ...] = (
    ("n_fcem_positive", "N-FCEM positive"),
    ("tool_set_accuracy", "Tool Set Accuracy"),
    ("recall_at_1", "Recall@1"),
    ("arg_em_given_correct_tool", "ArgEM | tool đúng"),
    ("negative_recall", "Negative Recall"),
)

LATENCY_STAGES: tuple[tuple[str, str], ...] = (
    ("total", "tổng"),
    ("t_cross_encode", "cross-encode"),
    ("t_query_embed", "embed query"),
    ("t_retrieve", "retrieve"),
    ("t_validate", "validate"),
)


def _nan(value: Any) -> float:
    """`None` → NaN để matplotlib bỏ trống điểm đó thay vì ném lỗi."""
    return float("nan") if value is None else float(value)


def _rows_for_mode(report: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in report["rows"] if row["mode"] == mode), key=lambda r: r["n"]
    )


def plot_accuracy(rows: list[dict[str, Any]], mode: str, output: Path) -> Path:
    import matplotlib.pyplot as plt

    ns = [row["n"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for key, label in ACCURACY_SERIES:
        values = [_nan(row.get(key)) for row in rows]
        if all(v != v for v in values):  # toàn NaN → không có gì để vẽ
            continue
        axis.plot(ns, values, marker="o", label=label)
    axis.set_xscale("log")
    axis.set_xticks(ns)
    axis.set_xticklabels([str(n) for n in ns])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Số tool trong haystack (N, log)")
    axis.set_ylabel("Giá trị metric")
    axis.set_title(f"Method 2 — accuracy theo N (distractor: {mode})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_latency(rows: list[dict[str, Any]], mode: str, output: Path) -> Path:
    import matplotlib.pyplot as plt

    ns = [row["n"] for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    for stage, label in LATENCY_STAGES:
        values = [
            _nan(((row.get("latency") or {}).get(stage) or {}).get("p50_ms"))
            for row in rows
        ]
        if all(v != v for v in values):  # toàn NaN → không có gì để vẽ
            continue
        axis.plot(ns, values, marker="o", label=label, linewidth=2 if stage == "total" else 1)
    axis.set_xscale("log")
    axis.set_xticks(ns)
    axis.set_xticklabels([str(n) for n in ns])
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Số tool trong haystack (N, log)")
    axis.set_ylabel("Latency p50 (ms)")
    axis.set_title(f"Method 2 — latency theo N (distractor: {mode})")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def plot_report(report: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mode in dict.fromkeys(row["mode"] for row in report["rows"]):
        rows = _rows_for_mode(report, mode)
        label = rows[0].get("distractor_label", mode)
        written.append(plot_accuracy(rows, label, output_dir / f"stress_accuracy_{mode}.png"))
        written.append(plot_latency(rows, label, output_dir / f"stress_latency_{mode}.png"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ stress test Phase 7")
    parser.add_argument(
        "--report", type=Path, default=Path("results/method2/stress/stress_report.json")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    for path in plot_report(report, args.output_dir or args.report.parent):
        print(f"[stress] → {path}")


if __name__ == "__main__":
    main()
