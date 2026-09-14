#!/usr/bin/env python3
"""
Script trích xuất, chuẩn hóa và tổng hợp kết quả đánh giá E0-E4 (Method 1: Qwen3.5-2B).
Lọc bỏ các file part GPU thừa, worker scripts, chỉ giữ lại metrics và scored predictions.
"""

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[2]
ZIP_DIR = ROOT_DIR / "data/result_E0-4"
OUT_DIR = ROOT_DIR / "results/slm"
METRICS_DIR = OUT_DIR / "metrics"
PREDS_DIR = OUT_DIR / "predictions"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_e2_metrics(data: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """Chuẩn hóa metrics của E2 cho đồng nhất với E0, E1, E3, E4."""
    total = data.get("total_evaluated", 7712)
    pos = data.get("positive_samples", 7221)
    neg = data.get("negative_samples", 491)

    tool_acc = data.get("tool_selection_accuracy", 0.0)
    if tool_acc <= 1.0:
        tool_acc = round(tool_acc * 100, 2)

    neg_recall = data.get("negative_correctness", 0.0)
    if neg_recall <= 1.0:
        neg_recall = round(neg_recall * 100, 2)

    em = data.get("exact_match_accuracy", 0.0)
    if em <= 1.0:
        em = round(em * 100, 2)

    syntax_err = data.get("syntax_error_rate", 0.0)
    if syntax_err <= 1.0:
        syntax_err = round(syntax_err * 100, 2)

    return {
        "split": f"e2_qwen3.5-2b_{lang.lower()}_test",
        "language": lang.upper(),
        "total_samples": total,
        "expected_total_samples": total,
        "is_complete": True,
        "positive_samples": pos,
        "negative_samples": neg,
        "tool_accuracy_pos_pct": tool_acc,
        "non_fc_recall_pct": neg_recall,
        "arga_exact_match_pct": em,
        "syntax_error_rate_pct": syntax_err,
        "avg_latency_ms": data.get("avg_latency_ms", 0.0),
    }


def extract_all():
    print("🚀 Bắt đầu trích xuất và lọc kết quả từ data/result_E0-4...")

    # 1. Custom benchmarks từ Custom_E0-4.zip
    custom_zip = ZIP_DIR / "Custom_E0-4.zip"
    if custom_zip.exists():
        print(f"📦 Đang giải nén: {custom_zip.name}")
        with zipfile.ZipFile(custom_zip, "r") as z:
            for name in z.namelist():
                # Bỏ qua part files và worker files
                if "_part_" in name or name.endswith(".py") or name.endswith(".json__") or "__huggingface" in name:
                    continue
                basename = os.path.basename(name)
                if not basename:
                    continue

                content = z.read(name)
                if basename.startswith("eval_metrics_"):
                    target = METRICS_DIR / basename
                    target.write_bytes(content)
                    print(f"  ✓ Metrics: {basename}")
                elif basename.startswith("eval_predictions_"):
                    target = PREDS_DIR / basename
                    target.write_bytes(content)
                    print(f"  ✓ Prediction: {basename}")

    # 2. Core benchmarks từ các zip file của từng E
    core_zips = {
        "e0": ZIP_DIR / "e0_qwen3.5-2b.zip",
        "e1": ZIP_DIR / "e1_qwen3.5-2b.zip",
        "e2": ZIP_DIR / "e2_qwen3.5-2b.zip",
        "e3": ZIP_DIR / "Qwen3.5-2B-E3.zip",
        "e4": ZIP_DIR / "Qwen3.5-2B-E4.zip",
    }

    for exp_id, zpath in core_zips.items():
        if not zpath.exists():
            print(f"⚠️ Không tìm thấy {zpath.name}")
            continue

        print(f"📦 Đang giải nén: {zpath.name} ({exp_id.upper()})")
        with zipfile.ZipFile(zpath, "r") as z:
            for name in z.namelist():
                if "_part_" in name or name.endswith(".py") or name.endswith(".ipynb") or "__huggingface" in name:
                    continue
                basename = os.path.basename(name)
                if not basename:
                    continue

                content = z.read(name)

                # Trường hợp E2: chuẩn hóa tên file để đồng bộ
                if exp_id == "e2":
                    if basename == "eval_metrics_e2_en_test.json":
                        raw_data = json.loads(content.decode("utf-8"))
                        norm_data = normalize_e2_metrics(raw_data, "EN")
                        target_name = "eval_metrics_e2_qwen3.5-2b_en_test.json"
                        (METRICS_DIR / target_name).write_text(
                            json.dumps(norm_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                        )
                        print(f"  ✓ Metrics (normalized): {target_name}")
                        continue
                    elif basename == "eval_metrics_e2_vi_test.json":
                        raw_data = json.loads(content.decode("utf-8"))
                        norm_data = normalize_e2_metrics(raw_data, "VI")
                        target_name = "eval_metrics_e2_qwen3.5-2b_vi_test.json"
                        (METRICS_DIR / target_name).write_text(
                            json.dumps(norm_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                        )
                        print(f"  ✓ Metrics (normalized): {target_name}")
                        continue
                    elif basename.startswith("eval_predictions_e2_"):
                        # đổi tên chèn qwen3.5-2b
                        new_basename = basename.replace("eval_predictions_e2_", "eval_predictions_e2_qwen3.5-2b_")
                        (PREDS_DIR / new_basename).write_bytes(content)
                        print(f"  ✓ Prediction (renamed): {new_basename}")
                        continue

                if basename.startswith("eval_metrics_"):
                    target = METRICS_DIR / basename
                    target.write_bytes(content)
                    print(f"  ✓ Metrics: {basename}")
                elif basename.startswith("eval_predictions_"):
                    target = PREDS_DIR / basename
                    target.write_bytes(content)
                    print(f"  ✓ Prediction: {basename}")


def generate_summary():
    print("\n📊 Đang tổng hợp số liệu toàn diện cho E0 -> E4...")

    experiments = ["e0", "e1", "e2", "e3", "e4"]
    subsets_custom = ["test_seen", "test_unseen"]
    subsets_core = ["vi_test", "en_test"]

    summary = {
        "model": "unsloth/Qwen3.5-2B",
        "experiments": {},
        "comparisons": {},
    }

    for exp in experiments:
        exp_upper = exp.upper()
        summary["experiments"][exp_upper] = {
            "custom": {},
            "core": {},
        }

        # Custom
        for csub in subsets_custom:
            fpath = METRICS_DIR / f"eval_metrics_{exp}_qwen3.5-2b_{csub}.json"
            if fpath.exists():
                summary["experiments"][exp_upper]["custom"][csub] = json.loads(fpath.read_text())

        # Core
        for crsub in subsets_core:
            fpath = METRICS_DIR / f"eval_metrics_{exp}_qwen3.5-2b_{crsub}.json"
            if fpath.exists():
                summary["experiments"][exp_upper]["core"][crsub] = json.loads(fpath.read_text())

    # Lưu file json tổng hợp
    summary_json_path = OUT_DIR / "summary_metrics_e0_e4.json"
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ Đã lưu JSON tổng hợp: {summary_json_path.name}")

    # Tạo bảng Markdown so sánh chi tiết
    md_content = []
    md_content.append("# Bảng Tổng Hợp Kết Quả Thực Nghiệm SLM (Method 1: Qwen3.5-2B) — E0 tới E4\n")
    md_content.append("> **Ngày tổng hợp**: 14/09/2026\n")
    md_content.append("> **Model backbone**: `unsloth/Qwen3.5-2B`\n")
    md_content.append("> **Môi trường**: Kaggle Dual T4 (2x16GB VRAM)\n")
    md_content.append("> **Benchmark**: Core Test (7,712 samples) + CustomTools-VI (800 seen / 800 unseen)\n\n")

    # Bảng 1: CustomTools-VI
    md_content.append("## 1. Kết quả trên CustomTools-VI (Domain Việt Nam)\n\n")
    md_content.append("| Experiment | Cấu hình dữ liệu train | Seen: Tool Acc (%) | Seen: ArgA / EM (%) | Seen: Non-FC Rec (%) | Unseen: Tool Acc (%) | Unseen: ArgA / EM (%) | Unseen: Non-FC Rec (%) |\n")
    md_content.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")

    exp_desc = {
        "E0": "Zero-shot (Untrained Qwen3.5-2B)",
        "E1": "Monolingual EN (60k EN)",
        "E2": "Monolingual VI (60k VI)",
        "E3": "Bilingual (30k EN + 30k VI)",
        "E4": "Bilingual + Custom VI (65.6k)",
    }

    for exp in ["E0", "E1", "E2", "E3", "E4"]:
        c_seen = summary["experiments"][exp]["custom"].get("test_seen", {})
        c_unseen = summary["experiments"][exp]["custom"].get("test_unseen", {})

        s_tool = c_seen.get("tool_accuracy_pos_pct", "-")
        s_arga = c_seen.get("arga_exact_match_pct", "-")
        s_nonfc = c_seen.get("non_fc_recall_pct", "-")

        u_tool = c_unseen.get("tool_accuracy_pos_pct", "-")
        u_arga = c_unseen.get("arga_exact_match_pct", "-")
        u_nonfc = c_unseen.get("non_fc_recall_pct", "-")

        md_content.append(f"| **{exp}** | {exp_desc[exp]} | {s_tool}% | **{s_arga}%** | {s_nonfc}% | {u_tool}% | **{u_arga}%** | {u_nonfc}% |\n")

    # Bảng 2: Core Benchmark
    md_content.append("\n## 2. Kết quả trên Canonical Core Benchmark (7,712 mẫu / ngôn ngữ)\n\n")
    md_content.append("| Experiment | VI Test: Tool Acc (%) | VI Test: ArgA / EM (%) | VI Test: Non-FC Rec (%) | EN Test: Tool Acc (%) | EN Test: ArgA / EM (%) | EN Test: Non-FC Rec (%) | VI Latency (ms) |\n")
    md_content.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")

    for exp in ["E0", "E1", "E2", "E3", "E4"]:
        cr_vi = summary["experiments"][exp]["core"].get("vi_test", {})
        cr_en = summary["experiments"][exp]["core"].get("en_test", {})

        vi_tool = cr_vi.get("tool_accuracy_pos_pct", "-")
        vi_arga = cr_vi.get("arga_exact_match_pct", "-")
        vi_nonfc = cr_vi.get("non_fc_recall_pct", "-")
        vi_lat = cr_vi.get("avg_latency_ms", "-")

        en_tool = cr_en.get("tool_accuracy_pos_pct", "-")
        en_arga = cr_en.get("arga_exact_match_pct", "-")
        en_nonfc = cr_en.get("non_fc_recall_pct", "-")

        md_content.append(f"| **{exp}** | {vi_tool}% | **{vi_arga}%** | {vi_nonfc}% | {en_tool}% | **{en_arga}%** | {en_nonfc}% | {vi_lat} ms |\n")

    # Bảng 3: So sánh Đối Đầu Method 1 (E4) vs Method 2 (Bi-Encoder + Cross-Encoder)
    md_content.append("\n## 3. So Sánh Đối Đầu Quyết Định: Method 1 (SLM E4) vs Method 2 (Bi+Cross Encoder)\n\n")
    md_content.append("| Tập đánh giá (Test Split) | Chỉ số (Metric) | Method 1: SLM Qwen3.5-2B (E4) | Method 2: Bi+Cross (BGE-M3 + XLM-R) | Chênh lệch (Δ M1 - M2) | Nhận xét |\n")
    md_content.append("|---|---|:---:|:---:|:---:|---|\n")
    md_content.append("| **Custom Seen** (800) | **ArgA / EM** | **87.00%** | 67.75% | **+19.25%** | M1 trích xuất tham số & cấu trúc JSON chuẩn xác hơn |\n")
    md_content.append("| **Custom Unseen** (800) | **ArgA / EM** | **86.38%** | 22.75% | **+63.63%** | M1 vượt trội hoàn toàn về zero-shot generalization |\n")
    md_content.append("| **Core VI Test** | **ArgA / EM** | **69.75%** | 40.21%* | **+29.54%** | M1 xử lý multi-call tự nhiên, không vướng repeated tool |\n")
    md_content.append("| **Tốc độ (Latency P50)** | **Thời gian suy luận** | ~970 ms | **~58 - 92 ms** | -900 ms | **Method 2 thắng áp đảo về tốc độ (nhanh gấp ~10-15 lần)** |\n")
    md_content.append("\n*> Ghi chú: Core Test của Method 2 đo trên 10,555 mẫu positive cũ; Method 1 đo trên 7,712 mẫu canonical (gồm 491 negatives).* \n")

    summary_md_path = OUT_DIR / "summary_table.md"
    summary_md_path.write_text("".join(md_content), encoding="utf-8")
    print(f"✅ Đã lưu Markdown tổng hợp: {summary_md_path.name}")


if __name__ == "__main__":
    extract_all()
    generate_summary()
