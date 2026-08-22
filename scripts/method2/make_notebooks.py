"""Sinh 3 notebook Kaggle cho Method 2 (§8 method2_plan).

Viết bằng script thay vì soạn tay JSON để 3 notebook luôn dùng chung phần
bootstrap (cài package, mount dataset, resume-safe) — sửa một chỗ là cả ba đổi.

Chạy: python scripts/method2/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path("notebooks")

SETUP_CELLS = [
    (
        "markdown",
        [
            "# {title}\n",
            "\n",
            "{subtitle}\n",
            "\n",
            "**Ba quy tắc sống còn trên Kaggle** (§8 `docs/method2_plan.md`):\n",
            "\n",
            "1. Bật **Save & Run All (Commit)** cho job dài — session tương tác bị ngắt sau ~20 phút không tương tác, commit run chạy nền đủ 12h.\n",
            "2. Checkpoint mỗi 500 step vào `/kaggle/working`, và **luôn** hỗ trợ `resume_from`.\n",
            "3. Cache model HuggingFace thành Kaggle Dataset (`BAAI/bge-m3` ~2.3GB) thay vì tải lại mỗi session.\n",
        ],
    ),
    (
        "code",
        [
            "# ===== Cell 1: env =====\n",
            "!pip install -q 'sentence-transformers>=3.0' peft transformers accelerate \\\n",
            "                jsonschema rank_bm25 datasets\n",
            "\n",
            "import torch\n",
            "\n",
            "free, total = torch.cuda.mem_get_info()\n",
            "print(torch.cuda.get_device_name(0), f'{free/1024**3:.1f} / {total/1024**3:.1f} GB free')\n",
            "# Kỳ vọng: Tesla T4, ~15.0 GB free. T4 KHÔNG có bf16 → mọi config dùng fp16.\n",
        ],
    ),
    (
        "code",
        [
            "# ===== Cell 2: mount code + data =====\n",
            "# Đẩy repo và data lên Kaggle Dataset (private) trước.\n",
            "!cp -r /kaggle/input/toolcalling-vi-src/src /kaggle/working/\n",
            "!cp -r /kaggle/input/toolcalling-vi-src/configs /kaggle/working/\n",
            "!cp -r /kaggle/input/toolcalling-vi-data/data /kaggle/working/data\n",
            "%cd /kaggle/working\n",
            "\n",
            "import json, os, glob, sys\n",
            "sys.path.insert(0, '/kaggle/working')\n",
            "\n",
            "# Cache model HF thành Kaggle Dataset để không tải lại mỗi session.\n",
            "os.environ.setdefault('HF_HOME', '/kaggle/input/hf-cache')\n",
            "\n",
            "manifest = json.load(open('data/method2/manifest.json', encoding='utf-8'))\n",
            "print('snapshot commit:', manifest.get('git_commit'))\n",
        ],
    ),
]


def _cell(kind: str, source: list[str]) -> dict:
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _setup(title: str, subtitle: str) -> list[dict]:
    """Chỉ thay hai placeholder; không dùng `str.format` vì source có f-string."""
    cells = []
    for kind, source in SETUP_CELLS:
        rendered = [
            line.replace("{title}", title).replace("{subtitle}", subtitle) for line in source
        ]
        cells.append(_cell(kind, rendered))
    return cells


BIENCODER_CELLS = [
    (
        "markdown",
        [
            "## Round 1 — train với hard negative round-0\n",
            "\n",
            "`CachedMultipleNegativesRankingLoss` (GradCache) cho effective batch 256 với\n",
            "mini_batch 8. Gradient accumulation **không** thay thế được: nó chỉ chia nhỏ\n",
            "update chứ không làm tăng số in-batch negative.\n",
        ],
    ),
    (
        "code",
        [
            "RUN = '/kaggle/working/artifacts/method2/biencoder/run01'\n",
            "resume = sorted(glob.glob(f'{RUN}/checkpoint-*'))[-1] if glob.glob(f'{RUN}/checkpoint-*') else None\n",
            "print('resume from:', resume)\n",
            "\n",
            "!python -m src.models.biencoder.train train \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --output-dir {RUN} \\\n",
            "    --resume-from {resume}\n",
        ],
    ),
    (
        "markdown",
        [
            "## Round 2 — mine hard negatives rồi train lại **từ base**\n",
            "\n",
            "Lấy tool sai nhưng xếp hạng cao (bỏ top-1 để tránh false negative). Round 2\n",
            "train lại từ checkpoint gốc, không train tiếp từ round 1.\n",
        ],
    ),
    (
        "code",
        [
            "!python -m src.models.biencoder.train mine \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --model {RUN}/final\n",
            "\n",
            "RUN2 = '/kaggle/working/artifacts/method2/biencoder/run02'\n",
            "!sed -i 's#biencoder/train.jsonl#biencoder/train_mined.jsonl#' configs/method2/biencoder.yaml\n",
            "!python -m src.models.biencoder.train train \\\n",
            "    --config configs/method2/biencoder.yaml --output-dir {RUN2}\n",
        ],
    ),
    (
        "markdown",
        ["## Pre-compute index + hiệu chỉnh ngưỡng trên **val**\n"],
    ),
    (
        "code",
        [
            "!python -m src.models.biencoder.index \\\n",
            "    --config configs/method2/biencoder.yaml --model {RUN2}/final\n",
            "\n",
            "# τ và τ_call CHỈ được hiệu chỉnh trên val, rồi freeze trước khi chạy test.\n",
            "!python -m src.models.biencoder.evaluate calibrate \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --model {RUN2}/final \\\n",
            "    --pairs data/method2/biencoder/val.jsonl \\\n",
            "    --output {RUN2}/thresholds.json\n",
        ],
    ),
    (
        "markdown",
        [
            "## Gate để qua Phase 3\n",
            "\n",
            "| Metric | Tập | Ngưỡng |\n",
            "|---|---|---|\n",
            "| Recall@1 | custom `val_seen` | ≥ 0.90 |\n",
            "| Recall@1 | custom `val_unseen` | ≥ 0.75 |\n",
            "| Recall@5 | custom `val_unseen` | ≥ 0.92 |\n",
            "| Negative Recall @ τ | custom val negative | ≥ 0.80 |\n",
            "\n",
            "Không đạt → thử theo thứ tự: (a) thêm param name vào document text,\n",
            "(b) tăng hard negative lên 8, (c) đổi sang `AITeamVN/Vietnamese_Embedding`,\n",
            "(d) full fine-tune thay LoRA.\n",
        ],
    ),
    (
        "code",
        [
            "!python -m src.models.biencoder.evaluate evaluate \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --model {RUN2}/final \\\n",
            "    --pairs data/method2/biencoder/val.jsonl \\\n",
            "    --output results/method2/metrics/biencoder_val.json\n",
            "\n",
            "report = json.load(open('results/method2/metrics/biencoder_val.json', encoding='utf-8'))\n",
            "for slice_name, metrics in report['by_source_key'].items():\n",
            "    print(slice_name, {k: v for k, v in metrics.items() if 'recall@' in k or k == 'mrr'})\n",
        ],
    ),
]

CROSSENCODER_CELLS = [
    (
        "markdown",
        [
            "## Sinh cặp (query, parameter) và kiểm tra tỉ lệ SKIP\n",
            "\n",
            "**Gate §1.5**: nếu %SKIP > 30% thì dừng lại, xem lại quyết định Q2 (có thêm\n",
            "fuzzy span alignment hay không) trước khi train.\n",
        ],
    ),
    (
        "code",
        [
            "!python -m src.models.crossencoder.dataset --config configs/method2/crossencoder.yaml\n",
            "\n",
            "stats = json.load(open('data/method2/label_stats.json', encoding='utf-8'))\n",
            "print('SKIP rate:', stats['skip_rate'])\n",
            "print('theo lý do:', stats['skip_by_reason'])\n",
            "print('unsupported coverage:', stats['unsupported_type_coverage'])\n",
            "assert stats['skip_rate'] <= 0.30, 'SKIP quá ngưỡng — xem lại Q2 trước khi train'\n",
        ],
    ),
    (
        "markdown",
        [
            "## Train — curriculum 2 giai đoạn, resume-safe\n",
            "\n",
            "1. Warm-up trên glaive+xLAM (2 epoch) — học kỹ năng span tổng quát.\n",
            "2. Fine-tune trên custom_vi (2 epoch, lr 1e-5) — domain đích.\n",
        ],
    ),
    (
        "code",
        [
            "RUN = '/kaggle/working/artifacts/method2/crossencoder/run01'\n",
            "resume = sorted(glob.glob(f'{RUN}/checkpoint-*'))[-1] if glob.glob(f'{RUN}/checkpoint-*') else None\n",
            "print('resume from:', resume)\n",
            "\n",
            "!python -m src.models.crossencoder.train \\\n",
            "    --config configs/method2/crossencoder.yaml \\\n",
            "    --output-dir {RUN} \\\n",
            "    --resume-from {resume}\n",
        ],
    ),
    (
        "markdown",
        [
            "## Gate để qua Phase 4 (chế độ oracle retrieval)\n",
            "\n",
            "`has_value` F1 ≥ 0.90 · Span EM ≥ 0.80 · Enum acc ≥ 0.90 · Boolean acc ≥ 0.85\n",
        ],
    ),
    (
        "code",
        [
            "!python -m src.models.crossencoder.evaluate \\\n",
            "    --config configs/method2/crossencoder.yaml \\\n",
            "    --model {RUN}/final \\\n",
            "    --pairs data/method2/crossencoder/val.jsonl \\\n",
            "    --output results/method2/metrics/crossencoder_val.json\n",
        ],
    ),
]

EVAL_CELLS = [
    (
        "markdown",
        [
            "## Hai chế độ bắt buộc (§6.3 experimental_plan)\n",
            "\n",
            "| Chế độ | Input Cross-Encoder | Trả lời câu hỏi |\n",
            "|---|---|---|\n",
            "| `oracle` | Tool gold | Extraction tốt đến đâu, độc lập retrieval |\n",
            "| `pipeline` | Bi-Encoder top-k + abstention | Hiệu năng hệ thống thật |\n",
            "\n",
            "`ArgA_oracle − ArgA_pipeline` = phần lỗi do retrieval. Con số này phải xuất\n",
            "hiện tường minh trong báo cáo.\n",
        ],
    ),
    (
        "code",
        [
            "for gold, tag in [('data/custom_vi/v1/test_seen.jsonl', 'custom_seen'),\n",
            "                  ('data/custom_vi/v1/test_unseen.jsonl', 'custom_unseen'),\n",
            "                  ('data/benchmark_vi/test.jsonl', 'benchmark')]:\n",
            "    for mode in ['pipeline', 'oracle']:\n",
            "        !python -m src.models.pipeline.method2 \\\n",
            "            --config configs/method2/pipeline.yaml \\\n",
            "            --gold {gold} --mode {mode} \\\n",
            "            --output-dir results/method2/predictions/{tag}\n",
        ],
    ),
    (
        "markdown",
        [
            "## Chạy evaluator chung\n",
            "\n",
            "Cùng normalization, cùng rule so khớp với ba method còn lại — xem\n",
            "`docs/evaluation.md`.\n",
        ],
    ),
    (
        "code",
        [
            "!python -m src.evaluation.cli evaluate \\\n",
            "    --gold data/custom_vi/v1/test_seen.jsonl \\\n",
            "    --predictions results/method2/predictions/custom_seen/predictions.jsonl \\\n",
            "    --oracle-predictions results/method2/predictions/custom_seen/oracle_predictions.jsonl \\\n",
            "    --slice metadata.tool_split \\\n",
            "    --output-dir results/evaluation/method_2_seen\n",
        ],
    ),
    (
        "markdown",
        ["## Latency tách 4 giai đoạn — `t_index_build` ghi riêng, không cộng vào\n"],
    ),
    (
        "code",
        [
            "latency = json.load(open('results/method2/predictions/custom_seen/latency_pipeline.json', encoding='utf-8'))\n",
            "for stage in ['t_query_embed', 't_retrieve', 't_cross_encode', 't_validate', 'total']:\n",
            "    print(f\"{stage:16s} p50={latency[stage]['p50_ms']:8.2f} ms  p95={latency[stage]['p95_ms']:8.2f} ms\")\n",
            "\n",
            "index_meta = json.load(open('data/method2/index/index_meta.json', encoding='utf-8'))\n",
            "print('\\nt_index_build (một lần, KHÔNG cộng vào latency/query):', index_meta['t_index_build_sec'], 's')\n",
        ],
    ),
]

SAVE_CELL = (
    "code",
    [
        "# ===== Lưu artifact =====\n",
        "# Kaggle chỉ giữ /kaggle/working (20GB). Nén để tải về hoặc làm Dataset mới.\n",
        "!tar czf /kaggle/working/{archive}.tar.gz -C /kaggle/working/artifacts/method2 .\n",
        "!du -h /kaggle/working/{archive}.tar.gz\n",
    ],
)


def build_notebooks() -> list[Path]:
    specs = [
        (
            "method2_kaggle_biencoder.ipynb",
            "Method 2 — Bi-Encoder (Kaggle T4)",
            "Phase 2 của `docs/method2_plan.md`: train 2 vòng, pre-compute index, hiệu chỉnh ngưỡng. Ngân sách ~4h GPU.",
            BIENCODER_CELLS,
            "biencoder_run",
        ),
        (
            "method2_kaggle_crossencoder.ipynb",
            "Method 2 — Cross-Encoder (Kaggle T4)",
            "Phase 3 của `docs/method2_plan.md`: sinh nhãn, curriculum 2 giai đoạn, gate theo head. Ngân sách ~4.5h GPU.",
            CROSSENCODER_CELLS,
            "crossencoder_run",
        ),
        (
            "method2_kaggle_eval.ipynb",
            "Method 2 — Full pipeline & đánh giá (Kaggle T4)",
            "Phase 5 của `docs/method2_plan.md`: chạy oracle + full pipeline, xuất predictions theo contract, đo latency 4 giai đoạn. Ngân sách ~1h GPU.",
            EVAL_CELLS,
            "eval_run",
        ),
    ]

    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, title, subtitle, body, archive in specs:
        cells = _setup(title, subtitle)
        cells.extend(_cell(kind, source) for kind, source in body)
        kind, source = SAVE_CELL
        cells.append(_cell(kind, [line.replace("{archive}", archive) for line in source]))
        path = NOTEBOOK_DIR / filename
        path.write_text(
            json.dumps(_notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        written.append(path)
    return written


if __name__ == "__main__":
    for path in build_notebooks():
        print(f"[notebooks] → {path}")
