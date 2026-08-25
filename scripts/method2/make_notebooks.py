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
            "# ===== Cell 0: dò dataset + HF cache =====\n",
            "# PHẢI chạy trước mọi import transformers: thư viện chốt cache lúc import,\n",
            "# set HF_HOME sau đó thì không còn tác dụng.\n",
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            "INPUT_ROOT = Path('/kaggle/input')\n",
            "\n",
            "\n",
            "def _dirs_within(base: Path, max_depth: int = 4):\n",
            "    \"\"\"Mọi thư mục tới độ sâu `max_depth`, bỏ qua `hub/` cho nhanh.\"\"\"\n",
            "    frontier, seen = [base], []\n",
            "    for _ in range(max_depth):\n",
            "        nxt = []\n",
            "        for d in frontier:\n",
            "            try:\n",
            "                children = [c for c in d.iterdir() if c.is_dir() and c.name != 'hub']\n",
            "            except (PermissionError, OSError):\n",
            "                continue\n",
            "            seen.extend(children)\n",
            "            nxt.extend(children)\n",
            "        frontier = nxt\n",
            "    return seen\n",
            "\n",
            "\n",
            "def find_root(marker: str, label: str) -> Path:\n",
            "    \"\"\"Tìm thư mục chứa `marker`.\n",
            "\n",
            "    Kaggle mount theo dạng /kaggle/input/datasets/<user>/<ds>/<ds>/, và số tầng\n",
            "    đổi theo cách upload. Dò theo marker thì không phải hardcode username hay\n",
            "    độ sâu — upload kiểu nào cũng tìm ra.\n",
            "    \"\"\"\n",
            "    for d in [INPUT_ROOT] + _dirs_within(INPUT_ROOT):\n",
            "        if (d / marker).exists():\n",
            "            return d\n",
            "    raise SystemExit(\n",
            "        f'Không tìm thấy {label}: không thư mục nào dưới {INPUT_ROOT} có {marker}.\\n'\n",
            "        'Kiểm tra đã Add đủ 3 dataset ở sidebar Input chưa.'\n",
            "    )\n",
            "\n",
            "\n",
            "SRC_ROOT = find_root('src/models/preflight.py', 'dataset src')\n",
            "DATA_ROOT = find_root('method2/manifest.json', 'dataset data')\n",
            "HF_HOME = find_root('hub/models--BAAI--bge-m3', 'dataset hf-cache')\n",
            "\n",
            "print('SRC :', SRC_ROOT)\n",
            "print('DATA:', DATA_ROOT)\n",
            "print('HF  :', HF_HOME)\n",
            "\n",
            "os.environ['HF_HOME'] = str(HF_HOME)\n",
            "os.environ['HF_HUB_OFFLINE'] = '1'\n",
            "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n",
            "\n",
            "# Có thư mục model chưa đủ — thiếu file trọng số thì lỗi chỉ lộ ra lúc nạp\n",
            "# model, sau khi đã tốn thời gian cài đặt và copy.\n",
            "for name in ('models--BAAI--bge-m3', 'models--xlm-roberta-base'):\n",
            "    weights = [\n",
            "        f for f in (HF_HOME / 'hub' / name).rglob('*')\n",
            "        if f.is_file() and f.suffix in ('.safetensors', '.bin') and f.stat().st_size > 10**8\n",
            "    ]\n",
            "    assert weights, f'{name}: không có file trọng số > 100 MB'\n",
            "    print(f'  {name}: {max(f.stat().st_size for f in weights) / 1024**3:.2f} GB')\n",
            "print('\\nHF cache OK')\n",
        ],
    ),
    (
        "code",
        [
            "# ===== Cell 1: env — PIN version =====\n",
            "# Ba package này quyết định API training VÀ tên metric của\n",
            "# InformationRetrievalEvaluator. Đổi bản là đổi khoá metric, hỏng cả\n",
            "# load_best_model_at_end lẫn khả năng so sánh giữa các run.\n",
            "!pip install -q 'transformers==5.15.1' 'sentence-transformers==6.0.0' 'peft==0.20.0' \\\n",
            "                accelerate jsonschema rank_bm25 datasets\n",
            "\n",
            "# PEFT 0.20 raise nếu image có torchao < 0.16. Method 2 không dùng\n",
            "# torchao quantization nên gỡ hẳn là xong.\n",
            "!pip uninstall -y -q torchao 2>/dev/null || true\n",
            "\n",
            "# torch KHÔNG pin: Kaggle cài sẵn bản CUDA riêng, ép cài lại vừa chậm vừa\n",
            "# dễ lệch CUDA runtime của image. Chỉ ghi nhận version vào manifest.\n",
            "import torch\n",
            "\n",
            "free, total = torch.cuda.mem_get_info()\n",
            "n_gpu = torch.cuda.device_count()\n",
            "print(torch.cuda.get_device_name(0), f'{free/1024**3:.1f} / {total/1024**3:.1f} GB free')\n",
            "print('số GPU:', n_gpu)\n",
            "\n",
            "# sentence-transformers tự bọc DataParallel khi thấy >1 GPU. Với GradCache\n",
            "# gọi model hàng trăm lần mỗi step thì phí đồng bộ cộng dồn rất nhanh.\n",
            "if n_gpu > 1:\n",
            "    print('  >1 GPU — truyền --single-gpu cho MỌI lệnh train')\n",
            "\n",
            "# T4 là Turing (sm_75), KHÔNG có bf16 phần cứng. torch vẫn có thể báo\n",
            "# is_bf16_supported()=True vì hỗ trợ qua emulation, chậm hơn fp16.\n",
            "# Giữ fp16 bất kể giá trị này.\n",
            "print('bf16 (emulated trên T4, vẫn dùng fp16):', torch.cuda.is_bf16_supported())\n",
        ],
    ),
    (
        "code",
        [
            "# ===== Cell 2: copy code + data vào /kaggle/working =====\n",
            "# Dataset chỉ đọc, mà code ghi checkpoint và dùng đường dẫn tương đối, nên\n",
            "# phải copy sang thư mục ghi được. Dùng path đã dò ở Cell 0.\n",
            "import shutil\n",
            "\n",
            "WORK = Path('/kaggle/working')\n",
            "# `scripts` cần thiết: benchmark_biencoder.py chạy trên Kaggle.\n",
            "for name in ('src', 'configs', 'scripts'):\n",
            "    target = WORK / name\n",
            "    if target.exists():\n",
            "        shutil.rmtree(target)\n",
            "    shutil.copytree(SRC_ROOT / name, target)\n",
            "\n",
            "# Dataset data bắt đầu thẳng bằng method2/ custom_vi/ benchmark_vi/ (KHÔNG có\n",
            "# tầng `data/`), còn code tham chiếu `data/method2/...` → copy vào data/.\n",
            "data_dir = WORK / 'data'\n",
            "if data_dir.exists():\n",
            "    shutil.rmtree(data_dir)\n",
            "data_dir.mkdir(parents=True)\n",
            "for child in DATA_ROOT.iterdir():\n",
            "    dest = data_dir / child.name\n",
            "    shutil.copytree(child, dest) if child.is_dir() else shutil.copy2(child, dest)\n",
            "\n",
            "%cd /kaggle/working\n",
            "\n",
            "import json, glob, sys\n",
            "sys.path.insert(0, '/kaggle/working')\n",
            "# HF_HOME đã set ở Cell 0, kế thừa sang mọi tiến trình con `!python`.\n",
            "\n",
            "print('src    :', sorted(p.name for p in (WORK / 'src').iterdir()))\n",
            "print('data   :', sorted(p.name for p in data_dir.iterdir()))\n",
        ],
    ),
    (
        "code",
        [
            "# ===== Cell 3: kiểm tra bản copy TRƯỚC khi preflight =====\n",
            "# Preflight kiểm tra tính đúng đắn của dữ liệu; cell này kiểm tra bước copy —\n",
            "# tách ra để khi hỏng thì biết ngay là hỏng ở đâu.\n",
            "REQUIRED = [\n",
            "    'data/method2/decontamination.json',\n",
            "    'data/method2/manifest.json',\n",
            "    'data/method2/tool_pool.json',\n",
            "    'data/method2/biencoder/train.jsonl',\n",
            "    'data/method2/biencoder/val.jsonl',\n",
            "    'data/method2/biencoder/pairs_stats.json',\n",
            "    'data/method2/crossencoder/train.jsonl',\n",
            "    'data/method2/crossencoder/val.jsonl',\n",
            "    'data/method2/label_stats.json',\n",
            "    'data/custom_vi/v1/test_seen.jsonl',\n",
            "    'data/benchmark_vi/test.jsonl',\n",
            "    'configs/method2/biencoder.yaml',\n",
            "    'configs/method2/pinned_versions.json',\n",
            "    'src/models/preflight.py',\n",
            "]\n",
            "missing = []\n",
            "for rel in REQUIRED:\n",
            "    path = WORK / rel\n",
            "    if path.exists() and path.stat().st_size > 0:\n",
            "        print(f'  {path.stat().st_size / 1024**2:8.2f} MB  {rel}')\n",
            "    else:\n",
            "        missing.append(rel)\n",
            "        print(f'  {\"THIẾU\":>11}  {rel}')\n",
            "assert not missing, f'Copy chưa đủ: {missing}'\n",
            "\n",
            "# import được thì mới chắc src/ copy nguyên vẹn.\n",
            "import importlib\n",
            "\n",
            "importlib.import_module('src.models.preflight')\n",
            "manifest = json.load(open('data/method2/manifest.json', encoding='utf-8'))\n",
            "print('\\nsnapshot commit:', manifest.get('git_commit'))\n",
            "print('copy OK')\n",
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


PREFLIGHT_CELLS = [
    (
        "markdown",
        [
            "## Pre-flight — cổng fail-closed TRƯỚC mọi training\n",
            "\n",
            "```\n",
            "decontamination.json tồn tại\n",
            "        ↓\n",
            "SHA-256 == manifest.json\n",
            "        ↓\n",
            "overlap train/val/test == 0\n",
            "        ↓\n",
            "unseen positive leakage == 0\n",
            "        ↓\n",
            "package versions khớp bản đã pin\n",
            "        ↓\n",
            "CHO PHÉP TRAIN\n",
            "```\n",
            "\n",
            "Thiếu file hoặc hash lệch → job dừng ngay, **không rebuild tự động**. Nếu\n",
            "experiment chính tự dựng lại index từ dữ liệu đang có trên máy thì ta mất\n",
            "đúng thứ cần đảm bảo: bằng chứng model được train trên đúng split đã kiểm\n",
            "định. Rebuild là lệnh preprocessing riêng, chạy ở local rồi upload lại:\n",
            "`python -m src.models.sources decontaminate && python -m src.models.sources manifest`\n",
            "\n",
            "Vì sao `val ∩ test` là rủi ro nặng nhất: dù không train trên query đó, việc\n",
            "chọn checkpoint/hyperparameter bằng val vẫn khiến metric test lạc quan hơn\n",
            "thực tế. `data/benchmark_vi` **giữ nguyên** — decontamination nằm ở tầng\n",
            "dataset của Method 2 nên bốn method vẫn được đánh giá trên cùng một tập test.\n",
        ],
    ),
    (
        "code",
        [
            "# Exit code != 0 → dừng notebook, không chạy tiếp cell training nào.\n",
            "!python -m src.models.preflight \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --require-gpu T4 \\\n",
            "    --output results/method2/preflight.json\n",
            "\n",
            "preflight = json.load(open('results/method2/preflight.json', encoding='utf-8'))\n",
            "assert preflight['passed'], f\"Preflight KHÔNG ĐẠT: {preflight['failures']}\"\n",
            "print('preflight PASS —', len(preflight['checks']), 'check')\n",
        ],
    ),
    (
        "code",
        [
            "# Số liệu split để đối chiếu bằng mắt trước khi tiêu giờ GPU.\n",
            "stats = json.load(open('data/method2/biencoder/pairs_stats.json', encoding='utf-8'))\n",
            "decon = stats['decontamination']\n",
            "\n",
            "print('unique query/split :', stats['unique_queries_per_split'])\n",
            "print('positive pairs     :', stats['n_positive_pairs'])\n",
            "print('negative samples   :', stats['n_negative_samples'])\n",
            "print('query trùng split  :', decon['n_overlapping_queries'], decon['overlapping_queries'])\n",
            "print('sample bị loại     :', decon['rows_dropped_total'], decon['rows_dropped_by_transition'])\n",
            "print('overlap còn lại    :', stats['split_overlap_after'])\n",
        ],
    ),
]

RUN0_CELLS = PREFLIGHT_CELLS + [
    (
        "markdown",
        [
            "## Kết thúc Run 0\n",
            "\n",
            "Run 0 **không train gì cả** — chỉ xác nhận package, artefact và split đúng\n",
            "như đã kiểm định ở local. Đạt hết thì thoát, sang notebook Bi-Encoder chạy\n",
            "Run 1 (smoke) rồi Run 2 (full).\n",
        ],
    ),
    (
        "code",
        [
            "import pprint\n",
            "\n",
            "for check in preflight['checks']:\n",
            "    mark = 'PASS' if check['passed'] else 'FAIL'\n",
            "    detail = check['detail']\n",
            "    suffix = f' — {detail}' if detail else ''\n",
            "    print('[' + mark + '] ' + check['name'] + suffix)\n",
            "\n",
            "print()\n",
            "pprint.pprint(json.load(open('data/method2/manifest.json', encoding='utf-8'))['derived'])\n",
        ],
    ),
]

BENCHMARK_CELLS = [
    (
        "markdown",
        [
            "# Run 1a — Benchmark cấu hình (BẮT BUỘC trước smoke)\n",
            "\n",
            "Lần chạy đầu trên T4 cho **475 s/step**: 100 step mất 13.2 giờ, một epoch\n",
            "mất 49 giờ, trong khi plan dự toán 50-70 phút/epoch. Lệch ~45× nên phải tìm\n",
            "cấu hình dùng được trước, đừng chạy tiếp smoke 100 step.\n",
            "\n",
            "Vì sao mỗi step đắt: `CachedMNRL` không phải một forward/backward bình\n",
            "thường. Effective batch 256, mỗi sample có anchor + positive + 4 negative →\n",
            "**1,536 lượt encode**. Chia mini_batch 8 thành 192 chunk, GradCache chạy\n",
            "**hai** pha (forward no-grad để cache, rồi forward+backward tính lại) →\n",
            "~384 lần gọi model mỗi step. Mỗi lần chỉ 8×192 = 1,536 token, quá nhỏ để lấp\n",
            "đầy T4 nên phần lớn thời gian là overhead — cộng thêm DataParallel giữa 2 GPU\n",
            "thì nhân lên tiếp.\n",
            "\n",
            "| Case | GPU | batch | mini | ckpt | đổi gì so với case trước |\n",
            "|---|---|---|---|---|---|\n",
            "| A | 1×T4 | 256 | 8 | on | tách ảnh hưởng DataParallel |\n",
            "| B | 1×T4 | 256 | 16 | on | nửa số lần gọi model |\n",
            "| C | 1×T4 | 256 | 32 | on | 1/4 số lần gọi model |\n",
            "| D | 1×T4 | 128 | 32 | on | giảm effective batch |\n",
            "| E | 1×T4 | 256 | 32 | off | tắt grad checkpointing |\n",
            "\n",
            "A→C chỉ đổi **tốc độ**. D đổi **chất lượng**: MNRL mạnh lên theo số in-batch\n",
            "negative, giảm batch là giảm negative — chỉ dùng khi A–C không đủ, và phải\n",
            "ghi rõ vào báo cáo.\n",
        ],
    ),
    (
        "code",
        [
            "# 5 step mỗi case, tắt eval (eval trên corpus 4,4k tool làm nhiễu số đo).\n",
            "# `sec_per_step` lấy từ `train_runtime` của HF nên KHÔNG gồm thời gian nạp\n",
            "# BGE-M3 — với run 5 step thì nạp model lấn át hoàn toàn wall-clock.\n",
            "!python scripts/method2/benchmark_biencoder.py \\\n",
            "    --steps 5 \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --output results/method2/benchmark_biencoder.json\n",
        ],
    ),
    (
        "markdown",
        [
            "### Chốt cấu hình\n",
            "\n",
            "Chọn case nhanh nhất mà VRAM còn an toàn (< ~13 GB để chừa chỗ cho eval),\n",
            "rồi ghi vào `configs/method2/biencoder.yaml` trước khi chạy smoke.\n",
            "\n",
            "Ngưỡng thực dụng: **> 60 s/step là chưa dùng được** — 370 step/epoch × 3\n",
            "epoch mà 60 s/step đã là 18 giờ, vượt quota tuần.\n",
        ],
    ),
    (
        "code",
        [
            "bench = json.load(open('results/method2/benchmark_biencoder.json', encoding='utf-8'))\n",
            "ok = [b for b in bench if b['ok'] and b['hours_per_epoch']]\n",
            "assert ok, 'Không case nào chạy được — xem log ở trên'\n",
            "\n",
            "# Chọn theo GIỜ/EPOCH, KHÔNG theo s/step: giảm effective batch làm s/step\n",
            "# đẹp hẳn lên nhưng số step mỗi epoch tăng đúng bấy nhiêu lần.\n",
            "best = min(ok, key=lambda b: b['hours_per_epoch'])\n",
            "print('nhanh nhất theo epoch:', best['case']['name'],\n",
            "      f\"{best['hours_per_epoch']:.2f} h/epoch,\",\n",
            "      f\"{best['sec_per_step']:.1f} s/step × {best['steps_per_epoch']:,} step,\",\n",
            "      f\"peak {best['peak_vram_mb']:.0f} MB\")\n",
            "print(f\"3 epoch ≈ {best['hours_3_epochs']:.1f} h\")\n",
            "\n",
            "# Ghi cấu hình thắng cuộc vào YAML để smoke và full train dùng chung.\n",
            "import yaml\n",
            "\n",
            "cfg_path = 'configs/method2/biencoder.yaml'\n",
            "cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))\n",
            "cfg['train']['batch_size'] = best['case']['batch_size']\n",
            "cfg['train']['mini_batch_size'] = best['case']['mini_batch_size']\n",
            "cfg['train']['gradient_checkpointing'] = best['case']['grad_checkpointing']\n",
            "yaml.safe_dump(cfg, open(cfg_path, 'w', encoding='utf-8'), allow_unicode=True, sort_keys=False)\n",
            "print('đã ghi vào', cfg_path)\n",
            "\n",
            "if best['hours_3_epochs'] > 6:\n",
            "    print('\\nVƯỢT NGÂN SÁCH Phase 2 (~4h):',\n",
            "          f\"{best['hours_3_epochs']:.1f} h cho 3 epoch\")\n",
            "    print('Thử tiếp: mini_batch 64, n_hard_negatives 4→2 (cắt ~33% compute),')\n",
            "    print('hoặc giảm số epoch, hoặc max_seq_length 192 → 128.')\n",
            "\n",
            "if best['peak_vram_mb'] > 9000:\n",
            "    print('\\nVRAM', f\"{best['peak_vram_mb']:.0f} MB đo khi TẮT eval.\",\n",
            "          'Chạy lại case này CÓ eval trước khi tin là an toàn —')\n",
            "    print('evaluator encode ~4,4k tool doc trong khi trạng thái train vẫn giữ VRAM.')\n",
            "if best['case']['batch_size'] != 256:\n",
            "    print('\\nLƯU Ý: effective batch giảm còn', best['case']['batch_size'],\n",
            "          '— đổi CHẤT LƯỢNG chứ không chỉ tốc độ, phải ghi vào báo cáo.')\n",
        ],
    ),
]

SMOKE_CELLS = [
    (
        "markdown",
        [
            "# Run 1 — Smoke run, 200 step\n",
            "\n",
            "Chưa chạy 3 epoch. Mục tiêu là xác minh môi trường trước khi tiêu ~4h T4:\n",
            "\n",
            "| Cần trả lời | Đọc ở đâu trong `train_report.json` |\n",
            "|---|---|\n",
            "| CUDA/fp16 hoạt động | `observed.fp16_enabled`, `observed.device` |\n",
            "| CachedMNRL + LoRA không OOM | chạy hết 200 step không lỗi |\n",
            "| effective batch đúng 256 | `observed.effective_batch_matches_config` |\n",
            "| VRAM thực tế | `peak_vram_mb` |\n",
            "| throughput thực tế | `observed.samples_per_sec`, `estimated_sec_per_epoch` |\n",
            "| tên metric của evaluator | `observed.evaluator_metric_names` |\n",
            "| checkpoint save/resume | mục kiểm tra resume bên dưới |\n",
            "\n",
            "Preset smoke **giữ nguyên** batch_size, mini_batch_size, fp16, LoRA và\n",
            "max_seq_length — đó chính là những thứ cần kiểm chứng. Chỉ đổi số step, độ\n",
            "dày eval/save và output_dir riêng (`smoke_run01`) để không lẫn vào run thật.\n",
        ],
    ),
    (
        "code",
        [
            "SMOKE = '/kaggle/working/artifacts/method2/biencoder/smoke_run01'\n",
            "!rm -rf {SMOKE}\n",
            "\n",
            "# Phase A: chạy 100 step rồi dừng — để có checkpoint làm mốc resume.\n",
            "!python -m src.models.biencoder.train train \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --single-gpu --smoke 100\n",
            "\n",
            "report_a = json.load(open(f'{SMOKE}/train_report.json', encoding='utf-8'))\n",
            "obs = report_a['observed']\n",
            "print('device            :', obs['device'])\n",
            "print('fp16              :', obs['fp16_enabled'], '| bf16:', obs['bf16_enabled'])\n",
            "print('effective batch   :', obs['effective_batch_size'], '| khớp config:', obs['effective_batch_matches_config'])\n",
            "print('mini batch        :', obs['mini_batch_size'])\n",
            "print('peak VRAM (MB)    :', report_a['peak_vram_mb'])\n",
            "print('steps/sec         :', obs['steps_per_sec'])\n",
            "print('samples/sec       :', obs['samples_per_sec'])\n",
            "print('ước tính sec/epoch:', obs['estimated_sec_per_epoch'])\n",
            "print('last_train_loss   :', obs['last_train_loss'])\n",
            "print('checkpoint đã ghi :', obs['checkpoints_written'])\n",
            "print('metric evaluator  :', obs['evaluator_metric_names'])\n",
            "\n",
            "# Ngoại suy thời lượng full training từ throughput đo được.\n",
            "n_full = sum(1 for _ in open('data/method2/biencoder/train.jsonl', encoding='utf-8'))\n",
            "steps_per_epoch = n_full / obs['effective_batch_size']\n",
            "hours = steps_per_epoch * 3 / obs['steps_per_sec'] / 3600\n",
            "print(f'\\nƯớc tính full training (3 epoch, {n_full} dòng): {hours:.2f} h')\n",
            "\n",
            "assert obs['effective_batch_matches_config'], (\n",
            "    f\"Effective batch {obs['effective_batch_size']} ≠ 256 — GradCache mất tác dụng, \"\n",
            "    'số in-batch negative không như thiết kế'\n",
            ")\n",
            "assert obs['fp16_enabled'], 'fp16 chưa bật — T4 không có bf16'\n",
            "assert obs['checkpoints_written'], 'Không có checkpoint nào được ghi'\n",
        ],
    ),
    (
        "markdown",
        [
            "### Kiểm tra resume\n",
            "\n",
            "Kaggle ngắt session giữa chừng là chuyện bình thường, nên resume phải hoạt\n",
            "động **trước** khi chạy job dài. Phase B tiếp tục từ checkpoint của Phase A\n",
            "lên 200 step; nếu resume đúng thì `completed_steps` phải là 200 chứ không\n",
            "phải 100 (tức không train lại từ đầu).\n",
        ],
    ),
    (
        "code",
        [
            "ckpts = sorted(glob.glob(f'{SMOKE}/checkpoint-*'), key=lambda p: int(p.split('-')[-1]))\n",
            "last = ckpts[-1]\n",
            "print('resume từ:', last)\n",
            "\n",
            "!python -m src.models.biencoder.train train \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --single-gpu --smoke 200 \\\n",
            "    --resume-from {last}\n",
            "\n",
            "report_b = json.load(open(f'{SMOKE}/train_report.json', encoding='utf-8'))\n",
            "obs_b = report_b['observed']\n",
            "print('resumed_from_step      :', obs_b['resumed_from_step'])\n",
            "print('completed_steps        :', obs_b['completed_steps'])\n",
            "print('steps_trained_this_run :', obs_b['steps_trained_this_run'])\n",
            "print('resume_verified        :', obs_b['resume_verified'])\n",
            "print('duration_sec           :', obs_b['duration_sec'])\n",
            "\n",
            "# `completed_steps == 200` KHÔNG phân biệt được resume với train lại từ\n",
            "# đầu — cả hai đều cho 200. Bằng chứng thật là số step chạy lần này.\n",
            "assert obs_b['resume_verified'], 'Không resume — trainer bắt đầu lại từ 0'\n",
            "assert obs_b['resumed_from_step'] == 100, f\"Resume sai mốc: {obs_b['resumed_from_step']}\"\n",
            "assert obs_b['completed_steps'] == 200\n",
            "assert obs_b['steps_trained_this_run'] == 100, (\n",
            "    f\"Chạy {obs_b['steps_trained_this_run']} step thay vì 100 — có thể đã train lại từ đầu\"\n",
            ")\n",
            "print('\\nRun 1 PASS — môi trường sẵn sàng cho full training')\n",
        ],
    ),
    (
        "markdown",
        [
            "### Chốt tên metric cho các run sau\n",
            "\n",
            "`evaluator_metric_names` ở trên là tên **thật** của phiên bản\n",
            "sentence-transformers đang chạy. Run 2 vẫn dùng last checkpoint làm model\n",
            "chính và lưu toàn bộ validation metric theo step; chỉ khi đã biết chắc khoá\n",
            "metric mới khai `train.metric_for_best_model` và bật `load_best_model_at_end`\n",
            "cho các run chính / multi-seed.\n",
        ],
    ),
    (
        "code",
        [
            "names = report_b['observed']['evaluator_metric_names']\n",
            "print('Khai vào configs/method2/biencoder.yaml → train.metric_for_best_model:')\n",
            "for name in names:\n",
            "    if name.endswith(('ndcg@10', 'recall@5', 'accuracy@1', 'mrr@10')):\n",
            "        print('   ', name)\n",
        ],
    ),
]

BIENCODER_CELLS = PREFLIGHT_CELLS + BENCHMARK_CELLS + SMOKE_CELLS + [
    (
        "markdown",
        [
            "# Run 2 — Full training\n",
            "\n",
            "Chỉ chạy sau khi Run 1a chốt được cấu hình và Run 1 pass mọi assert.\n",
        ],
    ),
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
            "import re\n",
            "\n",
            "RUN = '/kaggle/working/artifacts/method2/biencoder/run01'\n",
            "\n",
            "# sorted() theo tên là sai: 'checkpoint-1000' < 'checkpoint-500' theo thứ tự\n",
            "# chữ, nên sẽ resume nhầm checkpoint cũ hơn.\n",
            "ckpts = sorted(\n",
            "    glob.glob(f'{RUN}/checkpoint-*'),\n",
            "    key=lambda p: int(re.search(r'(\\d+)$', p).group(1)),\n",
            ")\n",
            "resume = ckpts[-1] if ckpts else None\n",
            "print('resume from:', resume or '(chưa có checkpoint — train từ đầu)')\n",
            "\n",
            "# KHÔNG truyền cờ khi không có checkpoint: `{None}` nội suy thành chuỗi\n",
            "# \"None\", HF Trainer coi đó là đường dẫn rồi đi tải từ Hub và chết vì offline.\n",
            "resume_arg = f'--resume-from {resume}' if resume else ''\n",
            "# --single-gpu: benchmark đã đo ở chế độ 1 GPU, để DataParallel bật lại thì\n",
            "# tốc độ thực tế khác hẳn con số đã chốt.\n",
            "!python -m src.models.biencoder.train train \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --output-dir {RUN} \\\n",
            "    --single-gpu {resume_arg}\n",
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
            "    --config configs/method2/biencoder.yaml --output-dir {RUN2} --single-gpu\n",
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
    (
        "markdown",
        [
            "## Run manifest — chốt lại toàn bộ mục audit\n",
            "\n",
            "Train xong mà không audit được thì coi như chưa train. Cell này gom: commit\n",
            "SHA (kèm cờ dirty), config YAML thực tế, fingerprint dataset + tool pool,\n",
            "query counts và overlap theo split, số positive/negative pair, checkpoint,\n",
            "best step + metric đã dùng để chọn, VRAM peak, thời lượng train, và\n",
            "Recall@1/@5/@10 + MRR. Thiếu mục nào thì `audit_complete.missing` liệt kê ra.\n",
            "\n",
            "`checkpoint_selection.available_metrics` cho biết tên metric thật của\n",
            "`InformationRetrievalEvaluator` ở phiên bản đang chạy — khai vào\n",
            "`train.metric_for_best_model` cho lần chạy sau.\n",
        ],
    ),
    (
        "code",
        [
            "!python -m src.models.run_manifest \\\n",
            "    --run-dir {RUN2} \\\n",
            "    --config configs/method2/biencoder.yaml \\\n",
            "    --stage biencoder \\\n",
            "    --report retrieval=results/method2/metrics/biencoder_val.json\n",
            "\n",
            "manifest = json.load(open(f'{RUN2}/run_manifest.json', encoding='utf-8'))\n",
            "missing = manifest['audit_complete']['missing']\n",
            "print('thiếu:', missing or 'không thiếu mục nào')\n",
            "print('Recall/MRR:', manifest.get('retrieval_gate', {}).get('metrics'))\n",
            "print('VRAM peak MB:', manifest['train']['peak_vram_mb'])\n",
            "print('thời lượng (giờ):', manifest['train']['duration_hours'])\n",
            "print('chọn checkpoint:', manifest['train']['checkpoint_selection'])\n",
            "assert not missing, f'Chưa đủ artifact để audit: {missing}'\n",
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
            "import re\n",
            "\n",
            "RUN = '/kaggle/working/artifacts/method2/crossencoder/run01'\n",
            "\n",
            "# Sắp theo SỐ step, không theo tên: 'checkpoint-1000' < 'checkpoint-500'\n",
            "# nếu so chuỗi.\n",
            "ckpts = sorted(\n",
            "    glob.glob(f'{RUN}/checkpoint-*'),\n",
            "    key=lambda p: int(re.search(r'(\\d+)$', p).group(1)),\n",
            ")\n",
            "resume = ckpts[-1] if ckpts else None\n",
            "print('resume from:', resume or '(chưa có checkpoint — train từ đầu)')\n",
            "\n",
            "# Không truyền cờ khi không có checkpoint — `{None}` nội suy thành \"None\".\n",
            "resume_arg = f'--resume-from {resume}' if resume else ''\n",
            "!python -m src.models.crossencoder.train \\\n",
            "    --config configs/method2/crossencoder.yaml \\\n",
            "    --output-dir {RUN} {resume_arg}\n",
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
            "import subprocess\n",
            "\n",
            "# subprocess thay vì `!` trong vòng lặp: exit code hiện ra rõ ràng nên một\n",
            "# lần chạy hỏng không bị trôi qua trong Save & Run All.\n",
            "for gold, tag in [('data/custom_vi/v1/test_seen.jsonl', 'custom_seen'),\n",
            "                  ('data/custom_vi/v1/test_unseen.jsonl', 'custom_unseen'),\n",
            "                  ('data/benchmark_vi/test.jsonl', 'benchmark')]:\n",
            "    for mode in ['pipeline', 'oracle']:\n",
            "        print(f'=== {tag} / {mode} ===', flush=True)\n",
            "        subprocess.run(\n",
            "            ['python', '-m', 'src.models.pipeline.method2',\n",
            "             '--config', 'configs/method2/pipeline.yaml',\n",
            "             '--gold', gold, '--mode', mode,\n",
            "             '--output-dir', f'results/method2/predictions/{tag}'],\n",
            "            check=True,\n",
            "        )\n",
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


#: Chuỗi bắt buộc phải có trong từng notebook.
#:
#: Từng có lần `SMOKE_CELLS` được định nghĩa nhưng quên nối vào `BIENCODER_CELLS`
#: — notebook sinh ra thiếu hẳn phần smoke/resume mà vẫn hợp lệ về cú pháp, nên
#: không ai phát hiện cho tới khi chạy thật trên Kaggle. Kiểm cú pháp là chưa đủ;
#: phải kiểm cả việc từng khối có mặt.
REQUIRED_MARKERS: dict[str, tuple[str, ...]] = {
    "method2_kaggle_run0_preflight.ipynb": (
        "find_root", "src.models.preflight", "Kết thúc Run 0",
    ),
    "method2_kaggle_biencoder.ipynb": (
        "find_root", "src.models.preflight",
        "benchmark_biencoder.py", "Run 1a",          # benchmark cấu hình
        "--smoke", "resume_verified", "steps_trained_this_run",  # smoke + resume
        "Run 2", "biencoder.train train", "biencoder.index",     # full training
        "evaluate calibrate", "run_manifest",
    ),
    "method2_kaggle_crossencoder.ipynb": (
        "crossencoder.dataset", "skip_rate", "crossencoder.train", "crossencoder.evaluate",
    ),
    "method2_kaggle_eval.ipynb": (
        "pipeline.method2", "oracle", "evaluation.cli", "latency",
    ),
}


def check_markers(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    return [m for m in REQUIRED_MARKERS.get(path.name, ()) if m not in source]


def build_notebooks() -> list[Path]:
    specs = [
        (
            "method2_kaggle_run0_preflight.ipynb",
            "Method 2 — Run 0: Pre-flight (không train)",
            "Cổng fail-closed trước mọi job training: artefact, SHA-256, split overlap, version, GPU. 0 giờ GPU training.",
            RUN0_CELLS,
            "preflight_run",
        ),
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
    problems: dict[str, list[str]] = {}
    for path in build_notebooks():
        missing = check_markers(path)
        status = "OK" if not missing else f"THIẾU {missing}"
        print(f"[notebooks] → {path}  {status}")
        if missing:
            problems[path.name] = missing
    if problems:
        raise SystemExit(f"Notebook thiếu khối bắt buộc: {problems}")
