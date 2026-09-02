# Kaggle Notebook Guide

Hướng dẫn chạy Method 1 trên Kaggle với Unsloth, native Qwen3.5 chat
template và một evaluation suite dùng chung.

## 1. Dataset Và Code

Upload một Kaggle Dataset bằng helper:

```bash
python scripts/data/upload_experiments_to_kaggle.py \
  <kaggle-username>/tool-calling-vi-experiments \
  --benchmark-revision data/benchmark_core/2026-09-02-full-dedup-seed42 \
  --custom-data data/custom_vi
```

Dataset sau khi upload có cấu trúc:

```text
e0/ ... e5/
benchmark_core/2026-09-02-full-dedup-seed42/
  en/{val,test}.jsonl
  vi/{val,test}.jsonl
custom_vi/{val_seen,val_unseen,test_seen,test_unseen}.jsonl
```

Notebook cần có source code của repository tại `/kaggle/working/tool-calling-vi`.
Có thể upload repository cùng notebook hoặc clone repository riêng; không copy
test/validation vào `e*`.

Chạy mỗi cặp `(experiment, model)` trong một notebook riêng. Thứ tự đề xuất:
E0, E1, E2, E4, E5 với 4B, sau đó lặp E1/E2/E4/E5 với 2B. E3 chỉ chạy khi
có checkpoint general-SFT độc lập.

| Experiment | Train | Validation để chọn checkpoint |
|---|---:|---|
| E0 | Không SFT | Không dùng |
| E1 | 60,000 EN | Core EN |
| E2 | 60,000 VI | Core VI |
| E3 | 60,000 EN | Core EN, sau general SFT |
| E4 | 30,000 EN + 30,000 VI | Macro rule trên core EN/VI |
| E5 | E4 + 5,600 CustomTools VI | Core + CustomTools validation |

## 2. Khai Báo Run

```python
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path("/kaggle/working/tool-calling-vi")
DATA_ROOT = Path("/kaggle/input/tool-calling-vi-experiments")
EXPERIMENT = "e1"                 # e0, e1, e2, e4, e5
MODEL_ID = "unsloth/Qwen3.5-4B"   # hoặc unsloth/Qwen3.5-2B
REVISION = "2026-09-02-full-dedup-seed42"

sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENT_DIR = DATA_ROOT / EXPERIMENT
REVISION_DIR = DATA_ROOT / "benchmark_core" / REVISION
CUSTOM_DIR = DATA_ROOT / "custom_vi"
RUN_NAME = f"{EXPERIMENT}_{MODEL_ID.rsplit('/', 1)[-1].lower()}"
RUN_DIR = Path("/kaggle/working/runs") / RUN_NAME
OUTPUT_DIR = Path("/kaggle/working/checkpoints") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

assert EXPERIMENT_DIR.is_dir(), EXPERIMENT_DIR
assert REVISION_DIR.is_dir(), REVISION_DIR
manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text())
assert manifest["benchmark_revision"] == REVISION
print(json.dumps(manifest["training"]["counts"], indent=2))
```

## 3. Cài Dependencies

```python
%pip install -q -e "/kaggle/working/tool-calling-vi[train]"
```

Nếu notebook không cài từ source repository, cài tối thiểu:

```python
%pip install -q "unsloth" "transformers>=5.2.0" "trl>=0.15" \
  "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

Khởi động lại kernel nếu Kaggle đã load một phiên bản cũ của `torch` hoặc
`transformers`. Không thay checkpoint khi model load lỗi; lưu traceback và
phiên bản package trong run directory.

## 4. Native Template Smoke Test

Template không được chép từ model này sang model kia. Mỗi checkpoint tự cung
cấp `chat_template.jinja` và được truyền `tools=` khi render.

```python
import subprocess

subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train/smoke_native_qwen.py"),
        "--model", MODEL_ID,
        "--max-seq-length", "4096",
    ],
    cwd=PROJECT_ROOT,
    check=True,
)
```

Smoke test phải pass single-call, multi-call, list/number/boolean arguments
và negative response trước khi train.

## 5. Tạo Validation View On-demand

Validation chỉ được format tạm thời từ frozen revision; không lưu bản copy
trong `data/experiments/e*`.

```python
from src.data.convert_to_instruction import convert_file

validation_sources = {
    "e1": [(REVISION_DIR / "en/val.jsonl", "en")],
    "e2": [(REVISION_DIR / "vi/val.jsonl", "vi")],
    "e4": [
        (REVISION_DIR / "en/val.jsonl", "en"),
        (REVISION_DIR / "vi/val.jsonl", "vi"),
    ],
    "e5": [
        (REVISION_DIR / "en/val.jsonl", "en"),
        (REVISION_DIR / "vi/val.jsonl", "vi"),
        (CUSTOM_DIR / "val_seen.jsonl", "vi"),
        (CUSTOM_DIR / "val_unseen.jsonl", "vi"),
    ],
}

parts = []
for index, (source, language) in enumerate(validation_sources[EXPERIMENT]):
    part = RUN_DIR / f"val_{index}.jsonl"
    convert_file(source, part, language=language)
    parts.append(part)

EVAL_FILE = RUN_DIR / "eval_native.jsonl"
with EVAL_FILE.open("w", encoding="utf-8") as output:
    for part in parts:
        output.write(part.read_text(encoding="utf-8"))
```

Không tạo validation view cho E0. E3 cần prerequisite riêng và không được
giả lập bằng cách đổi tên E1.

## 6. Train Với Unsloth

`train_unsloth.py` nhận `instruction/train_chat.jsonl`, không nhận
`train.jsonl` master. Script gọi `apply_chat_template()` của tokenizer và
mask toàn bộ system/user tokens; loss chỉ tính trên assistant response.

```python
assert EXPERIMENT != "e0"
TRAIN_FILE = EXPERIMENT_DIR / "instruction/train_chat.jsonl"

command = [
    sys.executable,
    "-m", "src.models.slm.train_unsloth",
    "--train-file", str(TRAIN_FILE),
    "--eval-file", str(EVAL_FILE),
    "--output-dir", str(OUTPUT_DIR),
    "--model-name", MODEL_ID,
    "--max-seq-length", "4096",
    "--per-device-train-batch-size", "1",
    "--per-device-eval-batch-size", "1",
    "--gradient-accumulation-steps", "16",
    "--epochs", "1.0",
    "--learning-rate", "5e-7",
    "--lora-rank", "16",
    "--lora-alpha", "16",
    "--seed", "42",
]
subprocess.run(command, cwd=PROJECT_ROOT, check=True)
```

Giữ cùng seed, sample IDs, epoch target, sequence length và effective batch
size cho 2B/4B trong cùng experiment. QLoRA dùng 4-bit trên T4/P100; chỉ
đổi precision khi ghi rõ trong run config.

## 7. E0 Zero-shot Và Evaluation Chung

E0 dùng checkpoint gốc, không đọc validation để chỉnh prompt hoặc decoding.
Mọi experiment đánh giá trực tiếp trên các file shared sau:

```text
benchmark_core/<revision>/en/test.jsonl
benchmark_core/<revision>/vi/test.jsonl
custom_vi/test_seen.jsonl
custom_vi/test_unseen.jsonl
```

Inference dùng deterministic decoding (`do_sample=False`, `temperature=0`).
Native model output được chuẩn hóa bằng `src/evaluation/native_output.py`; không
dùng parser `<tool_call>{...}</tool_call>` của protocol cũ. Native positive call
có dạng `<tool_call><function=...>...</function></tool_call>`, còn negative là
assistant content bình thường không có tool call.

Lưu trong `/kaggle/working/`:

- checkpoint/adapter và tokenizer;
- `manifest.json` của experiment và manifest của benchmark revision;
- train config, raw output, parsed output, metrics JSON/CSV;
- latency, resource log và lỗi parser.

Không đưa bất kỳ CustomTools validation/test row nào vào training. Trước khi
đóng notebook, kiểm tra lại sample ID hash và lưu toàn bộ run artifact bằng
Kaggle Notebook Output hoặc Kaggle Model private.
