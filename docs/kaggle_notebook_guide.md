# Kaggle Notebook Guide

Hướng dẫn chạy Method 1 trên Kaggle Dataset `phcthnho/tool-calling-vi-experiments`.

## Chuẩn Bị

Tạo Kaggle Notebook GPU, bật Internet, rồi attach dataset `phcthnho/tool-calling-vi-experiments`. Dữ liệu nằm tại `/kaggle/input/tool-calling-vi-experiments`.

Chạy mỗi cặp `(experiment, model)` trong một notebook riêng. Thứ tự: E0 4B, E1 4B, E2 4B, E4 4B, E5 4B, sau đó lặp E1/E2/E4/E5 với 2B. E3 chỉ chạy khi đã có checkpoint general-SFT độc lập.

| Experiment | Train | Model selection | Test |
|---|---:|---|---|
| E0 | Không SFT | Không dùng validation | Core EN/VI + CustomTools |
| E1 | 60,000 EN | Core EN validation | Core EN/VI + CustomTools |
| E2 | 60,000 VI | Core VI validation | Core EN/VI + CustomTools |
| E3 | 60,000 EN | Core EN validation | Core EN/VI + CustomTools |
| E4 | 30,000 EN + 30,000 VI | Core EN/VI validation | Core EN/VI + CustomTools |
| E5 | E4 + 5,600 CustomTools | Core + CustomTools validation | Core EN/VI + CustomTools |

## Cell 1: Khai Báo Run

```python
from pathlib import Path

EXPERIMENT = "e1"  # e0, e1, e2, e3, e4, e5
MODEL_ID = "Qwen/Qwen3.5-4B"  # Hoặc Qwen/Qwen3.5-2B

DATA_ROOT = Path("/kaggle/input/tool-calling-vi-experiments")
EXPERIMENT_DIR = DATA_ROOT / EXPERIMENT
WORK_DIR = Path("/kaggle/working")
RUN_NAME = f"{EXPERIMENT}_{MODEL_ID.rsplit('/', 1)[-1].lower()}"
OUTPUT_DIR = WORK_DIR / "checkpoints" / RUN_NAME

assert EXPERIMENT_DIR.is_dir(), f"Missing: {EXPERIMENT_DIR}"
```

## Cell 2: Lưu Manifest

```python
import json
import shutil

manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text())
print(json.dumps(manifest["train_counts"], indent=2))

RUN_DIR = WORK_DIR / "runs" / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(EXPERIMENT_DIR / "manifest.json", RUN_DIR / "data_manifest.json")
```

Expected train count: E1/E2/E3/E4 = 60,000; E5 = 65,600. E0 không có `train.jsonl`.

## Cell 3: Cài LLaMA-Factory

```python
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git /kaggle/working/LLaMA-Factory
!pip install -q -e "/kaggle/working/LLaMA-Factory[torch]"
!pip install -q -U transformers accelerate peft bitsandbytes datasets
!llamafactory-cli version
```

Nếu Qwen3.5 không load được, lưu error log và cập nhật LLaMA-Factory/Transformers. Không thay model âm thầm.

## Cell 4: Đăng Ký Dataset Cho LLaMA-Factory

Input đã format ShareGPT đúng prompt EN/VI:

```text
instruction/train_chat.jsonl
instruction/val_chat.jsonl
```

```python
import json
import shutil

LF_ROOT = Path("/kaggle/working/LLaMA-Factory")
LF_DATA = LF_ROOT / "data" / RUN_NAME
LF_DATA.mkdir(parents=True, exist_ok=True)

for split in ("train", "val"):
    shutil.copy2(
        EXPERIMENT_DIR / "instruction" / f"{split}_chat.jsonl",
        LF_DATA / f"{split}.jsonl",
    )

dataset_info = {
    f"{RUN_NAME}_train": {
        "file_name": "train.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    },
    f"{RUN_NAME}_val": {
        "file_name": "val.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    },
}
(LF_DATA / "dataset_info.json").write_text(
    json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
)
```

Không chạy Cell 4 cho E0.

## Cell 5: Config QLoRA SFT

Kaggle T4/P100 phù hợp QLoRA 4-bit. Giữ cùng seed, epoch target, data manifest và decoding cho 2B/4B trong cùng experiment.

```python
config = f"""
### model
model_name_or_path: {MODEL_ID}
trust_remote_code: true

### method
stage: sft
do_train: true
do_eval: true
finetuning_type: lora
quantization_bit: 4
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: {LF_DATA}
dataset: {RUN_NAME}_train
eval_dataset: {RUN_NAME}_val
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 2

### output
output_dir: {OUTPUT_DIR}
logging_steps: 10
save_strategy: epoch
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
num_train_epochs: 1.0
learning_rate: 5.0e-7
lr_scheduler_type: cosine
warmup_ratio: 0.05
gradient_checkpointing: true
optim: paged_adamw_32bit
fp16: true

### eval
eval_strategy: epoch
""".strip()

config_path = WORK_DIR / f"{RUN_NAME}.yaml"
config_path.write_text(config + "\n", encoding="utf-8")
print(config_path)
```

Xác nhận `template: qwen3_nothink` có trong phiên bản LLaMA-Factory đang cài. Nếu upstream đổi template cho Qwen3.5, chỉ thay template và lưu version/commit của tool trong output.

## Cell 6: Train E1/E2/E4/E5

```python
assert EXPERIMENT != "e0"
!llamafactory-cli train {config_path}

shutil.copy2(config_path, OUTPUT_DIR / "train_config.yaml")
shutil.copy2(RUN_DIR / "data_manifest.json", OUTPUT_DIR / "data_manifest.json")
```

E3 chỉ dùng checkpoint đã qua general SFT làm `model_name_or_path`, sau đó chạy 60,000 EN tool-calling examples như E1.

## E0 Zero-Shot

E0 không train và không dùng validation để chỉnh prompt hoặc decoding. Inference trực tiếp checkpoint gốc trên:

```text
e0/test_en.jsonl
e0/test_vi.jsonl
e0/custom_test_seen.jsonl
e0/custom_test_unseen.jsonl
```

Mọi run dùng deterministic decoding:

```text
do_sample=False
temperature=0
```

## Test Và Lưu Artifact

Tất cả E0-E5 test trên các file:

```text
test_en.jsonl
test_vi.jsonl
custom_test_seen.jsonl
custom_test_unseen.jsonl
```

Không đưa test vào train hoặc dùng test để chọn checkpoint. Lưu checkpoint, `train_config.yaml`, `data_manifest.json`, raw predictions, parsed predictions và metrics JSON/CSV vào `/kaggle/working/`. Trước khi session kết thúc, dùng `Save Version` để lưu Notebook Output hoặc publish checkpoint/log thành Kaggle Model/Dataset private.
