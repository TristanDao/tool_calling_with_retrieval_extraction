# Colab E1-E4 Training Guide

Guide nay dung de train Qwen3.5-2B hoac Qwen3.5-4B tren mot GPU Colab va
resume qua nhieu runtime. Colab runtime la tam thoi, vi vay data co the dat tren
Drive hoac Hugging Face, con checkpoint nen luu tren Drive.

## 1. Chon noi luu data

| Lua chon | Uu diem | Nhuoc diem | Khuyen nghi |
|---|---|---|---|
| Google Drive | De lam, khong can API key | I/O cham neu train truc tiep | Luu checkpoint; phu hop data nho/vua |
| Hugging Face Dataset | Download nhanh, tai lai de, phu hop nhieu runtime | Dataset private can HF token | Khuyen nghi cho data training co dinh |
| Kaggle Dataset | Data hien dang co san | Can Kaggle token va download lai moi runtime | Dung lam nguon backup |

Khong train truc tiep tu Drive. Hay copy experiment dang train vao `/content`
de doc data tren local disk. Chi copy checkpoint ve Drive sau moi stage.

## 2. Cai dat va mount Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
from pathlib import Path

EXPERIMENT = "e1"
MODEL_ID = "unsloth/Qwen3.5-2B"
DRIVE_ROOT = Path("/content/drive/MyDrive/tool_calling_vi")
DRIVE_DATA_ROOT = DRIVE_ROOT / "data"
DRIVE_RUN_DIR = DRIVE_ROOT / "runs" / f"{EXPERIMENT}_{MODEL_ID.rsplit('/', 1)[-1].lower()}"
LOCAL_DATA_ROOT = Path("/content/data")
LOCAL_RUN_DIR = Path("/content/run")
DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
```

Data tren Drive can co layout:

```text
MyDrive/tool_calling_vi/data/
└── e1/
    ├── manifest.json
    └── instruction/train_chat.jsonl
```

Copy data cua experiment dang train vao local:

```python
import shutil

shutil.copytree(
    DRIVE_DATA_ROOT / EXPERIMENT,
    LOCAL_DATA_ROOT / EXPERIMENT,
    dirs_exist_ok=True,
)
```

## 3. Tuy chon tai data tu Hugging Face

Neu data duoc upload vao Dataset repository, co the tai thang vao local disk.
Dataset public khong can token; dataset private can token cua Hugging Face.

```python
%pip install -q huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ThinhDao/tool-calling-vi-experiments",
    repo_type="dataset",
    local_dir=str(LOCAL_DATA_ROOT),
    allow_patterns=[f"{EXPERIMENT}/**"],
)
```

Neu repository private:

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))
```

Khong ghi token truc tiep vao notebook da chia se.

## 4. Tuy chon tai data tu Kaggle

Chi dung cach nay neu data chua duoc dua len Drive/Hugging Face.

```python
%pip install -q kagglehub
```

Can dat Kaggle token trong Colab Secrets voi ten `KAGGLE_API_TOKEN`, sau do:

```python
from google.colab import userdata
import os

os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
```

```python
import kagglehub

downloaded = kagglehub.dataset_download(
    "phcthnho/tool-calling-vi-experiments"
)
print(downloaded)
```

Dung thu muc dataset tra ve lam `DATA_ROOT`, hoac copy rieng `e1` vao
`/content/data/e1`. Kaggle download khong can thiet neu data da co tren Drive.

## 5. Cai dependency va kiem tra GPU

```python
%pip install -q "unsloth" "transformers>=5.2.0" "trl>=0.15" \
    "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

```python
import torch

print(torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
```

Colab Pro thuong chi co mot GPU. Khong dung `torchrun` va khong assert hai GPU.

## 6. Dinh nghia cac ham va logic training

Chay cell sau trong Colab de dinh nghia tokenizer, collator va ham `main`:

```python
import unsloth

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastLanguageModel


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(values[0] if values and isinstance(values[0], list) else values)


def tokenize_assistant_only(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_length: int,
) -> dict[str, list[int]]:
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full = tokenizer.apply_chat_template(
        messages,
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = token_ids(tokenizer, prompt)
    input_ids = token_ids(tokenizer, full)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Prompt is not a prefix for {row.get('id')}")
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"Assistant target was truncated for {row.get('id')}")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids):],
    }


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(item["input_ids"]) for item in features)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return {
            key: torch.tensor(
                [item[key] + [pad_value] * (max_length - len(item[key])) for item in features],
                dtype=torch.long,
            )
            for key, pad_value in (
                ("input_ids", pad_id),
                ("attention_mask", 0),
                ("labels", -100),
            )
        }


class StopAtStep(TrainerCallback):
    def __init__(self, stop_after_step: int | None):
        self.stop_after_step = stop_after_step

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.stop_after_step is not None and state.global_step >= self.stop_after_step:
            control.should_training_stop = True
        return control


def make_training_arguments(kwargs: dict[str, Any]) -> TrainingArguments:
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs["group_by_length"] = kwargs.pop("train_sampling_strategy") == "group_by_length"
        return TrainingArguments(**kwargs)


def tokenized_dataset(
    tokenizer: Any,
    path: Path,
    max_seq_length: int,
    limit: int | None,
    num_proc: int = 2,
) -> Dataset:
    raw_dataset = Dataset.from_text(str(path))
    if limit is not None:
        raw_dataset = raw_dataset.select(range(min(limit, len(raw_dataset))))

    def process_row(example: dict[str, Any]) -> dict[str, list[int]]:
        row = json.loads(example["text"])
        return tokenize_assistant_only(tokenizer, row, max_seq_length)

    return raw_dataset.map(
        process_row,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing dataset",
    )


def main(args_list: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--model-id", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=2)
    args = parser.parse_args(args_list)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.per_device_batch_size * args.gradient_accumulation_steps != 16:
        raise ValueError("Expected effective batch size 16 on one GPU")

    output_dir = Path(args.run_dir) / "checkpoint"
    train_path = Path(args.data_root) / args.experiment / "instruction/train_chat.jsonl"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_16bit=False,
        full_finetuning=False,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False
    train_dataset = tokenized_dataset(
        tokenizer,
        train_path,
        args.max_seq_length,
        args.train_limit,
        num_proc=args.num_proc,
    )
    training_kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": 1.0,
        "learning_rate": 5e-7,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_strategy": "steps",
        "logging_steps": 50,
        "logging_first_step": True,
        "disable_tqdm": True,
        "save_strategy": "steps",
        "save_steps": 250,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "optim": "adamw_8bit",
        "seed": args.seed,
        "fp16": True,
        "bf16": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "train_sampling_strategy": "group_by_length",
        "skip_memory_metrics": True,
    }
    trainer = Trainer(
        model=model,
        args=make_training_arguments(training_kwargs),
        train_dataset=train_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
        callbacks=[StopAtStep(args.stop_after_step)],
    )
    checkpoint = get_last_checkpoint(str(output_dir))
    print(
        f"world_size=1 effective_batch_size=16 resume_from={checkpoint} "
        f"stop_after_step={args.stop_after_step}"
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("training stage complete:", output_dir)
```

Script nay doc file JSONL bang `Dataset.from_text()`, dung `json.loads` va `.map()` voi
`remove_columns=["text"]` va `num_proc=2` de tokenize thanh dataset chi gom
`input_ids`, `attention_mask`, `labels`, mmap ra dia giup tranh xung dot schema va tiet kiem RAM toi da.

## 7. Resume checkpoint tu Drive

Chay cell nay moi khi bat dau runtime moi:

```python
import shutil

if (DRIVE_RUN_DIR / "checkpoint").exists():
    shutil.copytree(DRIVE_RUN_DIR, LOCAL_RUN_DIR, dirs_exist_ok=True)
```

Khong copy checkpoint cua model khac vao cung `LOCAL_RUN_DIR`.

## 8. Cau hinh single GPU

Effective batch size 16 duoc giu bang:

```python
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
TRAIN_LIMIT = None
STOP_AFTER_STEP = 750
```

Cong thuc:

```text
2 samples x 8 accumulation x 1 GPU = 16
```

## 9. Chay training

Goi truc tiep ham `main()` voi cac tham so da cau hinh de bat dau training:

```python
train_args = [
    "--experiment", EXPERIMENT,
    "--model-id", MODEL_ID,
    "--run-dir", str(LOCAL_RUN_DIR),
    "--data-root", str(LOCAL_DATA_ROOT),
    "--per-device-batch-size", str(PER_DEVICE_BATCH_SIZE),
    "--gradient-accumulation-steps", str(GRADIENT_ACCUMULATION_STEPS),
    "--stop-after-step", str(STOP_AFTER_STEP),
    "--num-proc", "2",
]
if TRAIN_LIMIT is not None:
    train_args.extend(["--train-limit", str(TRAIN_LIMIT)])

main(train_args)
```

Trial dau tien nen dung `TRAIN_LIMIT = 256` va `STOP_AFTER_STEP = 100`.
Khi trial pass, dung `TRAIN_LIMIT = None`.

> **Luu y khi chay lai (re-run)**: Neu can chay lai cell hoac tiep tuc stage khac tren cung runtime, hay giai phong bo nho VRAM GPU truoc:
> ```python
> import gc, torch
> try:
>     del trainer, model
> except NameError:
>     pass
> gc.collect()
> torch.cuda.empty_cache()
> ```

## 10. Luu checkpoint len Drive

Chi chay sau khi training stage ket thuc:

```python
import subprocess

subprocess.run([
    "rsync", "-a", "--delete",
    f"{LOCAL_RUN_DIR}/",
    f"{DRIVE_RUN_DIR}/",
], check=True)
```

`--delete` dam bao Drive khong giu lai checkpoint cu da bi Trainer xoa boi
`save_total_limit`. Khong chi copy adapter; phai giu ca `optimizer.pt`,
`scheduler.pt`, `trainer_state.json` va `rng_state.pth` trong cac thu muc
`checkpoint/checkpoint-*`.

## 11. Cac stage resume

Voi E1/E2/E3:

```text
750 → 1500 → 2250 → 3000 → 3750
```

Voi E4:

```text
750 → 1500 → 2250 → 3000 → 3750 → 4100
```

`STOP_AFTER_STEP` la moc tuyet doi. Neu checkpoint moi nhat la 750 va dat
`STOP_AFTER_STEP = 1500`, trainer chi train tiep tu step 751 den 1500.

Moi experiment va moi model can mot `DRIVE_RUN_DIR` rieng. Khong resume Qwen3.5-2B
tu checkpoint Qwen3.5-4B.

## 12. Kiem tra sau moi stage

```python
from pathlib import Path

checkpoints = sorted(
    (LOCAL_RUN_DIR / "checkpoint").glob("checkpoint-*"),
    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
)
print([path.name for path in checkpoints])
assert checkpoints
```

Khi runtime bi ngat truoc khi copy Drive, moi thay doi sau checkpoint gan nhat
se mat. Vi vay nen chon stage 500–750 steps thay vi dat sat gioi han runtime.
