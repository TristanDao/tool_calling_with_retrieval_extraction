# Kaggle E0 Evaluation Guide

Guide này chỉ dùng cho E0 zero-shot và đánh giá generation sau fine-tuning. E0
không train. Không dùng `torchrun`; mô hình 4-bit sẽ tự phân bổ trên 2x GPU T4
qua `accelerate` (`device_map="auto"`).

## Cell 1: Cau Hinh Va Doc Data

Thay `DATA_ROOT` bang dung duong dan Dataset tren Kaggle cua ban.

```python
import hashlib
import json
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/phcthnho/tool-calling-vi-experiments")
REVISION = "2026-09-02-full-dedup-seed42"
REVISION_DIR = DATA_ROOT / "benchmark_core" / REVISION
EXPERIMENT = "e0"
EXPERIMENT_DIR = DATA_ROOT / EXPERIMENT
MODEL_ID = "unsloth/Qwen3.5-4B"
RUN_NAME = "e0_qwen3.5-4b"
RUN_DIR = Path("/kaggle/working") / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

## Cell 2: Kiem Tra Dataset

```python
assert DATA_ROOT.is_dir(), DATA_ROOT
assert REVISION_DIR.is_dir(), REVISION_DIR
assert EXPERIMENT_DIR.is_dir(), EXPERIMENT_DIR

experiment_manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text(encoding="utf-8"))
revision_manifest = json.loads((REVISION_DIR / "manifest.json").read_text(encoding="utf-8"))
assert experiment_manifest["benchmark_revision"] == REVISION
assert revision_manifest["revision"] == REVISION
assert MODEL_ID in {
    experiment_manifest["checkpoint"]["primary"],
    experiment_manifest["checkpoint"]["alternative"],
}
for relative, expected in experiment_manifest["files"].items():
    assert sha256(EXPERIMENT_DIR / relative) == expected, relative

metadata = json.loads((REVISION_DIR / "metadata.json").read_text(encoding="utf-8"))
assert metadata["split_counts"]["vi"]["test"] == 7712
print("E0 preflight: PASS")
```

## Cell 3: Cai Dependency Va Kiem Tra GPU

Chay cell cai dat, restart kernel neu Kaggle yeu cau, roi chay lai tu Cell 1.

```python
%pip install -q "transformers>=5.2.0" "accelerate>=1.0" "peft>=0.14" "bitsandbytes>=0.43"
```

```python
import torch

assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator first"
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
```

## Cell 4: Native Qwen Helpers

```python
SYSTEM_PROMPTS = {
    "en": "You are an AI assistant capable of using tools.",
    "vi": "Bạn là trợ lý AI có khả năng sử dụng công cụ.",
}

def format_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        },
    }

def format_call(call: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": call.get("arguments", {}),
        },
    }

def native_row(record: dict, language: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPTS[language]},
        {"role": "user", "content": record["query"]},
    ]
    calls = [format_call(call) for call in record["function_calls"]]
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
    else:
        fallback = "I cannot complete that request right now."
        if language == "vi":
            fallback = "Hiện tại tôi chưa thể thực hiện yêu cầu này."
        messages.append({"role": "assistant", "content": record.get("assistant_content") or fallback})
    return {
        "id": record["id"],
        "messages": messages,
        "tools": [format_tool(tool) for tool in record["tools"]],
    }
```

## Cell 5: Tokenizer Smoke Test

```python
from transformers import AutoTokenizer

smoke_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
for record in read_jsonl(REVISION_DIR / "vi/test.jsonl")[:3]:
    row = native_row(record, "vi")
    rendered = smoke_tokenizer.apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False,
        add_generation_prompt=False, enable_thinking=False,
    )
    expected_calls = len(record["function_calls"])
    assistant_part = rendered.rsplit("<|im_start|>assistant", 1)[-1]
    assert assistant_part.count("<tool_call>") == expected_calls, record["id"]
print("native tokenizer smoke test: PASS")
```

## Cell 6: Nap Model (Native Transformers 4-bit)

E0 chỉ chạy generation/evaluation (không train), nên sử dụng trực tiếp Hugging Face `transformers` cùng `bitsandbytes` 4-bit NF4. Cách này tránh cảnh báo Unsloth fallback float32 trên GPU Tesla T4 và đảm bảo tính nhất quán chuẩn xác.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="left",
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.eval()
```

## Cell 7: Batched Generation (vLLM hoac PyTorch Toi Uu)

### Lua chon A: Dung vLLM (Khuyen nghi - Toc do vuot troi, tan dung ca 2 GPU T4)

vLLM ho tro Continuous Batching va tu dong phan bo tren ca 2 GPU T4 cua Kaggle (`tensor_parallel_size = 2`).

Hai lua chon A va B la doc lap. Neu da chay Cell 6, khong khoi tao vLLM trong
cung kernel khi model Transformers van dang chiem VRAM. Hay restart kernel
truoc khi chay vLLM, hoac giai phong model va CUDA cache:

```python
import gc
import torch

if "model" in globals():
    del model
gc.collect()
torch.cuda.empty_cache()
```

Tren Kaggle, mot GPU co the con it VRAM hon GPU con lai do tien trinh cu hoac
notebook khac. Vi vay khong dung `gpu_memory_utilization=0.90` mac dinh; gia
tri nay co the yeu cau nhieu VRAM hon so dang free tren mot GPU.

Kiem tra CUDA cua PyTorch truoc:

```python
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
```

Khong dung URL wheel vLLM tu GitHub voi version `0.29.0`: version nay khong
co release asset `+cu128`. Ngoai ra, ban vLLM moi ho tro Qwen3.5 dang dung
CUDA 13, trong khi Kaggle T4 cua guide nay dang dung CUDA 12.8. Neu import
vLLM bao loi `libcudart.so.13`, bo qua Lua chon A va dung Lua chon B (PyTorch
Native) ben duoi; khong tao symlink gia cho `libcudart.so.13`.

Sau khi cài hoặc nâng cấp vLLM, hãy restart kernel rồi mới import `torch` và
`vllm`. Không chạy lệnh cài đặt sau khi Cell 6 đã load model, vì việc thay đổi
Torch/vLLM giữa chừng có thể làm hỏng CUDA state của kernel.

Chay generation bang vLLM (chi dung neu import vLLM thanh cong):

```python
import json
import time
from pathlib import Path
import torch
from vllm import LLM, SamplingParams

EVALUATION_NAME = f"{RUN_NAME}_vi_test"
PREDICTIONS_PATH = RUN_DIR / f"eval_predictions_{EVALUATION_NAME}.jsonl"
EVAL_LIMIT = None  # Dat None de danh gia toan bo 7.712 mau test

test_records = read_jsonl(REVISION_DIR / "vi/test.jsonl")
if EVAL_LIMIT is not None:
    test_records = test_records[:EVAL_LIMIT]

def prompt_text(record: dict, language: str) -> str:
    row = native_row(record, language)
    return tokenizer.apply_chat_template(
        row["messages"][:-1],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

prompts = [prompt_text(record, "vi") for record in test_records]

# Khoi tao vLLM. Restart kernel truoc neu Cell 6 da duoc chay.
num_gpus = torch.cuda.device_count()
# vLLM TP cho guide nay can 2 GPU T4.
assert num_gpus >= 2, "vLLM TP cho guide nay can 2 GPU T4"
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,
    max_model_len=4096,
    gpu_memory_utilization=0.80,
    disable_custom_all_reduce=True,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    max_tokens=128,
    temperature=0.0,
    stop=["<|im_end|>", "<|endoftext|>"],
)

print(f"Bat dau sinh ket qua cho {len(prompts)} mau test bang vLLM...")
t0 = time.perf_counter()
outputs = llm.generate(prompts, sampling_params)
total_time = time.perf_counter() - t0
print(f"Hoan thanh trong {total_time:.1f}s ({len(prompts) / total_time:.1f} samples/s)!")

with PREDICTIONS_PATH.open("w", encoding="utf-8") as output_file:
    for record, output in zip(test_records, outputs, strict=True):
        raw_output = output.outputs[0].text
        output_file.write(json.dumps({
            "id": record["id"],
            "query": record["query"],
            "gold": record.get("function_calls", []),
            "raw_output": raw_output,
            "batch_latency_ms": round(total_time / len(prompts) * 1000.0, 2),
            "batch_size": 1,
        }, ensure_ascii=False) + "\n")

print("raw predictions:", PREDICTIONS_PATH)
```

Neu van gap loi `Free memory ... is less than desired GPU memory
utilization`, giam `gpu_memory_utilization` xuong `0.75` va dam bao khong co
Cell 6/model Transformers, vLLM process, hoac notebook khac dang chay. Cac
canh bao ve BF16, FlashAttention 2, FlashInfer va custom all-reduce tren Tesla
T4 la canh bao tuong thich, khong phai nguyen nhan lam engine dung; guide da
tat custom all-reduce de tranh thu nay.

---

### Lua chon B: Dung PyTorch Native (Toi uu gom theo do dai + Batch Size)

Neu khong muon cai `vllm`, cell duoi day da duoc toi uu:
1. Săp xep du lieu theo do dai prompt (`length-sorted`) de giam 80% padding thua.
2. Bat dung token dung `<|im_end|>`.
3. Dat `BATCH_SIZE = 16` tren 2x T4 (hoac 32 neu chi chay model 2B).

```python
import json
import os
import time
from pathlib import Path

BATCH_SIZE = 16  # 16-32 tren 2x T4 voi Qwen3.5-2B, 4-8 voi Qwen3.5-4B
MAX_NEW_TOKENS = 128
EVAL_LIMIT = None  # Dat None de danh gia toan bo 7.712 mau test
EVALUATION_NAME = f"{RUN_NAME}_vi_test"
PREDICTIONS_PATH = RUN_DIR / f"eval_predictions_{EVALUATION_NAME}.jsonl"

def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as source:
        return {json.loads(line)["id"] for line in source if line.strip()}

def prompt_text(record: dict, language: str) -> str:
    row = native_row(record, language)
    return tokenizer.apply_chat_template(
        row["messages"][:-1],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

test_records = read_jsonl(REVISION_DIR / "vi/test.jsonl")
if EVAL_LIMIT is not None:
    test_records = test_records[:EVAL_LIMIT]
completed_ids = read_completed_ids(PREDICTIONS_PATH)
pending_records = [record for record in test_records if record["id"] not in completed_ids]

# Toi uu 1: Sap xep theo do dai prompt
pending_records.sort(key=lambda r: len(r.get("query", "")) + len(str(r.get("tools", []))))

print(f"completed={len(completed_ids)}, pending={len(pending_records)}")

# Toi uu 2: Bat token dung
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
stop_ids = [tokenizer.eos_token_id]
if im_end_id is not None and im_end_id not in stop_ids:
    stop_ids.append(im_end_id)

with PREDICTIONS_PATH.open("a", encoding="utf-8") as output_file:
    for start in range(0, len(pending_records), BATCH_SIZE):
        batch = pending_records[start : start + BATCH_SIZE]
        prompts = [prompt_text(record, "vi") for record in batch]
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0
        generated_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        for record, raw_output in zip(batch, texts, strict=True):
            output_file.write(json.dumps({
                "id": record["id"],
                "query": record["query"],
                "gold": record.get("function_calls", []),
                "raw_output": raw_output,
                "batch_latency_ms": round(batch_latency_ms, 2),
                "batch_size": len(batch),
            }, ensure_ascii=False) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())

        done = len(completed_ids) + start + len(batch)
        batch_idx = start // BATCH_SIZE + 1
        total_batches = (len(pending_records) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Batch {batch_idx}/{total_batches} | Da xong: {done}/{len(test_records)} ({done / len(test_records) * 100:.1f}%) | Latency: {batch_latency_ms:.0f}ms")

print("raw predictions:", PREDICTIONS_PATH)
```

Kiem tra `raw_output` cua 20 mau dau. Neu native positive output co
`<tool_call><function=...>` va khong bi cat, dat `EVAL_LIMIT = None` va chay lai
cung cell. Cac ID da ghi se duoc bo qua.

## Cell 8: Cham Metric Tu Prediction JSONL

Cell nay khong generate token. No co the cham preview 20 mau sau smoke test,
nhung metric chi la ket qua chinh thuc khi `is_complete` la `true`.

```python
import re

TOOL_RE = re.compile(
    r"<tool_call>\s*<function\s*=\s*([^>\s]+)\s*>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
PARAM_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)

def parse_native_output(text: str) -> tuple[list[dict], list[str]]:
    calls = []
    errors = []
    open_tags = len(re.findall(r"<tool_call\b", text, re.IGNORECASE))
    close_tags = len(re.findall(r"</tool_call\s*>", text, re.IGNORECASE))
    if open_tags != close_tags:
        errors.append("unbalanced_tool_call_tags")
    for name, body in TOOL_RE.findall(text):
        arguments = {}
        for parameter, value in PARAM_RE.findall(body):
            value = value.strip()
            try:
                arguments[parameter.strip()] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                arguments[parameter.strip()] = value
        calls.append({"name": name.strip(), "arguments": arguments})
    if not calls and open_tags > 0:
        errors.append("malformed_tool_call")
    return calls, errors

rows = read_jsonl(PREDICTIONS_PATH)
assert len({row['id'] for row in rows}) == len(rows), "Duplicate prediction IDs"
expected_ids = {record["id"] for record in read_jsonl(REVISION_DIR / "vi/test.jsonl")}
assert {row["id"] for row in rows} <= expected_ids, "Prediction contains an unknown test ID"
is_complete = len(rows) == len(expected_ids)
if not is_complete:
    print(f"PREVIEW ONLY: {len(rows)}/{len(expected_ids)} predictions. Set EVAL_LIMIT = None and resume generation for final metrics.")

positive = negative = tool_correct = negative_correct = exact_match = syntax_errors = 0
latency_ms = 0.0
scored_rows = []
for row in rows:
    predicted, errors = parse_native_output(row["raw_output"])
    gold = row["gold"]
    is_positive = bool(gold)
    tool_match = [call["name"] for call in predicted] == [call["name"] for call in gold]
    exact = predicted == gold
    if is_positive:
        positive += 1
        tool_correct += tool_match
    else:
        negative += 1
        negative_correct += not predicted and not errors
    exact_match += exact
    syntax_errors += bool(errors)
    latency_ms += row["batch_latency_ms"] / row["batch_size"]
    scored_rows.append({
        **row,
        "predicted": predicted,
        "errors": errors,
        "tool_match": tool_match if is_positive else (not predicted and not errors),
        "exact_match": exact,
    })

metrics = {
    "split": EVALUATION_NAME,
    "total_samples": len(rows),
    "expected_total_samples": len(expected_ids),
    "is_complete": is_complete,
    "positive_samples": positive,
    "negative_samples": negative,
    "tool_accuracy_pos_pct": round(100 * tool_correct / positive, 2) if positive else 0.0,
    "non_fc_recall_pct": round(100 * negative_correct / negative, 2) if negative else 0.0,
    "arga_exact_match_pct": round(100 * exact_match / len(rows), 2) if rows else 0.0,
    "syntax_error_rate_pct": round(100 * syntax_errors / len(rows), 2) if rows else 0.0,
    "avg_latency_ms": round(latency_ms / len(rows), 2) if rows else 0.0,
}
(RUN_DIR / f"eval_predictions_{EVALUATION_NAME}_scored.json").write_text(
    json.dumps(scored_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(RUN_DIR / f"eval_metrics_{EVALUATION_NAME}.json").write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
```

## Chon Batch Size

1. Chay 200 mau voi `BATCH_SIZE = 4` va ghi lai samples/second, peak VRAM tren
   tung GPU.
2. Neu peak duoi 13.5 GiB/GPU va khong OOM, thu `BATCH_SIZE = 8` tren cung 200
   mau. Xoa file prediction thu nghiem truoc khi chay benchmark chinh thuc.
3. Chi dung batch 8 neu samples/second tang. Neu khong tang, batch 4 nhanh hon
   trong thuc te do padding va dong bo giua hai GPU.
4. Khong tang `MAX_NEW_TOKENS` len 512. Chi tang len 192 neu kiem tra raw output
   cho thay tool call bi cat o 128 token.

Voi model da fine-tune, nap checkpoint thay cho `MODEL_ID` trong
`FastLanguageModel.from_pretrained()` va dung lai dung cell evaluation nay.
