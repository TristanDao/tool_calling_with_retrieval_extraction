# HƯỚNG DẪN TÍCH HỢP & XÂY DỰNG DEMO TOOL CALLING (SLM QWEN3.5)

> **Mục đích**: Tài liệu bàn giao kỹ thuật dành riêng cho thành viên phát triển giao diện Demo / Trajectory (chưa trực tiếp tham gia quá trình huấn luyện SLM).

---

## 1. Bản chất: Mô hình này làm được gì?

Mô hình này là **Qwen3.5-2B** đã được tinh chỉnh (SFT LoRA) trên tập dữ liệu Tool Calling song ngữ (Anh - Việt).
* **Nhiệm vụ của mô hình**: Khi nhận câu hỏi của người dùng kèm danh sách các công cụ khả dụng (`tools`), mô hình sẽ:
  1. Đọc hiểu câu truy vấn tiếng Việt / tiếng Anh.
  2. **Tự động chọn đúng công cụ** cần dùng (Tool Selection).
  3. **Tự động trích xuất đúng tham số** từ câu nói của người dùng và điền vào schema của công cụ (Parameter Extraction).
  4. Nếu câu nói thông thường không cần công cụ (negative sample), mô hình sẽ trò chuyện tự nhiên như bình thường.

---

## 2. Kiến trúc Luồng Thực thi (Trajectory Flow)

Để demo trực quan quá trình "Agent suy nghĩ $\to$ gọi công cụ $\to$ nhận kết quả $\to$ trả lời", luồng xử lý (Trajectory) gồm các bước:

```
[1. User Query] + [Danh sách Tool Schemas]
            │
            ▼
[2. Model Inference] ──> Sinh văn bản thô (chứa thẻ <tool_call>)
            │
            ▼
[3. Hàm Parse] ────────> Bóc tách thành Python Dict: tool_name & arguments
            │
            ├── (Nếu KHÔNG gọi tool): Hiển thị câu chat tự nhiên ──> Kết thúc.
            │
            ▼ (Nếu CÓ gọi tool)
[4. Tool Execution] ───> Gọi hàm Python thật / mock data ──> Nhận kết quả
            │
            ▼
[5. Final Response] ───> Nạp kết quả tool vào ngữ cảnh để Model sinh câu trả lời hoàn chỉnh
```

---

## 3. Cài đặt Thư viện Môi trường

Chỉ cần cài đặt các thư viện PyTorch và Hugging Face chuẩn (chạy được trên mọi hệ điều hành Windows, Linux, Mac, Google Colab):

```bash
pip install torch transformers peft accelerate
```

---

## 4. Nạp Mô hình & Ghép Adapter (Model Loading)

Không cần cài đặt Unsloth (để tránh lỗi xung đột phiên bản CUDA/C++ trên máy cá nhân). Nạp trực tiếp qua `transformers` + `peft`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Khai báo repo
base_model_id = "unsloth/Qwen3.5-2B"
adapter_model_id = "ThinhDao/Qwen3.5-2B_E4"  # Repo LoRA adapter trên Hugging Face

print("📥 Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(adapter_model_id)

print("📥 Đang tải Base Model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

print("📥 Đang nạp LoRA Adapter vào mô hình...")
model = PeftModel.from_pretrained(model, adapter_model_id)
model.eval()
print("✅ Mô hình đã sẵn sàng!")
```

---

## 5. Chuẩn bị Input: Prompt & Danh sách Tool

### Định dạng Tool Schema chuẩn (JSON Schema)
Mỗi công cụ được định nghĩa dưới dạng một `dict`:

```python
demo_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_tutors",
            "description": "Tìm kiếm gia sư theo môn học và khu vực tỉnh/thành phố.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Môn học cần tìm (Toán, Lý, Hóa,...)"},
                    "location": {"type": "string", "description": "Thành phố hoặc tỉnh thành"}
                },
                "required": ["subject", "location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Tra cứu thông tin dự báo thời tiết tại một địa điểm cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Tên thành phố"}
                },
                "required": ["city"]
            }
        }
    }
]
```

### Cách áp dụng Chat Template
Dùng hàm native của tokenizer để ghép danh sách công cụ và câu hỏi vào đúng định dạng prompt:

```python
messages = [
    {"role": "user", "content": "Tôi muốn tìm gia sư dạy Toán ở Hà Nội"}
]

prompt_text = tokenizer.apply_chat_template(
    messages,
    tools=demo_tools,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

---

## 6. Đầu ra của Mô hình & Tại sao BẮT BUỘC cần hàm Parse?

### Đầu ra thô của mô hình (Raw Output)
Khi model quyết định gọi công cụ, nó sẽ trả ra chuỗi văn bản XML đặc trưng:
```xml
<tool_call>
<function=search_tutors>
<parameter=subject>Toán</parameter>
<parameter=location>Hà Nội</parameter>
</function>
</tool_call>
```

### Nhiệm vụ của hàm Parse
Chuyển đổi chuỗi XML thô ở trên thành Object/Dict có thể lập trình được:
```python
[
    {
        "name": "search_tutors",
        "arguments": {
            "subject": "Toán",
            "location": "Hà Nội"
        }
    }
]
```
Nếu **không parse**, mã nguồn Python sẽ không biết gọi hàm nào và truyền biến gì vào hàm để tiếp tục vẽ Trajectory!

---

## 7. Mã nguồn Hàm Parse Hoàn Chỉnh (Copy dùng ngay)

Dưới đây là hàm parse chuẩn xác đã được sử dụng xuyên suốt trong nghiên cứu:

```python
import json
import re
from dataclasses import dataclass
from typing import Any

_TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL | re.IGNORECASE)
_PARAMETER_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)

@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]

def parse_parameter_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except Exception:
        return stripped

def parse_model_tool_calls(text: str) -> list[ParsedToolCall]:
    """Bóc tách chuỗi XML của Qwen thành danh sách các ParsedToolCall."""
    calls: list[ParsedToolCall] = []
    
    # Tìm các khối <tool_call>...</tool_call>
    for block in _TOOL_BLOCK_RE.findall(text):
        for func_match in _FUNCTION_RE.finditer(block):
            tool_name = func_match.group(1).strip()
            body = func_match.group(2)
            
            args = {}
            for param_match in _PARAMETER_RE.finditer(body):
                param_key = param_match.group(1).strip()
                param_val = parse_parameter_value(param_match.group(2))
                args[param_key] = param_val
                
            calls.append(ParsedToolCall(name=tool_name, arguments=args))
            
    return calls
```

---

## 8. Kịch bản Thực thi Demo Hoàn Chỉnh (End-to-End Trajectory)

Đoạn script Python hoàn chỉnh mô phỏng toàn bộ hành trình xử lý một câu truy vấn:

```python
import json
import torch

# 1. Các hàm Python thực thi công cụ (Mock Execution)
def execute_tool(name: str, arguments: dict):
    if name == "search_tutors":
        return {
            "status": "success",
            "results": [
                {"id": 1, "name": "Thầy Nam", "subject": arguments.get("subject"), "area": arguments.get("location"), "rate": "300k/buổi"},
                {"id": 2, "name": "Cô Lan", "subject": arguments.get("subject"), "area": arguments.get("location"), "rate": "250k/buổi"}
            ]
        }
    elif name == "get_weather":
        return {"status": "success", "city": arguments.get("city"), "temperature": "28°C", "condition": "Nhiều mây"}
    return {"status": "error", "message": f"Không tìm thấy công cụ {name}"}

# 2. Quy trình chạy Trajectory
def run_agent_trajectory(user_query: str):
    trajectory_steps = []
    
    # BƯỚC 1: Nhận câu hỏi
    trajectory_steps.append({"step": "1. User Input", "content": user_query})
    messages = [{"role": "user", "content": user_query}]
    
    # Chuẩn bị Prompt
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=demo_tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # BƯỚC 2: Mô hình suy luận lần 1
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.01,
            do_sample=False,
        )
    in_len = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(output_ids[0][in_len:], skip_special_tokens=True).strip()
    
    trajectory_steps.append({"step": "2. Raw Model Generation", "content": raw_output})
    
    # BƯỚC 3: Parse kết quả
    tool_calls = parse_model_tool_calls(raw_output)
    
    if not tool_calls:
        # Trường hợp trò chuyện thông thường (không cần gọi công cụ)
        trajectory_steps.append({"step": "3. Action", "content": "Không cần gọi tool. Trả lời trực tiếp."})
        trajectory_steps.append({"step": "4. Final Response", "content": raw_output})
        return trajectory_steps
    
    # Trường hợp có gọi công cụ
    trajectory_steps.append({
        "step": "3. Parsed Tool Calls", 
        "content": [{"tool": call.name, "arguments": call.arguments} for call in tool_calls]
    })
    
    # BƯỚC 4: Thực thi công cụ (Execution & Observation)
    tool_observations = []
    for call in tool_calls:
        result = execute_tool(call.name, call.arguments)
        tool_observations.append({"tool": call.name, "result": result})
        
    trajectory_steps.append({"step": "4. Tool Execution Results (Observation)", "content": tool_observations})
    
    # BƯỚC 5: Tổng hợp câu trả lời cuối cùng
    # Đưa kết quả thực thi vào ngữ cảnh tiếp theo
    followup_messages = [
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": raw_output},
    ]
    for obs in tool_observations:
        followup_messages.append({
            "role": "tool",
            "name": obs["tool"],
            "content": json.dumps(obs["result"], ensure_ascii=False)
        })
        
    followup_prompt = tokenizer.apply_chat_template(
        followup_messages,
        tools=demo_tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs_2 = tokenizer(followup_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        final_ids = model.generate(**inputs_2, max_new_tokens=256, temperature=0.7)
    final_output = tokenizer.decode(final_ids[0][inputs_2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    trajectory_steps.append({"step": "5. Final User Answer", "content": final_output})
    
    return trajectory_steps
```

---

## 9. Gợi ý Hiển thị trên Giao diện (Gradio / Streamlit)

Khi thiết kế giao diện:
- Tạo một ô nhập câu hỏi (`Textbox`) và danh sách Checkbox/Dropdown chọn các tool đang kích hoạt.
- Kết quả đầu ra dùng component **Accordion** hoặc **Chatbot messages** để người xem có thể bấm mở ra xem từng bước:
  - 🟢 **Step 1 - Câu hỏi người dùng**: "Tìm gia sư Toán ở Hà Nội"
  - 🟡 **Step 2 - Quyết định hành động (Parsed Call)**: `search_tutors(subject="Toán", location="Hà Nội")`
  - 🔵 **Step 3 - Dữ liệu thực tế nhận được (Observation)**: Thầy Nam, Cô Lan...
  - 🟣 **Step 4 - Câu trả lời hoàn chỉnh**: "Tôi đã tìm thấy 2 gia sư dạy Toán phù hợp tại Hà Nội cho bạn: Thầy Nam (300k/buổi) và Cô Lan (250k/buổi)."
