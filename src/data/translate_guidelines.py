"""Protect identifiers during translation.

Provides:
- Prompt template cho Qwen-MT (Anh → Việt) với quy tắc bảo vệ identifier.
- Regex patterns để validate identifier integrity sau dịch.
- Helper build_translate_prompt(sample, dataset) cho translate.py.
"""

from __future__ import annotations

import json
import re
from typing import Any

SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CAMEL_CASE_RE = re.compile(r"^[a-z][a-zA-Z0-9]*$")
_LIST_GENERIC_RE = re.compile(r"^List\[.*\]$", re.IGNORECASE)
_OPTIONAL_SUFFIX_RE = re.compile(r",\s*optional\s*$", re.IGNORECASE)


TRANSLATE_SYSTEM_PROMPT = """Bạn là chuyên gia dịch Anh-Việt cho dataset Tool Calling (function calling).

QUY TẮC BẮT BUỘC:
1. Dịch natural language (user query, assistant text, description, function response) sang tiếng Việt tự nhiên, mượt mà.
2. GIỮ NGUYÊN 100%:
   - Function name (snake_case, ví dụ: search_tutors, get_user_profile)
   - Argument keys (snake_case)
   - JSON structure, key names, thứ tự key
   - Identifier kỹ thuật: UUID, mã sản phẩm, mã đơn hàng
   - Tên thương hiệu, protocol, library (OpenAI, HTTP, REST, JSON, ...)
   - Mã tiền tệ (USD, VND, EUR, ...)
   - Số, đơn vị khoa học (25°C, 50kg, 3.14, ...)
   - Enum values, mã định danh
3. KHÔNG thêm key mới, KHÔNG bớt key, KHÔNG thay đổi JSON structure.
4. KHÔNG thêm giải thích, comment, hay text ngoài JSON.

Output: CHỈ trả về JSON đã dịch, không có text thừa."""


USER_PROMPT_TEMPLATE = """Translate the following JSON sample from English to Vietnamese.
Keep ALL keys, identifiers, function names, argument keys, JSON structure unchanged.
Only translate natural language values to Vietnamese.

```json
{sample_json}
```

Output ONLY the translated JSON object, no explanation, no markdown fences."""


def _indent_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_translate_prompt_glaive(sample: dict[str, Any]) -> str:
    payload = {
        "system": sample.get("system", ""),
        "chat": sample.get("chat", ""),
    }
    return USER_PROMPT_TEMPLATE.format(sample_json=_indent_json(payload))


def build_translate_prompt_xlam(sample: dict[str, Any]) -> str:
    payload = {
        "id": sample.get("id"),
        "query": sample.get("query", ""),
        "answers": sample.get("answers", ""),
        "tools": sample.get("tools", ""),
    }
    return USER_PROMPT_TEMPLATE.format(sample_json=_indent_json(payload))


def build_translate_prompt_normalized(sample: dict[str, Any]) -> str:
    return USER_PROMPT_TEMPLATE.format(sample_json=_indent_json(sample))


def build_translate_prompt(sample: dict[str, Any], dataset: str) -> str:
    if dataset == "glaive":
        return build_translate_prompt_glaive(sample)
    if dataset == "xlam":
        return build_translate_prompt_xlam(sample)
    if dataset in ("glaive_normalized", "xlam_normalized"):
        return build_translate_prompt_normalized(sample)
    raise ValueError(f"Unknown dataset: {dataset}")


def is_snake_case(name: str) -> bool:
    return bool(name) and bool(SNAKE_CASE_RE.match(name))


def extract_function_names_glaive(chat: str) -> list[str]:
    pattern = re.compile(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"')
    return pattern.findall(chat)


def extract_function_names_xlam(answers_str: str) -> list[str]:
    if not answers_str:
        return []
    try:
        answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str
    except json.JSONDecodeError:
        return []
    names: list[str] = []
    for item in answers if isinstance(answers, list) else []:
        if isinstance(item, dict) and "name" in item:
            names.append(str(item["name"]))
    return names


def extract_function_names(sample: dict[str, Any], dataset: str) -> list[str]:
    if dataset == "glaive":
        return extract_function_names_glaive(sample.get("chat", ""))
    if dataset == "xlam":
        return extract_function_names_xlam(sample.get("answers", ""))
    return []


def extract_arg_keys_glaive(chat: str) -> list[str]:
    pattern = re.compile(r'<functioncall>\s*\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\'(\{.*?\})\'', re.DOTALL)
    keys: list[str] = []
    for match in pattern.findall(chat):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                keys.extend(obj.keys())
        except json.JSONDecodeError:
            continue
    return keys


def extract_arg_keys_xlam(answers_str: str) -> list[str]:
    if not answers_str:
        return []
    try:
        answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str
    except json.JSONDecodeError:
        return []
    keys: list[str] = []
    for item in answers if isinstance(answers, list) else []:
        args = item.get("arguments", {}) if isinstance(item, dict) else {}
        if isinstance(args, dict):
            keys.extend(args.keys())
    return keys


def extract_arg_keys(sample: dict[str, Any], dataset: str) -> list[str]:
    if dataset == "glaive":
        return extract_arg_keys_glaive(sample.get("chat", ""))
    if dataset == "xlam":
        return extract_arg_keys_xlam(sample.get("answers", ""))
    return []


def _normalized_function_calls(sample: dict[str, Any]) -> list[dict[str, Any]]:
    calls = sample.get("function_calls", [])
    return calls if isinstance(calls, list) else []


def _normalized_tools(sample: dict[str, Any]) -> list[dict[str, Any]]:
    tools = sample.get("tools", [])
    return tools if isinstance(tools, list) else []


def _argument_key_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _argument_key_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_argument_key_shape(child) for child in value]
    return None


def _schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _schema_shape(child)
            for key, child in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_schema_shape(child) for child in value]
    return value


def _normalized_structure(sample: dict[str, Any]) -> dict[str, Any]:
    calls = []
    for call in _normalized_function_calls(sample):
        if not isinstance(call, dict):
            calls.append(None)
            continue
        calls.append({
            "name": call.get("name"),
            "arguments": _argument_key_shape(call.get("arguments", {})),
        })

    tools = []
    for tool in _normalized_tools(sample):
        if not isinstance(tool, dict):
            tools.append(None)
            continue
        tools.append({
            "name": tool.get("name"),
            "parameters": _schema_shape(tool.get("parameters", {})),
        })

    return {
        "id": sample.get("id"),
        "source": sample.get("source"),
        "has_tool_call": sample.get("has_tool_call"),
        "function_calls": calls,
        "tools": tools,
    }


def extract_top_level_keys(sample: dict[str, Any], dataset: str) -> list[str]:
    if dataset == "glaive":
        return [k for k in ("system", "chat") if k in sample]
    if dataset == "xlam":
        return [k for k in ("id", "query", "answers", "tools") if k in sample]
    if dataset in ("glaive_normalized", "xlam_normalized"):
        return list(sample.keys())
    return list(sample.keys())


def check_identifier_integrity(
    original: dict[str, Any],
    translated: dict[str, Any],
    dataset: str,
) -> tuple[bool, str]:
    if not isinstance(translated, dict):
        return False, "translated is not a dict"

    if dataset in ("glaive_normalized", "xlam_normalized"):
        if set(original) != set(translated):
            return False, f"top-level keys changed: missing={set(original) - set(translated)}, added={set(translated) - set(original)}"
        if _normalized_structure(original) != _normalized_structure(translated):
            return False, "normalized schema, function names, or argument keys changed"
        for tool in _normalized_tools(translated):
            if not isinstance(tool, dict) or not is_snake_case(str(tool.get("name", ""))):
                return False, f"tool name not snake_case: {tool.get('name') if isinstance(tool, dict) else tool!r}"
        for call in _normalized_function_calls(translated):
            if not isinstance(call, dict) or not is_snake_case(str(call.get("name", ""))):
                return False, f"function name not snake_case: {call.get('name') if isinstance(call, dict) else call!r}"
        return True, "ok"

    orig_top = set(extract_top_level_keys(original, dataset))
    new_top = set(extract_top_level_keys(translated, dataset))
    if orig_top != new_top:
        return False, f"top-level keys changed: missing={orig_top - new_top}, added={new_top - orig_top}"

    for required in orig_top:
        if required not in translated:
            return False, f"missing required top-level key: {required}"

    orig_fns = set(extract_function_names(original, dataset))
    new_fns = set(extract_function_names(translated, dataset))
    if orig_fns != new_fns:
        return False, f"function names changed: missing={orig_fns - new_fns}, added={new_fns - orig_fns}"
    for fn in new_fns:
        if not is_snake_case(fn):
            return False, f"function name not snake_case: {fn!r}"

    orig_args = set(extract_arg_keys(original, dataset))
    new_args = set(extract_arg_keys(translated, dataset))
    if orig_args != new_args:
        return False, f"argument keys changed: missing={orig_args - new_args}, added={new_args - orig_args}"
    for ak in new_args:
        if not is_snake_case(ak):
            return False, f"argument key not snake_case: {ak!r}"

    return True, "ok"


def check_required_fields_present(
    translated: dict[str, Any],
    dataset: str,
) -> tuple[bool, str]:
    if dataset == "glaive":
        if not translated.get("system"):
            return False, "empty system"
        if not translated.get("chat"):
            return False, "empty chat"
    elif dataset == "xlam":
        if not translated.get("query"):
            return False, "empty query"
        if not translated.get("answers"):
            return False, "empty answers"
        if not translated.get("tools"):
            return False, "empty tools"
    elif dataset in ("glaive_normalized", "xlam_normalized"):
        for field in ("id", "source", "query", "function_calls", "tools", "has_tool_call"):
            if field not in translated:
                return False, f"missing {field}"
        if not translated.get("query"):
            return False, "empty query"
        if not isinstance(translated.get("function_calls"), list) or not translated["function_calls"]:
            return False, "empty function_calls"
        if not isinstance(translated.get("tools"), list) or not translated["tools"]:
            return False, "empty tools"
    return True, "ok"
