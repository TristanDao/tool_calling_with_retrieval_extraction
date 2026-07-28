"""Build Bộ 2 (benchmark_vi) từ Bộ 1 (translations).

Input:
  data/translations/glaive_vi.jsonl  (raw VI, multi-turn chat text)
  data/translations/xlam_vi.jsonl    (raw VI, structured JSON)

Output:
  data/benchmark_vi/
    ├── tool_pool.json       (gộp unique tools)
    ├── tool_schema/         (mỗi tool 1 file)
    ├── train.jsonl / val.jsonl / test.jsonl  (split 80/10/10, seed=42)
    └── metadata.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.feature_group_classify import load_cache, load_config, run_classify, save_cache
from src.data.normalize_schema import (
    is_standard_type,
    normalize_tool,
)
from src.data.translate_guidelines import is_snake_case


_GLAIVE_USER_RE = re.compile(
    r"(?:^|\n)\s*(?:USER|NGƯỜI DÙNG)\s*:\s*",
    re.IGNORECASE,
)
_GLAIVE_ASSISTANT_RE = re.compile(
    r"(?:^|\n)\s*(?:ASSISTANT|TRỢ LÝ|A)\s*:\s*",
    re.IGNORECASE,
)
_GLAIVE_FUNCTION_RESP_RE = re.compile(
    r"(?:^|\n)\s*(?:FUNCTION RESPONSE|PHẢN HỒI HÀM)\s*:\s*",
    re.IGNORECASE,
)
_FUNCTIONCALL_RE = re.compile(
    r"<functioncall>\s*(.+?)\s*(?:<\|endoftext\|>|$)",
    re.DOTALL,
)
_ENDOFTOKEN_RE = re.compile(r"<\|endoftext\|>")
_SYSTEM_PREFIX_RE = re.compile(
    r"^\s*(?:SYSTEM|HỆ THỐNG)\s*:\s*",
    re.IGNORECASE,
)
_SYSTEM_JSON_RE = re.compile(r"\{[\s\S]*\}")


@dataclass(kw_only=True)
class BuildConfig:
    input_glaive: Path
    input_xlam: Path
    source_glaive: Path | None = None
    source_xlam: Path | None = None
    output_dir: Path
    tool_schema_dir: Path
    tool_pool_path: Path
    train_path: Path
    val_path: Path
    test_path: Path
    metadata_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    shuffle: bool
    feature_group_config: Path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BuildConfig":
        split = d.get("split", {})
        feat = d.get("feature_group", {})
        return cls(
            input_glaive=Path(d["input_glaive"]),
            input_xlam=Path(d["input_xlam"]),
            source_glaive=Path(d.get("source_glaive", "")),
            source_xlam=Path(d.get("source_xlam", "")),
            output_dir=Path(d["output_dir"]),
            tool_schema_dir=Path(d["tool_schema_dir"]),
            tool_pool_path=Path(d["tool_pool_path"]),
            train_path=Path(d["train_path"]),
            val_path=Path(d["val_path"]),
            test_path=Path(d["test_path"]),
            metadata_path=Path(d["metadata_path"]),
            train_ratio=float(split.get("train_ratio", 0.8)),
            val_ratio=float(split.get("val_ratio", 0.1)),
            test_ratio=float(split.get("test_ratio", 0.1)),
            seed=int(split.get("seed", 42)),
            shuffle=bool(split.get("shuffle", True)),
            feature_group_config=Path(d.get("feature_group_config_path", "configs/data/feature_group.yaml")),
        )


def _safe_json_loads(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _strip_outer_braces(text: str) -> str:
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        )
    return text.strip()


def _parse_functioncall_block(block: str) -> dict[str, Any] | None:
    """Parse 1 <functioncall> block.

    Glaive format (note: arguments is single-quoted JSON string):
        {"name": "foo", "arguments": '{"k": "v"}'}

    Strategies (theo thứ tự):
      1. json.loads trực tiếp (work nếu arguments là double-quoted)
      2. Manual parse: extract name + arguments riêng
    """
    block = block.strip()
    if not block:
        return None

    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', block)
    if not name_m:
        return None
    name = name_m.group(1)

    args_m = re.search(r'"arguments"\s*:\s*\'(.*?)\'(?=\s*[,}\s])', block, re.DOTALL)
    if not args_m:
        args_m = re.search(r'"arguments"\s*:\s*"((?:[^"\\]|\\.)*)"', block, re.DOTALL)
        if args_m:
            args_str = args_m.group(1)
        else:
            args_str = "{}"
    else:
        args_str = args_m.group(1)

    if args_str.startswith("{") or args_str.startswith("["):
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {}
    else:
        arguments = {}

    return {"name": name, "arguments": arguments}


def _extract_function_calls_from_chat(chat: str) -> list[dict[str, Any]]:
    """Extract all <functioncall> blocks from Glaive chat text."""
    calls: list[dict[str, Any]] = []
    for match in _FUNCTIONCALL_RE.finditer(chat):
        block = match.group(1).strip()
        obj = _parse_functioncall_block(block)
        if obj is None:
            continue
        name = obj.get("name", "")
        if not is_snake_case(name):
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append({"name": name, "arguments": args})
    return calls


def _split_glaive_chat_turns(chat: str) -> list[dict[str, Any]]:
    """Parse Glaive chat text → list of turns.

    Strategy:
      - Trên ORIGINAL chat (chưa strip FC): tìm các marker (USER, ASSISTANT, FN_RESP)
        + function call blocks theo vị trí
      - Mỗi segment gắn với marker được classify dựa vào việc có FC block trong segment không
      - Function call: tạo turn assistant với function_calls, content=None
      - Text only: tạo turn assistant với content
      - Function response: dùng tên tool từ FC gần nhất
    """
    text = chat.strip()
    text = _ENDOFTOKEN_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    function_calls_in_text = _extract_function_calls_from_chat(chat)

    markers: list[tuple[int, int, str, str]] = []
    for m in _GLAIVE_USER_RE.finditer(text):
        markers.append((m.start(), m.end(), "user", m.group(0)))
    for m in _GLAIVE_FUNCTION_RESP_RE.finditer(text):
        markers.append((m.start(), m.end(), "function", m.group(0)))
    for m in _GLAIVE_ASSISTANT_RE.finditer(text):
        markers.append((m.start(), m.end(), "assistant", m.group(0)))
    markers.sort(key=lambda x: x[0])

    if not markers:
        return []

    raw_segments: list[tuple[str, str, int, int]] = []
    for i, (start, end, role, _marker) in enumerate(markers):
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        content = text[end:next_start]
        raw_segments.append((role, content, start, next_start))

    turns: list[dict[str, Any]] = []
    last_function_call_name: str | None = None
    pending_text = ""

    def has_functioncall_block(start: int, end: int) -> bool:
        for m in _FUNCTIONCALL_RE.finditer(text, start, end):
            return True
        return False

    for role, content, seg_start, seg_end in raw_segments:
        text_in_seg = text[seg_start:seg_end]
        if role == "user":
            user_content = (pending_text + "\n" + content).strip() if pending_text else content.strip()
            pending_text = ""
            turns.append({"role": "user", "content": user_content})
        elif role == "assistant":
            if has_functioncall_block(seg_start, seg_end):
                fc = None
                for m in _FUNCTIONCALL_RE.finditer(text, seg_start, seg_end):
                    fc = _parse_functioncall_block(m.group(1).strip())
                    if fc is not None and is_snake_case(fc.get("name", "")):
                        break
                if fc is not None:
                    last_function_call_name = fc["name"]
                    turns.append({
                        "role": "assistant",
                        "content": None,
                        "function_calls": [fc],
                    })
                else:
                    text_only = _FUNCTIONCALL_RE.sub("", text_in_seg).strip()
                    if text_only:
                        turns.append({"role": "assistant", "content": text_only})
            else:
                text_only = content.strip()
                if text_only:
                    if pending_text:
                        text_only = (pending_text + "\n" + text_only).strip()
                    pending_text = ""
                    turns.append({"role": "assistant", "content": text_only})
        elif role == "function":
            func_name = last_function_call_name or ""
            func_content = content.strip()
            if func_name and func_content.startswith(func_name):
                func_content = func_content[len(func_name):].lstrip(":\n ")
            turns.append({
                "role": "function",
                "name": func_name,
                "content": func_content,
            })
            pending_text = ""
            last_function_call_name = None

    return turns


def _extract_glaive_tool_from_system(system_text: str) -> dict[str, Any] | None:
    cleaned = _SYSTEM_PREFIX_RE.sub("", system_text or "")
    match = _SYSTEM_JSON_RE.search(cleaned)
    if not match:
        return None
    candidate = match.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            from src.data.translate import _init_fallback_exc_types
        except ImportError:
            pass
        depth = 0
        in_str = False
        escape = False
        end_idx = -1
        for i, c in enumerate(cleaned[match.start():]):
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx > 0:
            try:
                obj = json.loads(cleaned[match.start():match.start() + end_idx + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name", "")
    if not is_snake_case(name):
        return None
    return obj


def parse_glaive_sample(raw: dict[str, Any], idx: int) -> dict[str, Any] | None:
    system_text = raw.get("system", "")
    chat_text = raw.get("chat", "")
    if not chat_text.strip():
        return None

    tool_schema_raw = _extract_glaive_tool_from_system(system_text)

    turns = _split_glaive_chat_turns(chat_text)
    if not turns:
        return None
    if not any(t["role"] == "user" for t in turns):
        return None

    tools: list[dict[str, Any]] = []
    if tool_schema_raw:
        normalized = normalize_tool(tool_schema_raw, dataset="glaive")
        tools.append(normalized)
    called_names = set()
    for t in turns:
        if t["role"] == "assistant":
            for fc in t.get("function_calls", []):
                called_names.add(fc["name"])
    for t in turns:
        if t["role"] == "function" and t.get("name"):
            called_names.add(t["name"])
    for cn in called_names:
        if cn and not any(t.get("name") == cn for t in tools):
            tools.append({
                "name": cn,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            })

    return {
        "id": f"glaive_{idx:05d}",
        "source": "glaive",
        "conversation": turns,
        "tools": tools,
    }


def parse_xlam_sample(raw: dict[str, Any], idx: int) -> dict[str, Any] | None:
    answers = _safe_json_loads(raw.get("answers", ""))
    tools_raw = _safe_json_loads(raw.get("tools", ""))
    query = raw.get("query", "")
    if not query or not isinstance(answers, list) or not isinstance(tools_raw, list):
        return None

    function_calls: list[dict[str, Any]] = []
    for ans in answers:
        if not isinstance(ans, dict):
            continue
        name = ans.get("name", "")
        if not is_snake_case(name):
            continue
        args = ans.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        function_calls.append({"name": name, "arguments": args})
    if not function_calls:
        return None

    tools: list[dict[str, Any]] = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue
        normalized = normalize_tool(t, dataset="xlam")
        tools.append(normalized)

    conversation = [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": None,
            "function_calls": function_calls,
        },
    ]

    return {
        "id": f"xlam_{idx:05d}",
        "source": "xlam",
        "conversation": conversation,
        "tools": tools,
    }


def _attach_feature_group(tools: list[dict[str, Any]], cache: dict[str, str]) -> list[dict[str, Any]]:
    for tool in tools:
        name = tool.get("name", "")
        if name in cache:
            tool["feature_group"] = cache[name]
        else:
            tool["feature_group"] = "Khác"
    return tools


def _build_tool_pool(all_samples: list[dict[str, Any]], cache: dict[str, str]) -> list[dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for sample in all_samples:
        for tool in sample.get("tools", []):
            name = tool.get("name", "")
            if not name:
                continue
            if name not in pool:
                pool[name] = {
                    "name": name,
                    "description": tool.get("description", ""),
                    "feature_group": cache.get(name, "Khác"),
                    "parameters": tool.get("parameters", {}),
                }
            else:
                if not pool[name].get("description") and tool.get("description"):
                    pool[name]["description"] = tool["description"]
                if not pool[name].get("parameters", {}).get("properties"):
                    if tool.get("parameters", {}).get("properties"):
                        pool[name]["parameters"] = tool["parameters"]
    return list(pool.values())


def _write_tool_schema_files(pool: list[dict[str, Any]], schema_dir: Path) -> None:
    schema_dir.mkdir(parents=True, exist_ok=True)
    for tool in pool:
        name = tool.get("name", "unknown")
        if not name or not is_snake_case(name):
            continue
        path = schema_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(tool, f, ensure_ascii=False, indent=2)


def _split_dataset(
    samples: list[dict[str, Any]],
    cfg: BuildConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if cfg.shuffle:
        rng = random.Random(cfg.seed)
        samples = samples.copy()
        rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


def _validate_sample(sample: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if "id" not in sample or not sample["id"]:
        errors.append("missing id")
    if "conversation" not in sample or not isinstance(sample["conversation"], list):
        errors.append("missing or invalid conversation")
    if "tools" not in sample or not isinstance(sample["tools"], list):
        errors.append("missing or invalid tools")
    if errors:
        return False, errors

    for t in sample["conversation"]:
        if t.get("role") not in {"user", "assistant", "function"}:
            errors.append(f"invalid role: {t.get('role')}")
        if t.get("role") == "assistant" and t.get("content") is None and not t.get("function_calls"):
            errors.append("assistant turn with no content and no function_calls")
        if t.get("role") == "function" and (not t.get("name") or t.get("content") is None):
            errors.append("function turn missing name or content")

    tool_names = {t.get("name") for t in sample["tools"]}
    for t in sample["conversation"]:
        if t.get("role") == "assistant":
            for fc in t.get("function_calls", []):
                if fc.get("name") not in tool_names:
                    errors.append(f"function_call name not in tools: {fc.get('name')}")
        if t.get("role") == "function" and t.get("name") not in tool_names:
            errors.append(f"function name not in tools: {t.get('name')}")

    for t in sample["tools"]:
        if not is_snake_case(t.get("name", "")):
            errors.append(f"tool name not snake_case: {t.get('name')}")
        if not t.get("feature_group"):
            errors.append(f"tool missing feature_group: {t.get('name')}")
        params = t.get("parameters", {})
        if not isinstance(params, dict) or params.get("type") != "object":
            errors.append(f"tool parameters not object: {t.get('name')}")

    return len(errors) == 0, errors


def _load_failed_indices(failed_path: Path) -> set[int]:
    indices: set[int] = set()
    if not failed_path.exists():
        return indices
    with failed_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                idx = d.get("source_index")
                if isinstance(idx, int):
                    indices.add(idx)
            except json.JSONDecodeError:
                continue
    return indices


def _load_source_samples(path: Path) -> list[tuple[int, dict[str, Any]]]:
    samples: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append((idx, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return samples


def _load_translated_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _pair_source_translated(
    source_path: Path | None,
    translated_path: Path,
    failed: set[int],
) -> list[tuple[int, dict[str, Any]]]:
    if source_path is None:
        out: list[tuple[int, dict[str, Any]]] = []
        with translated_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip() or i in failed:
                    continue
                try:
                    out.append((i, json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return out

    source_samples = _load_source_samples(source_path)
    translated = _load_translated_records(translated_path)

    out: list[tuple[int, dict[str, Any]]] = []
    t_i = 0
    for idx, _raw in source_samples:
        if idx in failed:
            continue
        if t_i >= len(translated):
            break
        rec = translated[t_i]
        source_index = rec.get("source_index")
        sample = rec.get("sample") if isinstance(rec.get("sample"), dict) else rec
        if isinstance(source_index, int) and source_index != idx:
            sample = rec
        out.append((idx, sample if isinstance(sample, dict) else {}))
        t_i += 1
    return out


def build_benchmark(cfg: BuildConfig) -> dict[str, Any]:
    print(f"[build] output dir: {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.tool_schema_dir.mkdir(parents=True, exist_ok=True)

    glaive_failed = _load_failed_indices(cfg.input_glaive.parent / "failed" / "glaive_failed.jsonl")
    xlam_failed = _load_failed_indices(cfg.input_xlam.parent / "failed" / "xlam_failed.jsonl")
    print(f"[build] glaive failed: {len(glaive_failed)}, xlam failed: {len(xlam_failed)}")

    glaive_raw = _pair_source_translated(cfg.source_glaive, cfg.input_glaive, glaive_failed)
    xlam_raw = _pair_source_translated(cfg.source_xlam, cfg.input_xlam, xlam_failed)
    print(f"[build] loaded glaive: {len(glaive_raw)}, xlam: {len(xlam_raw)}")

    samples: list[dict[str, Any]] = []
    parse_errors = Counter()
    for orig_idx, raw in glaive_raw:
        sample = parse_glaive_sample(raw, orig_idx)
        if sample is None:
            parse_errors["glaive_parse_fail"] += 1
            continue
        samples.append(sample)
    for orig_idx, raw in xlam_raw:
        sample = parse_xlam_sample(raw, orig_idx)
        if sample is None:
            parse_errors["xlam_parse_fail"] += 1
            continue
        samples.append(sample)

    print(f"[build] parsed samples: {len(samples)} (parse errors: {dict(parse_errors)})")

    seen_ids: set[str] = set()
    unique_samples: list[dict[str, Any]] = []
    dup_count = 0
    for s in samples:
        if s["id"] in seen_ids:
            dup_count += 1
            continue
        seen_ids.add(s["id"])
        unique_samples.append(s)
    print(f"[build] unique samples: {len(unique_samples)} (dup removed: {dup_count})")

    fg_cfg = load_config(cfg.feature_group_config)
    cache_path = Path(fg_cfg["cache_path"])
    cache = load_cache(cache_path)

    all_tool_names = set()
    for s in unique_samples:
        for t in s.get("tools", []):
            if t.get("name"):
                all_tool_names.add(t["name"])
    tools_for_classify = []
    for s in unique_samples:
        for t in s.get("tools", []):
            if t.get("name") in all_tool_names and t.get("name") not in cache:
                tools_for_classify.append(t)

    unique_tools_for_classify: dict[str, dict[str, Any]] = {}
    for t in tools_for_classify:
        if t["name"] not in unique_tools_for_classify:
            unique_tools_for_classify[t["name"]] = t
    tools_for_classify = list(unique_tools_for_classify.values())

    if tools_for_classify:
        print(f"[build] classifying {len(tools_for_classify)} new tools")
        cache = asyncio_run_classify(tools_for_classify, fg_cfg, cache)
        save_cache(cache, cache_path)

    for s in unique_samples:
        s["tools"] = _attach_feature_group(s.get("tools", []), cache)

    valid_samples: list[dict[str, Any]] = []
    invalid_count = 0
    for s in unique_samples:
        ok, errs = _validate_sample(s)
        if ok:
            valid_samples.append(s)
        else:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"[build] invalid sample {s.get('id')}: {errs[:3]}")
    print(f"[build] valid samples: {len(valid_samples)} (invalid: {invalid_count})")

    pool = _build_tool_pool(valid_samples, cache)
    print(f"[build] tool pool: {len(pool)} unique tools")

    with cfg.tool_pool_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    _write_tool_schema_files(pool, cfg.tool_schema_dir)

    train, val, test = _split_dataset(valid_samples, cfg)
    print(f"[build] split: train={len(train)} val={len(val)} test={len(test)}")

    for path, data in [
        (cfg.train_path, train),
        (cfg.val_path, val),
        (cfg.test_path, test),
    ]:
        with path.open("w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    source_counter = Counter(s["source"] for s in valid_samples)
    fg_counter = Counter(t.get("feature_group", "Khác") for t in pool)
    n_turns = [len(s["conversation"]) for s in valid_samples]
    n_calls = [
        sum(len(t.get("function_calls", [])) for t in s["conversation"] if t["role"] == "assistant")
        for s in valid_samples
    ]
    metadata = {
        "total_samples": len(valid_samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "split": {
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "seed": cfg.seed,
        },
        "source_distribution": dict(source_counter),
        "feature_group_distribution": dict(fg_counter.most_common()),
        "n_unique_tools": len(pool),
        "turns_per_sample": {
            "min": min(n_turns) if n_turns else 0,
            "max": max(n_turns) if n_turns else 0,
            "mean": round(sum(n_turns) / max(len(n_turns), 1), 2),
        },
        "calls_per_sample": {
            "min": min(n_calls) if n_calls else 0,
            "max": max(n_calls) if n_calls else 0,
            "mean": round(sum(n_calls) / max(len(n_calls), 1), 2),
        },
        "parse_errors": dict(parse_errors),
        "duplicates_removed": dup_count,
        "invalid_samples": invalid_count,
    }
    with cfg.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[build] DONE. metadata → {cfg.metadata_path}")
    return metadata


def asyncio_run_classify(tools, cfg, cache):
    import asyncio as _asyncio
    return _asyncio.run(
        __import__("src.data.feature_group_classify", fromlist=["classify_tools"]).classify_tools(
            tools, cfg, cache,
        )
    )


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Build Bộ 2 (benchmark_vi) from Bộ 1 (translations)")
    parser.add_argument("--config", type=Path, default=Path("configs/data/benchmark.yaml"))
    parser.add_argument("--input-glaive", type=Path, default=None)
    parser.add_argument("--input-xlam", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-classify", action="store_true", help="Skip LLM classification, use 'Khác' fallback")
    args = parser.parse_args()

    import yaml
    with args.config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if args.input_glaive:
        raw["input_glaive"] = str(args.input_glaive)
    if args.input_xlam:
        raw["input_xlam"] = str(args.input_xlam)
    if args.output_dir:
        raw["output_dir"] = str(args.output_dir)

    cfg = BuildConfig.from_dict(raw)

    if args.no_classify:
        original = build_benchmark

        def patched(cfg):
            print("[build] --no-classify: skipping LLM, all tools get 'Khác'")
            return _build_benchmark_no_classify(cfg)

        patched(cfg)
    else:
        build_benchmark(cfg)


def _build_benchmark_no_classify(cfg: BuildConfig) -> dict[str, Any]:
    """Build benchmark without LLM classification (all 'Khác')."""
    print(f"[build] output dir: {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.tool_schema_dir.mkdir(parents=True, exist_ok=True)

    glaive_failed = _load_failed_indices(cfg.input_glaive.parent / "failed" / "glaive_failed.jsonl")
    xlam_failed = _load_failed_indices(cfg.input_xlam.parent / "failed" / "xlam_failed.jsonl")
    glaive_raw = _pair_source_translated(cfg.source_glaive, cfg.input_glaive, glaive_failed)
    xlam_raw = _pair_source_translated(cfg.source_xlam, cfg.input_xlam, xlam_failed)
    print(f"[build] loaded glaive: {len(glaive_raw)}, xlam: {len(xlam_raw)}")

    samples: list[dict[str, Any]] = []
    for orig_idx, raw in glaive_raw:
        sample = parse_glaive_sample(raw, orig_idx)
        if sample is not None:
            samples.append(sample)
    for orig_idx, raw in xlam_raw:
        sample = parse_xlam_sample(raw, orig_idx)
        if sample is not None:
            samples.append(sample)
    print(f"[build] parsed samples: {len(samples)}")

    seen_ids: set[str] = set()
    unique_samples: list[dict[str, Any]] = []
    for s in samples:
        if s["id"] in seen_ids:
            continue
        seen_ids.add(s["id"])
        unique_samples.append(s)
    print(f"[build] unique samples: {len(unique_samples)}")

    for s in unique_samples:
        for t in s.get("tools", []):
            t["feature_group"] = "Khác"

    valid_samples: list[dict[str, Any]] = []
    invalid_count = 0
    for s in unique_samples:
        ok, _ = _validate_sample(s)
        if ok:
            valid_samples.append(s)
        else:
            invalid_count += 1
    print(f"[build] valid samples: {len(valid_samples)} (invalid: {invalid_count})")

    pool = _build_tool_pool(valid_samples, {"Khác": "Khác"})
    print(f"[build] tool pool: {len(pool)} unique tools")

    with cfg.tool_pool_path.open("w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    _write_tool_schema_files(pool, cfg.tool_schema_dir)

    train, val, test = _split_dataset(valid_samples, cfg)
    print(f"[build] split: train={len(train)} val={len(val)} test={len(test)}")
    for path, data in [
        (cfg.train_path, train),
        (cfg.val_path, val),
        (cfg.test_path, test),
    ]:
        with path.open("w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    metadata = {
        "total_samples": len(valid_samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "split": {
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "test_ratio": cfg.test_ratio,
            "seed": cfg.seed,
        },
        "n_unique_tools": len(pool),
        "invalid_samples": invalid_count,
    }
    with cfg.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[build] DONE. metadata → {cfg.metadata_path}")
    return metadata


if __name__ == "__main__":
    main()
