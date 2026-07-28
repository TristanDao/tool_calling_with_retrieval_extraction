"""Feature group classification via LLM (cache theo tool name).

Classify 1 tool → 1 trong các category VI:
  Tìm kiếm & Kết nối, Đặt vé & Du lịch, Mua sắm & Thương mại, ...
  (xem configs/data/feature_group.yaml)

Cache lưu tại `data/benchmark_vi/.cache/feature_group.json` để tránh gọi LLM lặp.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.data.translate_guidelines import is_snake_case


DEFAULT_CATEGORIES: list[str] = [
    "Tìm kiếm & Kết nối",
    "Đặt vé & Du lịch",
    "Mua sắm & Thương mại",
    "Tài chính & Ngân hàng",
    "Giáo dục & Học tập",
    "Y tế & Sức khỏe",
    "Giải trí & Truyền thông",
    "Nhà hàng & Ẩm thực",
    "Bản đồ & Địa điểm",
    "Tin tức & Thời sự",
    "Thời tiết",
    "Thể thao",
    "Công nghệ & Lập trình",
    "Mạng xã hội",
    "Năng suất & Công việc",
    "Khác",
]


def _resolve_env(obj: Any) -> Any:
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            name = obj[2:-1]
            v = os.getenv(name)
            if v is None and "__" not in name and "_" in name:
                prefix, rest = name.split("_", 1)
                v = os.getenv(f"{prefix}__{rest}", "")
            return v or ""
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(v) for v in obj]
    return obj


def load_config(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        return _resolve_env(yaml.safe_load(f))


def load_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_cache(cache: dict[str, str], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache_path)


def build_classify_prompt(tool: dict[str, Any], categories: list[str]) -> str:
    name = tool.get("name", "")
    desc = tool.get("description", "")
    params = tool.get("parameters", {})
    param_names = list(params.get("properties", {}).keys()) if isinstance(params, dict) else []

    cat_str = "\n".join(f"- {c}" for c in categories)
    return (
        f"Phân loại tool sau vào ĐÚNG 1 trong các nhóm chức năng bên dưới.\n\n"
        f"Tool name: {name}\n"
        f"Description: {desc}\n"
        f"Parameters: {', '.join(param_names[:10])}\n\n"
        f"Các nhóm:\n{cat_str}\n\n"
        f"Chỉ trả về TÊN NHÓM, không giải thích."
    )


async def classify_one(
    client: AsyncOpenAI,
    tool: dict[str, Any],
    categories: list[str],
    semaphore: asyncio.Semaphore,
    cfg: dict[str, Any],
) -> str:
    async with semaphore:
        prompt = build_classify_prompt(tool, categories)
        try:
            response = await client.chat.completions.create(
                model=cfg["api"]["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=float(cfg["api"].get("temperature", 0.0)),
                max_tokens=int(cfg["api"].get("max_tokens", 64)),
            )
            content = (response.choices[0].message.content or "").strip()
            for cat in categories:
                if cat in content:
                    return cat
            return content or "Khác"
        except Exception:
            return "Khác"


async def classify_tools(
    tools: list[dict[str, Any]],
    cfg: dict[str, Any],
    cache: dict[str, str],
) -> dict[str, str]:
    """Classify all tools, sử dụng cache để skip đã phân loại.

    Returns: dict {tool_name: feature_group}
    """
    api_key = cfg["api"]["api_key"]
    if not api_key:
        raise ValueError("api_key missing in config (ALIBABA_API_KEY?)")
    base_url = cfg["api"]["base_url"]
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    categories = cfg.get("categories", DEFAULT_CATEGORIES)
    concurrency = int(cfg["api"].get("concurrency", 10))

    todo: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name", "")
        if not name:
            continue
        if name in cache:
            continue
        todo.append(tool)

    print(f"[feature_group] total tools: {len(tools)}, cached: {len(cache)}, to classify: {len(todo)}", flush=True)

    semaphore = asyncio.Semaphore(concurrency)

    cache_file = Path(cfg["cache_path"])
    n_done = 0
    chunk_size = max(1, int(concurrency))
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        tasks = [classify_one(client, t, categories, semaphore, cfg) for t in chunk]
        results = await asyncio.gather(*tasks)
        for tool, group in zip(chunk, results):
            cache[tool.get("name", "")] = group
        n_done += len(chunk)
        if n_done % max(chunk_size, 10) == 0 or n_done == len(todo):
            print(f"[feature_group] classified {n_done}/{len(todo)}", flush=True)
        save_cache(cache, cache_file)

    return cache


def run_classify(
    tools: list[dict[str, Any]],
    config_path: Path,
) -> dict[str, str]:
    cfg = load_config(config_path)
    cache_path = Path(cfg["cache_path"])
    cache = load_cache(cache_path)
    updated = asyncio.run(classify_tools(tools, cfg, cache))
    save_cache(updated, cache_path)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify feature_group for tools")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True, help="JSON list of tools (name+description+parameters)")
    args = parser.parse_args()

    with args.tools.open("r", encoding="utf-8") as f:
        tools = json.load(f)
    if not isinstance(tools, list):
        print("ERROR: --tools must be a JSON list", file=sys.stderr)
        sys.exit(1)

    cache = run_classify(tools, args.config)
    counter = Counter(cache.values())
    print(f"\nDone. Cache size: {len(cache)}")
    print("Top categories:")
    for cat, n in counter.most_common(10):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
