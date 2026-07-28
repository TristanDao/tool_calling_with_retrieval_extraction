"""Async batch translation pipeline (EN -> VI) cho tool-calling datasets.

Thiết kế:
- Đọc raw JSONL theo stream (generator), không load full dataset vào RAM.
- Gửi K=50 samples/batch qua Alibaba OpenAI-compatible API.
- 20 concurrent requests (asyncio.Semaphore).
- 3 retry/sample với exponential backoff.
- Validate per-sample ngay khi response về.
- Append JSONL output, flush + fsync mỗi sample.
- Atomic checkpoint JSON để resume.
- Failed samples ghi `failed.jsonl` riêng.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.data.feature_group_classify import classify_tools, load_cache, load_config, save_cache
from src.data.translate_guidelines import (
    TRANSLATE_SYSTEM_PROMPT,
    build_translate_prompt,
    check_identifier_integrity,
    check_required_fields_present,
)
from src.data.translation_checkpoint import Checkpoint, load as load_checkpoint, save_atomic


@dataclass(kw_only=True)
class TranslationConfig:
    input_path: Path
    output_path: Path
    failed_path: Path
    checkpoint_path: Path
    log_path: Path
    dataset: str
    start_index: int
    end_index: int | None
    batch_size: int
    concurrency: int
    max_retries: int
    retry_initial_delay: float
    retry_backoff: float
    api_base_url: str
    api_key: str
    api_model: str
    api_backup_model: str | None
    api_temperature: float
    api_timeout: float
    api_max_tokens: int
    progress_log_every_n_batches: int
    feature_group_enabled: bool = True
    feature_group_config_path: Path = Path("configs/data/feature_group.yaml")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranslationConfig":
        end = d.get("end_index")
        return cls(
            input_path=Path(d["input"]),
            output_path=Path(d["output"]),
            failed_path=Path(d["failed_output"]),
            checkpoint_path=Path(d["checkpoint"]),
            log_path=Path(d.get("log_file", str(Path(d["output"]).with_suffix(".log")))),
            dataset=d["dataset"],
            start_index=int(d.get("start_index", 0)),
            end_index=None if end is None else int(end),
            batch_size=int(d.get("batch_size", 50)),
            concurrency=int(d.get("concurrency", 20)),
            max_retries=int(d.get("max_retries", 3)),
            retry_initial_delay=float(d.get("retry_initial_delay", 1.0)),
            retry_backoff=float(d.get("retry_backoff", 2.0)),
            api_base_url=d["api"]["base_url"],
            api_key=d["api"]["api_key"],
            api_model=d["api"].get("model") or "qwen3.7-flash-2026-07-15",
            api_backup_model=d["api"].get("backup_model") or None,
            api_temperature=float(d["api"].get("temperature", 0.1)),
            api_timeout=float(d["api"].get("timeout", 60.0)),
            api_max_tokens=int(d["api"].get("max_tokens", 8192)),
            progress_log_every_n_batches=int(d.get("progress_log_every_n_batches", 5)),
            feature_group_enabled=bool(d.get("feature_group", {}).get("enabled", True)),
            feature_group_config_path=Path(d.get("feature_group", {}).get("config_path", "configs/data/feature_group.yaml")),
        )


def setup_logger(log_path: Path) -> Any:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import logging

    logger = logging.getLogger(f"translate.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def read_jsonl_generator(path: Path, start: int = 0, end: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_response(content: str) -> dict[str, Any]:
    text = content.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"cannot parse JSON from response: {text[:200]}")


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _extract_glaive_tools(sample: dict[str, Any]) -> list[dict[str, Any]]:
    system_text = sample.get("system", "")
    if not isinstance(system_text, str) or not system_text.strip():
        return []
    cleaned = system_text.strip()
    if cleaned.startswith("SYSTEM:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first == -1 or last <= first:
        return []
    try:
        obj = json.loads(cleaned[first:last + 1])
    except json.JSONDecodeError:
        return []
    return [obj] if isinstance(obj, dict) else []


def _extract_xlam_tools(sample: dict[str, Any]) -> list[dict[str, Any]]:
    tools = _safe_json_loads(sample.get("tools", ""))
    if isinstance(tools, list):
        return [t for t in tools if isinstance(t, dict)]
    return []


def _extract_feature_group_tools(sample: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    if dataset == "glaive":
        return _extract_glaive_tools(sample)
    if dataset == "xlam":
        return _extract_xlam_tools(sample)
    return []


_FALLBACK_EXC_TYPES: tuple[type, ...] = ()


def _init_fallback_exc_types() -> tuple[type, ...]:
    global _FALLBACK_EXC_TYPES
    if _FALLBACK_EXC_TYPES:
        return _FALLBACK_EXC_TYPES
    from openai import (
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        PermissionDeniedError,
        RateLimitError,
    )
    _FALLBACK_EXC_TYPES = (
        RateLimitError,
        PermissionDeniedError,
        AuthenticationError,
        APITimeoutError,
        APIStatusError,
    )
    return _FALLBACK_EXC_TYPES


def _is_fallback_worthy(exc: BaseException) -> bool:
    for t in _init_fallback_exc_types():
        if isinstance(exc, t):
            return True
    return False


async def translate_one(
    client: AsyncOpenAI,
    sample: dict[str, Any],
    source_index: int,
    cfg: TranslationConfig,
    semaphore: asyncio.Semaphore,
    log_callback=None,
) -> tuple[int, dict[str, Any] | None, str | None]:
    async with semaphore:
        prompt = build_translate_prompt(sample, cfg.dataset)
        last_error: str | None = None
        last_fallback_exc: BaseException | None = None
        models_tried: set[str] = set()

        models_to_try: list[str] = [cfg.api_model]
        if cfg.api_backup_model and cfg.api_backup_model != cfg.api_model:
            models_to_try.append(cfg.api_backup_model)

        for model_name in models_to_try:
            models_tried.add(model_name)
            for attempt in range(cfg.max_retries):
                try:
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=cfg.api_temperature,
                        max_tokens=cfg.api_max_tokens,
                        timeout=cfg.api_timeout,
                    )
                    content = response.choices[0].message.content or ""
                    translated = parse_response(content)

                    ok, reason = check_identifier_integrity(sample, translated, cfg.dataset)
                    if not ok:
                        raise ValueError(f"identifier integrity: {reason}")
                    ok2, reason2 = check_required_fields_present(translated, cfg.dataset)
                    if not ok2:
                        raise ValueError(f"required fields: {reason2}")

                    return source_index, translated, None

                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    is_fallback = _is_fallback_worthy(e) and model_name == cfg.api_model
                    if is_fallback and cfg.api_backup_model and cfg.api_backup_model not in models_tried:
                        last_fallback_exc = e
                        if log_callback:
                            log_callback(
                                f"[idx={source_index}] main model '{cfg.api_model}' failed with "
                                f"{type(e).__name__}; falling back to '{cfg.api_backup_model}'"
                            )
                        break
                    if attempt < cfg.max_retries - 1:
                        delay = cfg.retry_initial_delay * (cfg.retry_backoff ** attempt)
                        await asyncio.sleep(delay)

        if last_fallback_exc is None and last_error is not None:
            return source_index, None, last_error
        if last_fallback_exc is not None and last_error is not None:
            return source_index, None, f"all models failed; last={last_error}"
        return source_index, None, last_error or "unknown error"


async def process_batch(
    client: AsyncOpenAI,
    batch: list[tuple[int, dict[str, Any]]],
    cfg: TranslationConfig,
    semaphore: asyncio.Semaphore,
    fout_success,
    fout_failed,
    fg_cfg: dict[str, Any] | None = None,
    fg_cache: dict[str, str] | None = None,
    logger=None,
) -> tuple[int, int]:
    def log_cb(msg: str) -> None:
        if logger is not None:
            logger.warning(msg)

    tasks = [
        translate_one(client, sample, idx, cfg, semaphore, log_callback=log_cb)
        for idx, sample in batch
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    n_success = 0
    n_failed = 0
    for source_index, translated, error in results:
        if translated is not None:
            fout_success.write(json.dumps(translated, ensure_ascii=False) + "\n")
            n_success += 1
        else:
            failed_record = {
                "source_index": source_index,
                "error": error,
            }
            fout_failed.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
            n_failed += 1

    fout_success.flush()
    os.fsync(fout_success.fileno())
    fout_failed.flush()
    os.fsync(fout_failed.fileno())

    if fg_cfg is not None and fg_cache is not None:
        tools_to_classify: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for _, original in batch:
            for tool in _extract_feature_group_tools(original, cfg.dataset):
                name = tool.get("name", "")
                if not name or name in seen_names or name in fg_cache:
                    continue
                seen_names.add(name)
                tools_to_classify.append(tool)
        if tools_to_classify:
            await classify_tools(tools_to_classify, fg_cfg, fg_cache)
            save_cache(fg_cache, Path(fg_cfg["cache_path"]))

    return n_success, n_failed


async def run_translation(cfg: TranslationConfig) -> dict[str, Any]:
    logger = setup_logger(cfg.log_path)
    tracemalloc.start()
    start_time = time.time()

    if not cfg.api_key:
        raise ValueError("ALIBABA_API_KEY is empty; check .env")

    client = AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.api_base_url,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(cfg.concurrency)

    fg_cfg: dict[str, Any] | None = None
    fg_cache: dict[str, str] | None = None
    if cfg.feature_group_enabled:
        fg_cfg = load_config(cfg.feature_group_config_path)
        fg_cache = load_cache(Path(fg_cfg["cache_path"]))
        logger.info("feature_group labeling enabled: cache=%s cached=%d", fg_cfg["cache_path"], len(fg_cache))
    else:
        logger.info("feature_group labeling disabled")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.failed_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(cfg.checkpoint_path, dataset=cfg.dataset)
    effective_start = max(cfg.start_index, checkpoint.last_processed_index)

    logger.info(
        "starting translation: dataset=%s input=%s start=%d end=%s batch=%d conc=%d",
        cfg.dataset, cfg.input_path, effective_start, cfg.end_index,
        cfg.batch_size, cfg.concurrency,
    )
    if cfg.api_backup_model and cfg.api_backup_model != cfg.api_model:
        logger.info(
            "backup model enabled: main=%s backup=%s",
            cfg.api_model, cfg.api_backup_model,
        )
    else:
        logger.info("backup model: (none)")
    logger.info(
        "checkpoint: last_processed=%d success=%d failed=%d -> resuming from %d",
        checkpoint.last_processed_index, checkpoint.total_success, checkpoint.total_failed,
        effective_start,
    )

    total_success = 0
    total_failed = 0
    n_batches = 0

    with cfg.output_path.open("a", encoding="utf-8") as fout_success, \
         cfg.failed_path.open("a", encoding="utf-8") as fout_failed:

        batch: list[tuple[int, dict[str, Any]]] = []
        for source_index, sample in read_jsonl_generator(
            cfg.input_path, start=effective_start, end=cfg.end_index
        ):
            batch.append((source_index, sample))
            if len(batch) >= cfg.batch_size:
                n_s, n_f = await process_batch(
                    client, batch, cfg, semaphore, fout_success, fout_failed,
                    fg_cfg=fg_cfg, fg_cache=fg_cache, logger=logger,
                )
                checkpoint.advance(n_s, n_f)
                save_atomic(checkpoint, cfg.checkpoint_path)
                total_success += n_s
                total_failed += n_f
                n_batches += 1
                batch = []
                if n_batches % cfg.progress_log_every_n_batches == 0:
                    elapsed = time.time() - start_time
                    rate = (total_success + total_failed) / max(elapsed, 1e-6)
                    current, peak = tracemalloc.get_traced_memory()
                    logger.info(
                        "progress: idx=%d success=%d failed=%d rate=%.1f/s peak_mem=%.1fMB",
                        checkpoint.last_processed_index, total_success, total_failed,
                        rate, peak / 1024 / 1024,
                    )

        if batch:
            n_s, n_f = await process_batch(
                client, batch, cfg, semaphore, fout_success, fout_failed,
                fg_cfg=fg_cfg, fg_cache=fg_cache, logger=logger,
            )
            checkpoint.advance(n_s, n_f)
            save_atomic(checkpoint, cfg.checkpoint_path)
            total_success += n_s
            total_failed += n_f
            n_batches += 1

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "dataset": cfg.dataset,
        "input": str(cfg.input_path),
        "output": str(cfg.output_path),
        "failed": str(cfg.failed_path),
        "elapsed_sec": round(elapsed, 2),
        "total_success": total_success,
        "total_failed": total_failed,
        "success_rate": round(total_success / max(total_success + total_failed, 1), 4),
        "throughput_samples_per_sec": round((total_success + total_failed) / max(elapsed, 1e-6), 2),
        "peak_memory_mb": round(peak / 1024 / 1024, 2),
        "checkpoint_final": checkpoint.to_dict(),
    }
    logger.info("DONE: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def _resolve_env_placeholders(obj: Any) -> Any:
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            env_name = obj[2:-1]
            value = os.getenv(env_name)
            if value is None and "__" not in env_name:
                if "_" in env_name:
                    prefix, rest = env_name.split("_", 1)
                    alt_name = f"{prefix}__{rest}"
                    value = os.getenv(alt_name, "")
            return value or ""
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    return obj


def load_config_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_env_placeholders(raw)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async batch translation EN -> VI")
    p.add_argument("--config", type=Path, required=True, help="Path to translate.yaml")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--failed-output", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config_yaml(args.config)
    overrides = {
        "input": args.input,
        "output": args.output,
        "failed_output": args.failed_output,
        "checkpoint": args.checkpoint,
        "log_file": args.log_file,
        "dataset": args.dataset,
        "start_index": args.start,
        "end_index": args.end,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
    }
    for k, v in overrides.items():
        if v is not None:
            raw[k] = v

    cfg = TranslationConfig.from_dict(raw)
    summary = asyncio.run(run_translation(cfg))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
