"""QA translation: rule-based check (100%) + LLM judge (5% sample).

Usage:
    python -m src.data.qa_translation --config configs/data/qa.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.data.translate_guidelines import (
    check_identifier_integrity,
    check_required_fields_present,
    is_snake_case,
)


@dataclass(kw_only=True)
class QAConfig:
    input_path: Path
    output_path: Path
    report_path: Path
    source_input_path: Path | None = None
    failed_input_path: Path | None = None
    dataset: str
    rule_check_enabled: bool
    require_keys: list[str]
    identifier_pattern: str
    llm_judge_enabled: bool
    sample_rate: float
    api_base_url: str
    api_key: str
    api_model: str
    api_temperature: float
    api_max_tokens: int
    concurrency: int
    fail_on_rule_violation: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QAConfig":
        rc = d.get("rule_check", {})
        lj = d.get("llm_judge", {})
        api = lj.get("api", {}) if lj else {}
        if "api_key" not in api and "api" in d:
            api = d["api"]
        return cls(
            input_path=Path(d["input"]),
            output_path=Path(d["output"]),
            report_path=Path(d["report"]),
            source_input_path=Path(d["source_input"]) if d.get("source_input") else None,
            failed_input_path=Path(d["failed_input"]) if d.get("failed_input") else None,
            dataset=d["dataset"],
            rule_check_enabled=bool(rc.get("enabled", True)),
            require_keys=list(rc.get("require_keys", {}).get(d["dataset"], [])),
            identifier_pattern=rc.get("identifier_pattern", "^[a-z][a-z0-9_]*$"),
            llm_judge_enabled=bool(lj.get("enabled", True)),
            sample_rate=float(lj.get("sample_rate", 0.05)),
            api_base_url=api.get("base_url", ""),
            api_key=api.get("api_key", ""),
            api_model=api.get("model", "qwen3.7-max"),
            api_temperature=float(api.get("temperature", 0.0)),
            api_max_tokens=int(api.get("max_tokens", 1024)),
            concurrency=int(lj.get("concurrency", 10)),
            fail_on_rule_violation=bool(d.get("fail_on_rule_violation", False)),
        )


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def rule_check_sample(
    sample: dict[str, Any],
    cfg: QAConfig,
    original: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(sample, dict):
        return False, ["sample is not a dict"]

    for key in cfg.require_keys:
        if key not in sample:
            errors.append(f"missing required key: {key}")

    if "system" in sample and isinstance(sample["system"], str):
        text = sample["system"]
        if "SYSTEM:" in text and "{" in text:
            import re as _re
            json_part = text[text.find("{"):]
            try:
                schema = json.loads(json_part)
                name = schema.get("name", "")
                if name and not is_snake_case(name):
                    errors.append(f"function name not snake_case: {name!r}")
            except json.JSONDecodeError:
                pass

    if "chat" in sample and isinstance(sample["chat"], str):
        chat = sample["chat"]
        import re as _re
        fns = _re.findall(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"', chat)
        for fn in fns:
            if not is_snake_case(fn):
                errors.append(f"function name not snake_case: {fn!r}")

    if "tools" in sample and isinstance(sample["tools"], str):
        try:
            tools = json.loads(sample["tools"])
            for tool in tools:
                if not is_snake_case(tool.get("name", "")):
                    errors.append(f"tool name not snake_case: {tool.get('name')!r}")
        except json.JSONDecodeError:
            errors.append("tools field is not valid JSON")

    if "answers" in sample and isinstance(sample["answers"], str):
        try:
            answers = json.loads(sample["answers"])
            for ans in answers:
                if not is_snake_case(ans.get("name", "")):
                    errors.append(f"answer tool name not snake_case: {ans.get('name')!r}")
        except json.JSONDecodeError:
            errors.append("answers field is not valid JSON")

    if cfg.dataset == "glaive_normalized":
        if original is not None:
            ok, reason = check_identifier_integrity(original, sample, cfg.dataset)
            if not ok:
                errors.append(reason)
        ok, reason = check_required_fields_present(sample, cfg.dataset)
        if not ok:
            errors.append(reason)
        for call in sample.get("function_calls", []):
            if not isinstance(call, dict) or not is_snake_case(str(call.get("name", ""))):
                errors.append(f"function name not snake_case: {call.get('name') if isinstance(call, dict) else call!r}")
        for tool in sample.get("tools", []):
            if not isinstance(tool, dict) or not is_snake_case(str(tool.get("name", ""))):
                errors.append(f"tool name not snake_case: {tool.get('name') if isinstance(tool, dict) else tool!r}")

    return len(errors) == 0, errors


async def llm_judge_one(
    client: AsyncOpenAI,
    original: dict[str, Any],
    translated: dict[str, Any],
    cfg: QAConfig,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        prompt = (
            "Bạn là chuyên gia QA dịch Anh-Việt cho tool-calling dataset. "
            "Đánh giá bản dịch JSON sau. Trả về JSON với các field:\n"
            '- "natural_score": 1-5 (mức tự nhiên tiếng Việt)\n'
            '- "preserve_identifier": true/false (function name + arg keys giữ nguyên EN)\n'
            '- "preserve_structure": true/false (JSON structure không đổi)\n'
            '- "issues": list[str] (vấn đề phát hiện, [] nếu OK)\n'
            '- "verdict": "pass" / "minor" / "fail"\n\n'
            "ORIGINAL (EN):\n```json\n" + json.dumps(original, ensure_ascii=False, indent=2) + "\n```\n\n"
            "TRANSLATED (VI):\n```json\n" + json.dumps(translated, ensure_ascii=False, indent=2) + "\n```\n\n"
            'Chỉ trả về JSON object, không giải thích thêm.'
        )
        try:
            response = await client.chat.completions.create(
                model=cfg.api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.api_temperature,
                max_tokens=cfg.api_max_tokens,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = "\n".join(
                    ln for ln in content.split("\n") if not ln.strip().startswith("```")
                ).strip()
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last > first:
                content = content[first:last + 1]
            return json.loads(content)
        except Exception as e:
            return {
                "natural_score": 0,
                "preserve_identifier": False,
                "preserve_structure": False,
                "issues": [f"judge error: {type(e).__name__}: {e}"],
                "verdict": "fail",
            }


def load_pairs(translated_path: Path) -> list[tuple[int, dict[str, Any]]]:
    pairs: list[tuple[int, dict[str, Any]]] = []
    with translated_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append((idx, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return pairs


def _load_failed_indices(failed_path: Path | None) -> set[int]:
    indices: set[int] = set()
    if failed_path is None or not failed_path.exists():
        return indices
    with failed_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = d.get("source_index")
            if isinstance(idx, int):
                indices.add(idx)
    return indices


def load_paired_samples(
    source_path: Path | None,
    translated_path: Path,
    failed_path: Path | None,
) -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    translated = load_pairs(translated_path)
    if source_path is None:
        return [(idx, sample, sample) for idx, sample in translated]

    failed = _load_failed_indices(failed_path)
    source_samples: list[tuple[int, dict[str, Any]]] = []
    with source_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                source_samples.append((idx, json.loads(line)))
            except json.JSONDecodeError:
                continue

    pairs: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    t_i = 0
    for idx, original in source_samples:
        if idx in failed:
            continue
        if t_i >= len(translated):
            break
        _, translated_sample = translated[t_i]
        pairs.append((idx, original, translated_sample))
        t_i += 1
    return pairs


async def run_qa(cfg: QAConfig) -> dict[str, Any]:
    logger = setup_logger(f"qa.{cfg.dataset}")
    logger.info("loading translated samples from %s", cfg.input_path)
    pairs = load_paired_samples(cfg.source_input_path, cfg.input_path, cfg.failed_input_path)
    logger.info("loaded %d samples", len(pairs))

    rule_pass = 0
    rule_fail = 0
    rule_errors: dict[str, int] = {}
    rule_violations: list[dict[str, Any]] = []

    if cfg.rule_check_enabled:
        for idx, _original, sample in pairs:
            ok, errors = rule_check_sample(sample, cfg, _original)
            if ok:
                rule_pass += 1
            else:
                rule_fail += 1
                if cfg.fail_on_rule_violation:
                    rule_violations.append({"index": idx, "errors": errors})
                for e in errors:
                    short = e.split(":")[0]
                    rule_errors[short] = rule_errors.get(short, 0) + 1

    judge_results: list[dict[str, Any]] = []
    judge_verdict: dict[str, int] = {"pass": 0, "minor": 0, "fail": 0}

    if cfg.llm_judge_enabled and cfg.api_key:
        rng = random.Random(42)
        n_sample = max(1, int(len(pairs) * cfg.sample_rate))
        sampled = rng.sample(pairs, min(n_sample, len(pairs)))
        logger.info("LLM judge on %d samples (%.0f%% of %d)", len(sampled), cfg.sample_rate * 100, len(pairs))

        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.api_base_url, max_retries=0)
        semaphore = asyncio.Semaphore(cfg.concurrency)

        tasks = [llm_judge_one(client, original, translated, cfg, semaphore) for _, original, translated in sampled]
        results = await asyncio.gather(*tasks)

        for (idx, _original, _translated), result in zip(sampled, results):
            verdict = result.get("verdict", "fail")
            judge_verdict[verdict] = judge_verdict.get(verdict, 0) + 1
            judge_results.append({"index": idx, **result})
    elif cfg.llm_judge_enabled:
        logger.warning("LLM judge enabled but api_key missing; skipping")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        for r in judge_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "dataset": cfg.dataset,
        "input": str(cfg.input_path),
        "n_total": len(pairs),
        "rule_check": {
            "enabled": cfg.rule_check_enabled,
            "pass": rule_pass,
            "fail": rule_fail,
            "pass_rate": round(rule_pass / max(len(pairs), 1), 4),
            "error_breakdown": rule_errors,
            "violations": rule_violations[:20],
        },
        "llm_judge": {
            "enabled": cfg.llm_judge_enabled,
            "n_sampled": len(judge_results),
            "verdict_breakdown": judge_verdict,
            "pass_rate": round(judge_verdict.get("pass", 0) / max(len(judge_results), 1), 4),
        },
    }

    cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("DONE: %s", json.dumps(report, ensure_ascii=False))
    return report


def _resolve_env_placeholders(obj):
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            import os
            name = obj[2:-1]
            value = os.getenv(name)
            if value is None and "__" not in name and "_" in name:
                prefix, rest = name.split("_", 1)
                alt_name = f"{prefix}__{rest}"
                value = os.getenv(alt_name, "")
            return value or ""
        return obj
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    return obj


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_env_placeholders(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="QA translation pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raw = load_yaml(args.config)
    cfg = QAConfig.from_dict(raw)
    asyncio.run(run_qa(cfg))


if __name__ == "__main__":
    main()
