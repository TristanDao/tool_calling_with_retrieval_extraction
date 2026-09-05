"""Export untranslated natural-language argument values for interactive review."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NATURAL_KEYS = {"q", "query", "keyword", "keywords", "text", "caption"}
SKIP = re.compile(r"(?:https?://|www\.|@|\b[\w.+-]+@[\w.-]+\b|^[\w.-]+://|^[A-Z]{2,}[A-Z0-9_/-]*$)")
STOPWORDS = {
    "a", "an", "and", "are", "at", "can", "could", "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "please", "the", "this", "to", "was", "what", "where", "with", "you",
}


def is_candidate(key: str, value: object) -> bool:
    if key not in NATURAL_KEYS or not isinstance(value, str) or " " not in value:
        return False
    text = value.strip()
    if len(text) < 4 or SKIP.search(text):
        return False
    if any(ch in text for ch in ("://", "()", "{}", "[]", "=", "<", ">")):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text.lower())
    if len(words) < 2:
        return False
    if key == "text":
        return len(words) >= 4
    return text[:1].islower() or bool(set(words) & STOPWORDS)


def walk(value: Any, path: str, out: dict[tuple[str, str], int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if is_candidate(key, item):
                out[(key, item)] = out.get((key, item), 0) + 1
            walk(item, f"{path}.{key}", out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, f"{path}[{index}]", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("translated", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    values: dict[tuple[str, str], int] = {}
    for source_line, translated_line in zip(args.source.open(encoding="utf-8"), args.translated.open(encoding="utf-8")):
        source = json.loads(source_line)
        translated = json.loads(translated_line)
        def collect(src: Any, dst: Any, path: str = "") -> None:
            if isinstance(src, dict) and isinstance(dst, dict):
                for key in src:
                    if key in dst:
                        if is_candidate(key, src[key]) and src[key] == dst[key]:
                            values[(key, src[key])] = values.get((key, src[key]), 0) + 1
                        collect(src[key], dst[key], f"{path}.{key}")
            elif isinstance(src, list) and isinstance(dst, list):
                for index, (left, right) in enumerate(zip(src, dst)):
                    collect(left, right, f"{path}[{index}]")
        collect(source, translated)
    items = [{"key": key, "source": value, "occurrences": count} for (key, value), count in sorted(values.items())]
    print(json.dumps(items[args.offset : args.offset + args.limit], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
