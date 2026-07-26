"""Download tool-calling datasets from HuggingFace.

Datasets:
- glaiveai/glaive-function-calling-v2  (~110k samples, multi-turn chat)
- Salesforce/xlam-function-calling-60k (~60k samples, flat function_call)

Output: data/raw/{glaive,xlam}_raw.jsonl
"""

import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_RAW_DIR = Path("data/raw")

DATASETS: dict[str, str] = {
    "glaive": "glaiveai/glaive-function-calling-v2",
    "xlam": "Salesforce/xlam-function-calling-60k",
}


def _hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    return token if token else None


def _hf_cache() -> str | None:
    cache = os.getenv("HF_HOME")
    return cache if cache else None


def _load_split(hf_name: str, split: str = "train") -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"split": split}
    token = _hf_token()
    if token:
        kwargs["token"] = token
    cache = _hf_cache()
    if cache:
        kwargs["cache_dir"] = cache
    ds = load_dataset(hf_name, **kwargs)
    return list(ds)


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def collect_glaive() -> list[dict[str, Any]]:
    return _load_split(DATASETS["glaive"])


def collect_xlam() -> list[dict[str, Any]]:
    return _load_split(DATASETS["xlam"])


def collect_all(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    print("[1/2] Downloading glaiveai/glaive-function-calling-v2 ...")
    glaive = collect_glaive()
    save_jsonl(glaive, raw_dir / "glaive_raw.jsonl")
    counts["glaive"] = len(glaive)
    print(f"  -> {len(glaive)} samples -> {raw_dir / 'glaive_raw.jsonl'}")

    print("[2/2] Downloading Salesforce/xlam-function-calling-60k ...")
    xlam = collect_xlam()
    save_jsonl(xlam, raw_dir / "xlam_raw.jsonl")
    counts["xlam"] = len(xlam)
    print(f"  -> {len(xlam)} samples -> {raw_dir / 'xlam_raw.jsonl'}")

    return counts


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    counts = collect_all()
    print("\nDone.")
    print(f"  Total samples: {sum(counts.values())}")
    for name, n in counts.items():
        print(f"  - {name}: {n}")


if __name__ == "__main__":
    main()
