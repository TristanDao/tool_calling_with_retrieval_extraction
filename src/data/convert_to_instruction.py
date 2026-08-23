"""Convert master schema → LLaMA-Factory sharegpt format for Method 1 (SLM).

Input:  data/benchmark_vi/{train,val,test}.jsonl  (master schema)
Output: data/benchmark_vi/instruction/{train,val,test}_chat.jsonl  (sharegpt)

**Decontamination**: benchmark gốc chia split theo sample chứ không theo query
nên 1,718 normalized query nằm ở nhiều split. Method 1 phải dùng **cùng một**
`data/method2/decontamination.json` với Bi-Encoder và Cross-Encoder — nếu không,
Method 1 có train→test leakage trong khi Method 2 không, và bảng so sánh giữa
hai method mất hiệu lực. Xem `docs/method2.md` §5.

Tập test không mất sample nào (test có thứ tự ưu tiên cao nhất), nên bốn method
vẫn được đánh giá trên đúng cùng một tập.

Master schema:
  {"id": "...", "source": "...", "query": "VI text",
   "function_calls": [{"name": "...", "arguments": {...}}],
   "tools": [{"name": "...", "description": "...", "parameters": {...}}]}

Sharegpt format (LLaMA-Factory):
  {
    "conversations": [
      {"from": "system", "value": "You are a tool-calling assistant. Available tools:\n<tools_json>"},
      {"from": "human", "value": "<query>"},
      {"from": "gpt", "value": "<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>"}
    ]
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT_VI = (
    "Bạn là một trợ lý AI có khả năng gọi công cụ (tool calling). "
    "Dưới đây là danh sách các công cụ có sẵn. "
    "Nếu cần gọi công cụ, hãy trả lời đúng định dạng:\n"
    "<tool_call>{\"name\": \"<tên_công_cụ>\", \"arguments\": {<tham_số>}}</tool_call>\n"
    "Nếu không cần gọi công cụ, trả lời: <no_tool_call>\n\n"
    "Công cụ có sẵn:\n"
)


def _format_tool_for_prompt(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {}),
    }


def _format_tool_call_for_prompt(fc: dict[str, Any]) -> str:
    return f'<tool_call>{json.dumps(fc, ensure_ascii=False)}</tool_call>'


def convert_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    query = sample.get("query", "")
    function_calls = sample.get("function_calls", [])
    tools = sample.get("tools", [])

    if not query or not function_calls:
        return None

    tools_json = json.dumps(
        [_format_tool_for_prompt(t) for t in tools],
        ensure_ascii=False,
        indent=2,
    )

    system_content = SYSTEM_PROMPT_VI + tools_json

    assistant_content: str
    if len(function_calls) == 1:
        assistant_content = _format_tool_call_for_prompt(function_calls[0])
    else:
        assistant_content = "\n".join(
            _format_tool_call_for_prompt(fc) for fc in function_calls
        )

    return {
        "conversations": [
            {"from": "system", "value": system_content},
            {"from": "human", "value": query},
            {"from": "gpt", "value": assistant_content},
        ],
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    decontamination: Any = None,
    split: str | None = None,
) -> tuple[int, int, int]:
    """Trả `(converted, skipped, contaminated)`.

    `contaminated` = số sample bị loại vì query của nó thuộc split cao hơn.
    """
    from src.models.sources import normalize_query_key

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    contaminated = 0

    with output_path.open("w", encoding="utf-8") as out:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if decontamination is not None and split is not None:
                    effective = decontamination.effective_split.get(
                        normalize_query_key(sample.get("query", ""))
                    )
                    if effective is not None and effective != split:
                        contaminated += 1
                        continue

                result = convert_sample(sample)
                if result is None:
                    skipped += 1
                    continue

                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                converted += 1

    return converted, skipped, contaminated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert master schema → LLaMA-Factory sharegpt format"
    )
    parser.add_argument(
        "--input", type=Path,
        help="Input JSONL (master schema)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSONL (sharegpt format)",
    )
    parser.add_argument(
        "--all-splits", action="store_true",
        help="Convert all splits (train, val, test) from benchmark dir",
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=Path("data/benchmark_vi"),
        help="Benchmark directory (used with --all-splits)",
    )
    parser.add_argument(
        "--decontamination", type=Path,
        default=Path("data/method2/decontamination.json"),
        help="Index dùng chung với Method 2. Bắt buộc trừ khi --no-decontamination.",
    )
    parser.add_argument(
        "--no-decontamination", action="store_true",
        help="BỎ QUA decontamination — kết quả sẽ có train/test leakage, "
             "không dùng được để so với Method 2",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Split của --input (train/val/test), cần khi có decontamination",
    )
    args = parser.parse_args()

    decontamination = None
    if args.no_decontamination:
        print(
            "[convert] CẢNH BÁO: bỏ qua decontamination. Method 1 sẽ train trên "
            "query có mặt ở val/test, kết quả KHÔNG so sánh được với Method 2."
        )
    else:
        from src.models.sources import load_decontamination

        decontamination = load_decontamination(args.decontamination)
        print(f"[convert] decontamination: {args.decontamination}")

    totals: dict[str, dict[str, int]] = {}
    if args.all_splits:
        for split in ["train", "val", "test"]:
            inp = args.benchmark_dir / f"{split}.jsonl"
            out = args.benchmark_dir / "instruction" / f"{split}_chat.jsonl"
            if not inp.exists():
                print(f"[convert] SKIP {split} — {inp} not found")
                continue
            c, s, contaminated = convert_file(inp, out, decontamination, split)
            print(
                f"[convert] {split}: {c} converted, {s} skipped, "
                f"{contaminated} loại do trùng split khác → {out}"
            )
            totals[split] = {"converted": c, "skipped": s, "contaminated": contaminated}
    else:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required unless --all-splits is used")
        c, s, contaminated = convert_file(
            args.input, args.output, decontamination, args.split
        )
        print(
            f"[convert] {c} converted, {s} skipped, "
            f"{contaminated} loại do trùng split khác → {args.output}"
        )


if __name__ == "__main__":
    main()
