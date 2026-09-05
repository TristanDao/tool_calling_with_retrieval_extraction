"""Rank xLAM queries by likely untranslated English outside protected literals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENGLISH = re.compile(
    r"\b(?:what|is|are|the|from|when|with|and|or|also|check|if|contains|any|language|"
    r"text|response|service|filling|spaces|adding|additional|parameters|determine|in|for|"
    r"to|of|on|at|this|that|how|can|could|would|please|find|search|latest|news|today|"
    r"used|use|using|your|my|meaning|life|retrieve|get|provide|list|return|results|about|"
    r"into|available|based|given|need|want|first|second|details|data|current|most|top|all|"
    r"only|following|each|where|which|over|has|have|does|do|will|be|a|an)\b",
    re.IGNORECASE,
)
VIETNAMESE = re.compile(
    r"\b(?:tôi|bạn|hãy|vui lòng|có thể|là|của|cho|với|và|hoặc|trong|trên|từ|đến|"
    r"tìm|lấy|truy xuất|cung cấp|tính|kiểm tra|xác định|danh sách|kết quả|thông tin|"
    r"chi tiết|hiện tại|mới nhất|đầu tiên|thứ hai|bao nhiêu|như thế nào)\b",
    re.IGNORECASE,
)
QUOTED = re.compile(r"(?<!\w)'.*'(?!\w)|\".*?\"")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
CORRUPTION = re.compile(
    r"Bản dịch:|ftrong|trtr|hoặce|fhoặc|trtrêng|schoặce|mtrênthly|trênly|whtại|"
    r"tạieghoặcy|strtr|Parlà|Smột|Pmột|Một[A-Z]|jstrên|hoặcg|mộtlo|đểp|dmộth|"
    r"temộtms|collectitrên|củaficial|potrtrêngts|pythtrên",
    re.IGNORECASE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-english", type=int, default=4)
    args = parser.parse_args()
    findings: list[dict[str, object]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            query = record["query"]
            unprotected = URL.sub("", QUOTED.sub("", query))
            english_count = len(ENGLISH.findall(unprotected))
            vietnamese_count = len(VIETNAMESE.findall(unprotected))
            corrupt = bool(CORRUPTION.search(query))
            if corrupt or english_count >= args.minimum_english:
                findings.append(
                    {
                        "line": line_number,
                        "id": record["id"],
                        "english": english_count,
                        "vietnamese": vietnamese_count,
                        "corrupt": corrupt,
                        "query": query,
                    }
                )
    report = {"matches": len(findings), "examples": findings}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matches": len(findings), "examples": findings[:10]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
