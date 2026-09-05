"""Audit serialized CustomTools-VI artifacts independently from the generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, time
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

SPLIT_ORDER = ["train", "val_seen", "val_unseen", "test_seen", "test_unseen"]
TOP_LEVEL_KEYS = ["id", "source", "query", "function_calls", "tools", "metadata"]
RELATED_DOMAINS = {
    "food": {"travel", "ecommerce"},
    "travel": {"transport", "food"},
    "finance": {"public_services", "ecommerce"},
    "education": {"public_services", "health"},
    "health": {"public_services", "ecommerce"},
    "ecommerce": {"finance", "transport"},
    "transport": {"travel", "public_services"},
    "public_services": {"finance", "transport"},
    "entertainment": {"travel", "food"},
    "agriculture": {"finance", "travel"},
}
ZODIACS = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]
BOOLEAN_SURFACES = {
    "spicy": {True: "có cay", False: "không cay"},
    "family_friendly_only": {
        True: "chỉ lấy nơi phù hợp với gia đình",
        False: "không giới hạn theo tiêu chí gia đình",
    },
    "online": {True: "học online", False: "học trực tiếp"},
    "accepts_bhyt_only": {True: "chỉ lấy cơ sở nhận BHYT", False: "không giới hạn theo BHYT"},
    "open_now_only": {
        True: "chỉ lấy nhà thuốc đang mở cửa",
        False: "không lọc theo trạng thái mở cửa",
    },
    "official_store_only": {
        True: "chỉ lấy gian hàng chính hãng",
        False: "không giới hạn loại gian hàng",
    },
    "include_shipping": {True: "tính cả phí ship", False: "không tính phí ship"},
    "online_only": {
        True: "chỉ lấy thủ tục làm trực tuyến",
        False: "không giới hạn hình thức thực hiện",
    },
    "include_rain_probability": {True: "có xác suất mưa", False: "không cần xác suất mưa"},
    "certified_only": {
        True: "chỉ lấy nhà cung cấp có chứng nhận",
        False: "không giới hạn theo chứng nhận",
    },
    "public_university_only": {
        True: "chỉ tra trường công lập",
        False: "không giới hạn trường công lập",
    },
    "weekend_only": {True: "chỉ lấy lịch thi cuối tuần", False: "không giới hạn ngày thi"},
    "full_funding_only": {True: "chỉ lấy học bổng toàn phần", False: "không giới hạn mức tài trợ"},
    "include_prescription_requirements": {
        True: "có kèm thông tin về yêu cầu đơn thuốc",
        False: "không kèm thông tin về yêu cầu đơn thuốc",
    },
    "include_history": {True: "có lấy lịch sử vận chuyển", False: "không lấy lịch sử vận chuyển"},
}
AWKWARD_PATTERNS = [
    r"\bcó chỉ\b",
    r"không cần phù hợp",
    r"không bắt buộc chứng nhận",
    r"bằng phút\s*:",
    r"\bSố (?:khẩu phần|hành khách|khách lưu trú)\s*:",
    r"\s{2,}",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_query(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    value = re.sub(r"[^\wÀ-ỹ<>]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def query_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE))


def token_prefix(tokens: set[str], threshold: float) -> list[str]:
    ordered = sorted(
        tokens, key=lambda token: hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    )
    prefix_length = len(tokens) - math.ceil(threshold * len(tokens)) + 1
    return ordered[: max(1, prefix_length)]


def exact_lexical_audit(records: list[dict[str, Any]], threshold: float = 0.90) -> dict[str, Any]:
    groups = {"all_records": records}
    pairs: list[dict[str, Any]] = []
    for group_name, group in groups.items():
        prior_tokens: list[set[str]] = []
        prior_records: list[dict[str, Any]] = []
        prefix_index: dict[str, set[int]] = defaultdict(set)
        for record in group:
            tokens = query_tokens(record["query"])
            candidates: set[int] = set()
            for token in token_prefix(tokens, threshold):
                candidates.update(prefix_index[token])
            for index in candidates:
                other = prior_tokens[index]
                if min(len(tokens), len(other)) / max(len(tokens), len(other)) < threshold:
                    continue
                similarity = len(tokens & other) / len(tokens | other)
                if similarity >= threshold:
                    pairs.append(
                        {
                            "group": group_name,
                            "left_id": prior_records[index]["id"],
                            "right_id": record["id"],
                            "left_split": prior_records[index]["metadata"]["split"],
                            "right_split": record["metadata"]["split"],
                            "similarity": similarity,
                        }
                    )
            for token in token_prefix(tokens, threshold):
                prefix_index[token].add(len(prior_tokens))
            prior_tokens.append(tokens)
            prior_records.append(record)
    return {
        "method": "exact word-set Jaccard with complete prefix filtering",
        "threshold": threshold,
        "pairs": len(pairs),
        "cross_split_pairs": sum(pair["left_split"] != pair["right_split"] for pair in pairs),
        "examples": pairs[:20],
    }


def values_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def clean_registry_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tool.items() if not key.startswith("x-")}


def load_jsonl(path: Path, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                errors.append(
                    {"file": path.name, "line": line_number, "error": "line lacks LF terminator"}
                )
            if not line.strip():
                errors.append({"file": path.name, "line": line_number, "error": "blank JSONL line"})
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    {"file": path.name, "line": line_number, "error": f"invalid JSON: {exc}"}
                )
                continue
            if not isinstance(value, dict):
                errors.append(
                    {"file": path.name, "line": line_number, "error": "record is not an object"}
                )
                continue
            records.append(value)
    return records


def verify_manifest(release_dir: Path, config_path: Path, errors: list[dict[str, Any]]) -> None:
    manifest_path = release_dir / "generation_manifest.json"
    if not manifest_path.exists():
        errors.append({"file": manifest_path.name, "error": "manifest missing"})
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generator_path = Path(manifest.get("generator", ""))
    if not generator_path.is_file() or sha256_file(generator_path) != manifest.get(
        "generator_sha256"
    ):
        errors.append({"file": str(generator_path), "error": "manifest generator sha256 mismatch"})
    auditor_path = Path(manifest.get("auditor", ""))
    if not auditor_path.is_file() or sha256_file(auditor_path) != manifest.get("auditor_sha256"):
        errors.append({"file": str(auditor_path), "error": "manifest auditor sha256 mismatch"})
    if sha256_file(config_path) != manifest.get("config_sha256"):
        errors.append({"file": str(config_path), "error": "manifest config sha256 mismatch"})
    for relative, expected in manifest.get("files", {}).items():
        path = release_dir / relative
        if not path.is_file():
            errors.append({"file": relative, "error": "manifest target missing"})
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected.get("sha256"):
            errors.append({"file": relative, "error": "manifest sha256 mismatch"})
        if path.stat().st_size != expected.get("bytes"):
            errors.append({"file": relative, "error": "manifest byte count mismatch"})


def audit_release(release_dir: Path, config_path: Path) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    verify_manifest(release_dir, config_path, errors)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    registry = json.loads((release_dir / "tools.json").read_text(encoding="utf-8"))
    if not isinstance(registry, list):
        raise ValueError("tools.json must contain a list")
    by_name = {tool["name"]: tool for tool in registry}
    if len(registry) != 40 or len(by_name) != 40:
        errors.append({"file": "tools.json", "error": "tool registry must contain 40 unique tools"})
    for tool in registry:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", tool.get("name", "")):
            errors.append({"id": tool.get("name"), "error": "tool name is not snake_case"})
        try:
            Draft202012Validator.check_schema(tool["parameters"])
        except Exception as exc:
            errors.append({"id": tool.get("name"), "error": f"invalid JSON Schema: {exc}"})
    records: list[dict[str, Any]] = []
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    styles: Counter[str] = Counter()
    negative_types: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    tool_positive_counts: Counter[str] = Counter()
    ids: set[str] = set()
    queries: set[str] = set()
    families: dict[str, set[str]] = defaultdict(set)
    expected_serial = 1
    record_checks_passed = 0
    boolean_mentions_checked = 0
    argument_mentions_checked = 0
    repeated_surface_mentions = 0
    for split in SPLIT_ORDER:
        path = release_dir / f"{split}.jsonl"
        split_records = load_jsonl(path, errors)
        records.extend(split_records)
        for record in split_records:
            local: list[str] = []
            rid_value = record.get("id")
            rid = str(rid_value)
            if not isinstance(rid_value, str):
                local.append("id must be a string")
            if list(record) != TOP_LEVEL_KEYS:
                local.append("top-level keys/order mismatch")
            expected_id = f"custom_vi_v1_{expected_serial:06d}"
            if rid != expected_id:
                local.append(f"non-contiguous id; expected {expected_id}")
            expected_serial += 1
            if rid in ids:
                local.append("duplicate id")
            ids.add(rid)
            if record.get("source") != "custom_vi":
                local.append("source mismatch")
            query = record.get("query")
            if not isinstance(query, str) or not query.strip():
                local.append("query must be a non-empty string")
                query = ""
            if query != unicodedata.normalize("NFC", query):
                local.append("query is not Unicode NFC")
            normalized = normalized_query(query)
            if normalized in queries:
                local.append("normalized duplicate query")
            queries.add(normalized)
            if "[[ARG:" in query or "[[/ARG]]" in query:
                local.append("unstripped argument marker")
            if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in AWKWARD_PATTERNS):
                local.append("known awkward-language pattern")
            metadata = record.get("metadata", {})
            if metadata.get("split") != split:
                local.append("metadata split mismatch")
            if metadata.get("dataset_version") != str(raw_config["dataset_version"]):
                local.append("dataset version mismatch")
            if metadata.get("as_of") != str(raw_config["as_of"]):
                local.append("as_of mismatch")
            if metadata.get("reference_datetime") != str(raw_config["reference_datetime"]):
                local.append("reference datetime mismatch")
            expected_seed = int(raw_config["seed"]) * 100_000 + expected_serial - 2
            if metadata.get("candidate_seed") != expected_seed:
                local.append("candidate seed mismatch")
            candidates = record.get("tools", [])
            candidate_names = [
                candidate.get("name") for candidate in candidates if isinstance(candidate, dict)
            ]
            if len(candidates) != int(raw_config["candidate_count"]) or len(
                set(candidate_names)
            ) != len(candidate_names):
                local.append("candidate count/uniqueness mismatch")
            for candidate in candidates:
                candidate_name = candidate.get("name")
                if candidate_name not in by_name:
                    local.append("unknown candidate tool")
                elif candidate != clean_registry_tool(by_name[candidate_name]):
                    local.append("candidate schema differs from registry")
            allowed_splits = {"seen"}
            if split == "val_unseen":
                allowed_splits.add("dev_unseen")
            if split.startswith("test"):
                allowed_splits.add("test_unseen")
            if any(
                by_name[name]["x-tool-split"] not in allowed_splits
                for name in candidate_names
                if name in by_name
            ):
                local.append("candidate registry leakage")
            calls = record.get("function_calls", [])
            expected_mentions: dict[tuple[int, str], Any] = {}
            for call_index, call in enumerate(calls):
                if list(call) != ["name", "arguments"]:
                    local.append("function call keys/order mismatch")
                name = call.get("name")
                if name not in candidate_names:
                    local.append("gold tool absent from candidates")
                    continue
                expected_tool_split = (
                    "dev_unseen"
                    if split == "val_unseen"
                    else "test_unseen"
                    if split == "test_unseen"
                    else "seen"
                )
                if by_name[name]["x-tool-split"] != expected_tool_split:
                    local.append("gold tool split mismatch")
                validator = Draft202012Validator(
                    by_name[name]["parameters"], format_checker=FormatChecker()
                )
                if list(validator.iter_errors(call.get("arguments"))):
                    local.append("gold arguments fail JSON Schema")
                for key, value in call.get("arguments", {}).items():
                    expected_mentions[(call_index, key)] = value
                    schema = by_name[name]["parameters"]["properties"][key]
                    if schema.get("format") == "date":
                        try:
                            date.fromisoformat(value)
                        except (TypeError, ValueError):
                            local.append("invalid canonical date")
                    if schema.get("format") == "time":
                        try:
                            time.fromisoformat(value)
                        except (TypeError, ValueError):
                            local.append("invalid canonical time")
                tool_positive_counts[name] += 1
                if name == "vi_lookup_zodiac_information" and "zodiac" in call.get("arguments", {}):
                    expected_zodiac = ZODIACS[
                        (int(call["arguments"]["birth_year"]) - 4) % len(ZODIACS)
                    ]
                    if call["arguments"]["zodiac"] != expected_zodiac:
                        local.append("birth year and zodiac mismatch")
                if name == "vi_convert_currency" and call["arguments"].get("from_currency") == call[
                    "arguments"
                ].get("to_currency"):
                    local.append("currency source and target are identical")
                if name in {
                    "vi_search_domestic_flights",
                    "vi_search_intercity_buses",
                    "vi_calculate_route_distance",
                } and call["arguments"].get("origin") == call["arguments"].get("destination"):
                    local.append("route origin and destination are identical")
                if name == "vi_calculate_registration_fee":
                    asset = call["arguments"].get("asset_type")
                    value = call["arguments"].get("asset_value_vnd")
                    plausible_ranges = {
                        "ô tô": (100_000_000, 20_000_000_000),
                        "xe máy": (5_000_000, 500_000_000),
                        "nhà đất": (100_000_000, 100_000_000_000),
                    }
                    if (
                        asset in plausible_ranges
                        and not plausible_ranges[asset][0] <= value <= plausible_ranges[asset][1]
                    ):
                        local.append("implausible registration-fee asset value")
            actual_mentions: dict[tuple[int, str], Any] = {}
            spans: list[tuple[int, int]] = []
            for mention in metadata.get("argument_mentions", []):
                key = (mention.get("call_index"), mention.get("parameter_path"))
                start = mention.get("start")
                end = mention.get("end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(query)
                ):
                    local.append("invalid mention span bounds")
                    continue
                if query[start:end] != mention.get("surface"):
                    local.append("mention surface/offset mismatch")
                argument_mentions_checked += 1
                if query.count(str(mention.get("surface"))) > 1:
                    repeated_surface_mentions += 1
                if key in actual_mentions:
                    local.append("duplicate argument mention")
                actual_mentions[key] = mention.get("canonical_value")
                spans.append((start, end))
                parameter = str(mention.get("parameter_path"))
                canonical = mention.get("canonical_value")
                if parameter in BOOLEAN_SURFACES and isinstance(canonical, bool):
                    boolean_mentions_checked += 1
                    if mention.get("surface") != BOOLEAN_SURFACES[parameter][canonical]:
                        local.append("boolean surface does not entail canonical value")
            if set(actual_mentions) != set(expected_mentions):
                local.append("argument mention coverage mismatch")
            for key, value in expected_mentions.items():
                if key in actual_mentions and not values_equal(actual_mentions[key], value):
                    local.append("argument mention canonical value mismatch")
            for left, right in zip(sorted(spans), sorted(spans)[1:], strict=False):
                if left[1] > right[0]:
                    local.append("argument mention spans overlap")
            if calls and metadata.get("negative_type") is not None:
                local.append("positive sample has negative_type")
            if (
                not calls
                and metadata.get("negative_type") not in raw_config["negative_distribution"]
            ):
                local.append("negative sample has invalid negative_type")
            if any(
                call.get("name") == "vi_lookup_agricultural_prices"
                and "date" not in call.get("arguments", {})
                for call in calls
            ) and re.search(r"\bhôm nay\b", query, flags=re.IGNORECASE):
                local.append("query implies current date but date argument is absent")
            if calls:
                primary = by_name[calls[0]["name"]]
                if metadata.get("domain") != primary["x-domain"]:
                    local.append("metadata domain differs from primary tool")
                if metadata.get("tool_split") != primary["x-tool-split"]:
                    local.append("metadata tool_split differs from primary tool")
            elif metadata.get("tool_split") != "none":
                local.append("negative tool_split must be none")
            if not calls:
                hard_names = metadata.get("hard_distractor_names", [])
                if not set(hard_names).issubset(candidate_names):
                    local.append("hard distractor absent from candidates")
                for name in hard_names:
                    if name in by_name and by_name[name]["x-domain"] not in {
                        metadata.get("domain"),
                        *RELATED_DOMAINS[metadata.get("domain", "")],
                    }:
                        local.append("hard distractor is not domain-related")
                if metadata.get("negative_type") == "hard_near_miss" and len(hard_names) < 5:
                    local.append("hard near-miss has fewer than five hard distractors")
            split_counts[split]["positive" if calls else "negative"] += 1
            styles[metadata.get("query_style")] += 1
            if metadata.get("negative_type"):
                negative_types[metadata["negative_type"]] += 1
            domains[metadata.get("domain")] += 1
            families[metadata.get("scenario_family_id")].add(split)
            if local:
                errors.append({"id": rid, "errors": sorted(set(local))})
            else:
                record_checks_passed += 1
    for split, quota in raw_config["splits"].items():
        for label in ("positive", "negative"):
            if split_counts[split][label] != int(quota[label]):
                errors.append({"split": split, "error": f"{label} quota mismatch"})
    leaked_families = [family for family, splits in families.items() if len(splits) > 1]
    if leaked_families:
        errors.append(
            {"error": "cross-split scenario family leakage", "count": len(leaked_families)}
        )
    for tool_split in ("seen", "dev_unseen", "test_unseen"):
        tier_counts = [
            tool_positive_counts[tool["name"]]
            for tool in registry
            if tool["x-tool-split"] == tool_split
        ]
        if not tier_counts or min(tier_counts) == 0 or max(tier_counts) - min(tier_counts) > 1:
            errors.append(
                {
                    "error": "positive call coverage is missing or imbalanced within tool split",
                    "tool_split": tool_split,
                    "counts": tier_counts,
                }
            )
    lexical = exact_lexical_audit(records)
    if lexical["pairs"]:
        errors.append({"error": "lexical near-duplicate pairs found", "count": lexical["pairs"]})
    return {
        "dataset": "CustomTools-VI",
        "version": str(raw_config["dataset_version"]),
        "audit_kind": "independent serialized-artifact audit",
        "status": "passed" if not errors else "failed",
        "total_records": len(records),
        "total_tools": len(registry),
        "record_level_checks_passed": record_checks_passed,
        "manifest_integrity": "passed"
        if not any(error.get("error", "").startswith("manifest") for error in errors)
        else "failed",
        "split_counts": {split: dict(counts) for split, counts in split_counts.items()},
        "style_counts": dict(styles),
        "negative_type_counts": dict(negative_types),
        "domain_counts": dict(domains),
        "positive_call_counts_by_tool": dict(sorted(tool_positive_counts.items())),
        "normalized_duplicate_queries": len(records) - len(queries),
        "cross_split_scenario_family_leakage": len(leaked_families),
        "boolean_mentions_checked": boolean_mentions_checked,
        "argument_mentions_checked": argument_mentions_checked,
        "repeated_surface_mentions_with_exact_offsets": repeated_surface_mentions,
        "lexical_near_duplicate_audit": lexical,
        "errors": errors,
        "limitations": [
            "This audit checks every serialized record but does not constitute human annotation.",
            "BGE-M3 semantic embedding deduplication was not run because model weights are not locally available.",
            "No external API was executed to validate real-world tool results.",
        ],
    }


def write_audit_report(release_dir: Path, report: dict[str, Any]) -> None:
    path = release_dir / "independent_audit.json"
    temp = path.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    manifest_path = release_dir / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = sorted(
        item
        for item in release_dir.rglob("*")
        if item.is_file() and item.name != manifest_path.name
    )
    manifest["files"] = {
        str(item.relative_to(release_dir)).replace("\\", "/"): {
            "sha256": sha256_file(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    }
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    with temp_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_manifest, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit serialized CustomTools-VI artifacts")
    parser.add_argument("--release-dir", type=Path, default=Path("data/custom_vi/v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/data/custom_vi.yaml"))
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = audit_release(args.release_dir, args.config)
    if args.write_report:
        write_audit_report(args.release_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
