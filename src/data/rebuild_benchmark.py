"""Build and promote a frozen paired English/Vietnamese benchmark revision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.data.feature_group_classify import load_cache
from src.data.normalize_schema import normalize_tool
from src.data.translate_guidelines import is_snake_case

DEFAULT_REVISION = "2026-09-02-full-dedup-seed42"
DEFAULT_BENCHMARK_ROOT = Path("data/benchmark_core")
DEFAULT_ACTIVE_DIR = Path("data/benchmark_vi")
DEFAULT_FEATURE_GROUP_CONFIG = Path("configs/data/feature_group.yaml")
DEFAULT_CACHE_PATH = Path("data/benchmark_vi/.cache/feature_group.json")
_ALLOWED_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


@dataclass(frozen=True, slots=True)
class RevisionBuildConfig:
    input_en_glaive: Path
    input_en_glaive_negative: Path
    input_en_xlam: Path
    input_vi_glaive: Path
    input_vi_glaive_negative: Path
    input_vi_xlam: Path
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT
    active_dir: Path = DEFAULT_ACTIVE_DIR
    revision: str = DEFAULT_REVISION
    feature_group_cache: Path | None = DEFAULT_CACHE_PATH
    feature_group_config: Path = DEFAULT_FEATURE_GROUP_CONFIG
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    promote: bool = True
    overwrite_revision: bool = False
    force_active: bool = False

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RevisionBuildConfig:
        """Create a build config from the repository YAML config."""
        inputs = values.get("inputs", {})
        split = values.get("split", {})

        def required_path(name: str, default: Path) -> Path:
            value = inputs.get(name, values.get(name))
            return Path(value) if value else default

        def optional_path(name: str, default: Path | None) -> Path | None:
            value = values.get(name, default)
            return Path(value) if value else None

        return cls(
            input_en_glaive=required_path(
                "en_glaive", Path("data/normalized_en/glaive_normalized.jsonl")
            ),
            input_en_glaive_negative=required_path(
                "en_glaive_negative", Path("data/normalized_en/glaive_negative.jsonl")
            ),
            input_en_xlam=required_path(
                "en_xlam", Path("data/normalized_en/xlam_normalized.jsonl")
            ),
            input_vi_glaive=required_path(
                "vi_glaive", Path("data/translations/glaive_normalized_vi.jsonl")
            ),
            input_vi_glaive_negative=required_path(
                "vi_glaive_negative", Path("data/translations/glaive_negative_vi.jsonl")
            ),
            input_vi_xlam=required_path(
                "vi_xlam", Path("data/translations/xlam_normalized_vi.jsonl")
            ),
            benchmark_root=Path(values.get("benchmark_root", DEFAULT_BENCHMARK_ROOT)),
            active_dir=Path(values.get("active_dir", DEFAULT_ACTIVE_DIR)),
            revision=str(values.get("revision", DEFAULT_REVISION)),
            feature_group_cache=optional_path("feature_group_cache", DEFAULT_CACHE_PATH),
            feature_group_config=Path(
                values.get("feature_group_config", DEFAULT_FEATURE_GROUP_CONFIG)
            ),
            train_ratio=float(split.get("train_ratio", values.get("train_ratio", 0.8))),
            val_ratio=float(split.get("val_ratio", values.get("val_ratio", 0.1))),
            test_ratio=float(split.get("test_ratio", values.get("test_ratio", 0.1))),
            seed=int(split.get("seed", values.get("seed", 42))),
            promote=bool(values.get("promote", True)),
            overwrite_revision=bool(values.get("overwrite_revision", False)),
            force_active=bool(values.get("force_active", False)),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _query_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        ignored = {"description", "default", "feature_group"}
        return {key: _schema_shape(child) for key, child in value.items() if key not in ignored}
    if isinstance(value, list):
        return [_schema_shape(child) for child in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _call_shape(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls = record.get("function_calls", [])
    return [
        {
            "name": call.get("name"),
            "argument_keys": sorted(
                call.get("arguments", {}).keys()
                if isinstance(call.get("arguments"), dict)
                else []
            ),
        }
        for call in calls
        if isinstance(call, dict)
    ]


def _tool_shape(record: dict[str, Any]) -> list[dict[str, Any]]:
    tools = record.get("tools", [])
    return sorted(
        [
            {
                "name": tool.get("name"),
                "parameters": _schema_shape(tool.get("parameters", {})),
            }
            for tool in tools
            if isinstance(tool, dict)
        ],
        key=lambda value: str(value["name"]),
    )


def _canonical_record(record: dict[str, Any], expected_source: str) -> dict[str, Any]:
    source = record.get("source", expected_source)
    if source != expected_source:
        raise ValueError(f"Source mismatch for {record.get('id')}: {source!r} != {expected_source!r}")
    sample_id = record.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"Invalid record id: {sample_id!r}")
    raw_tools = record.get("tools", [])
    tools: list[dict[str, Any]] | Any = raw_tools
    if isinstance(raw_tools, list):
        normalized_tools: list[dict[str, Any]] = []
        definitions_by_name: dict[str, str] = {}
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            tool = normalize_tool(raw_tool, dataset=expected_source)
            name = str(tool.get("name", ""))
            definition = _canonical_json(tool)
            if name in definitions_by_name and definitions_by_name[name] == definition:
                continue
            definitions_by_name.setdefault(name, definition)
            normalized_tools.append(tool)
        tools = normalized_tools
    return {
        "id": sample_id,
        "source": expected_source,
        "query": record.get("query", ""),
        "function_calls": copy.deepcopy(record.get("function_calls", [])),
        "tools": copy.deepcopy(tools),
    }


def _load_source_map(paths: list[tuple[Path, str]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    records: dict[str, dict[str, Any]] = {}
    counts: dict[str, dict[str, int]] = {}
    for path, source in paths:
        rows = _read_jsonl(path)
        positive = 0
        negative = 0
        for row in rows:
            record = _canonical_record(row, source)
            if record["id"] in records:
                raise ValueError(f"Duplicate source ID {record['id']} in {path}")
            records[record["id"]] = record
            if record["function_calls"]:
                positive += 1
            else:
                negative += 1
        counts[str(path)] = {"records": len(rows), "positive": positive, "negative": negative}
    return records, counts


def _pair_records(
    english: dict[str, dict[str, Any]], vietnamese: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    english_ids = set(english)
    vietnamese_ids = set(vietnamese)
    missing_vi = sorted(english_ids - vietnamese_ids)
    missing_en = sorted(vietnamese_ids - english_ids)
    if missing_vi or missing_en:
        raise ValueError(
            f"EN/VI IDs are not paired: missing_vi={len(missing_vi)}, missing_en={len(missing_en)}"
        )

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for sample_id, english_record in english.items():
        vietnamese_record = vietnamese[sample_id]
        if english_record["source"] != vietnamese_record["source"]:
            raise ValueError(f"Source mismatch in pair {sample_id}")
        if bool(english_record["function_calls"]) != bool(vietnamese_record["function_calls"]):
            raise ValueError(f"Positive/negative label changed in pair {sample_id}")
        if _call_shape(english_record) != _call_shape(vietnamese_record):
            raise ValueError(f"Function-call structure changed in pair {sample_id}")
        if _tool_shape(english_record) != _tool_shape(vietnamese_record):
            raise ValueError(f"Tool schema structure changed in pair {sample_id}")
        pairs.append((english_record, vietnamese_record))
    return pairs


def _deduplicate_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    seen_ids: set[str] = set()
    seen_scenarios: set[str] = set()
    unique: list[tuple[dict[str, Any], dict[str, Any]]] = []
    duplicate_reasons: Counter[str] = Counter()
    for english, vietnamese in pairs:
        sample_id = english["id"]
        scenario_key = _canonical_json(
            {
                "source": english["source"],
                "query": _query_key(english["query"]),
                "function_calls": english["function_calls"],
                "tools": _tool_shape(english),
            }
        )
        reason: str | None = None
        if sample_id in seen_ids:
            reason = "duplicate_id"
        elif scenario_key in seen_scenarios:
            reason = "duplicate_scenario"
        if reason is not None:
            duplicate_reasons[reason] += 1
            continue
        seen_ids.add(sample_id)
        seen_scenarios.add(scenario_key)
        unique.append((english, vietnamese))
    return unique, dict(duplicate_reasons)


def _validate_schema(value: Any, path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{path} is not an object"]
    schema_type = value.get("type")
    if schema_type not in _ALLOWED_TYPES:
        errors.append(f"{path}.type is invalid: {schema_type!r}")
    properties = value.get("properties")
    if schema_type == "object":
        if properties is not None and not isinstance(properties, dict):
            errors.append(f"{path}.properties is not an object")
        elif isinstance(properties, dict):
            for name, child in properties.items():
                errors.extend(_validate_schema(child, f"{path}.properties.{name}"))
        required = value.get("required", [])
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            errors.append(f"{path}.required is invalid")
        elif isinstance(properties, dict):
            missing = sorted(set(required) - set(properties))
            if missing:
                errors.append(f"{path}.required references missing properties: {missing}")
    if schema_type == "array" and "items" in value:
        errors.extend(_validate_schema(value["items"], f"{path}.items"))
    if "enum" in value and not isinstance(value["enum"], list):
        errors.append(f"{path}.enum is not a list")
    return errors


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate one canonical record, allowing an empty function-call list."""
    errors: list[str] = []
    sample_id = record.get("id")
    source = record.get("source")
    if not isinstance(sample_id, str) or not sample_id:
        errors.append("missing id")
    if source not in {"glaive", "xlam", "custom_vi"}:
        errors.append(f"invalid source: {source!r}")
    elif isinstance(sample_id, str) and not sample_id.startswith(f"{source}_"):
        errors.append(f"id does not match source: {sample_id!r}")
    query = record.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append("missing or invalid query")
    calls = record.get("function_calls")
    tools = record.get("tools")
    if not isinstance(calls, list):
        errors.append("function_calls is not a list")
    if not isinstance(tools, list) or not tools:
        errors.append("tools is empty or invalid")
    if errors:
        return False, errors

    tool_names: set[str] = set()
    tools_by_name: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            errors.append("tool is not an object")
            continue
        name = tool.get("name", "")
        if not isinstance(name, str) or not is_snake_case(name):
            errors.append(f"tool name is not snake_case: {name!r}")
        elif name in tool_names:
            errors.append(f"duplicate tool name: {name}")
        else:
            tool_names.add(name)
            tools_by_name[name] = tool
        if not isinstance(tool.get("description", ""), str):
            errors.append(f"tool description is not a string: {name!r}")
        parameters = tool.get("parameters")
        errors.extend(_validate_schema(parameters, f"tool[{name}].parameters"))
        if isinstance(parameters, dict) and parameters.get("type") != "object":
            errors.append(f"tool parameters are not object: {name!r}")

    for call in calls:
        if not isinstance(call, dict):
            errors.append("function_call is not an object")
            continue
        name = call.get("name", "")
        if not isinstance(name, str) or not is_snake_case(name):
            errors.append(f"function name is not snake_case: {name!r}")
        elif name not in tool_names:
            errors.append(f"function_call name not in tools: {name}")
        if not isinstance(call.get("arguments"), dict):
            errors.append(f"function_call arguments are not object: {name!r}")
        else:
            parameters = tools_by_name.get(name, {}).get("parameters", {})
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            if isinstance(properties, dict):
                extra = sorted(set(call["arguments"]) - set(properties))
                if extra:
                    errors.append(f"arguments not declared by schema for {name}: {extra}")
                required = parameters.get("required", [])
                if isinstance(required, list):
                    missing = sorted(set(required) - set(call["arguments"]))
                    if missing:
                        errors.append(f"required arguments missing for {name}: {missing}")
            for argument_name in call["arguments"]:
                if not isinstance(argument_name, str) or not is_snake_case(argument_name):
                    errors.append(f"argument key is not snake_case: {argument_name!r}")
    return len(errors) == 0, errors


def _filter_valid_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    valid: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for english, vietnamese in pairs:
        en_ok, en_errors = validate_record(english)
        vi_ok, vi_errors = validate_record(vietnamese)
        if en_ok and vi_ok:
            valid.append((english, vietnamese))
            continue
        rejected.append(
            {
                "id": english.get("id"),
                "source": english.get("source"),
                "english_errors": en_errors,
                "vietnamese_errors": vi_errors,
            }
        )
    return valid, rejected


def _attach_feature_groups(
    records: Iterable[dict[str, Any]], cache: dict[str, str],
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for record in records:
        updated = copy.deepcopy(record)
        for tool in updated["tools"]:
            tool["feature_group"] = cache.get(tool["name"], "Khác")
        attached.append(updated)
    return attached


def _build_tool_pool(records: Iterable[dict[str, Any]], cache: dict[str, str]) -> list[dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for record in records:
        for tool in record["tools"]:
            name = tool["name"]
            if name not in pool:
                pool[name] = {
                    "name": name,
                    "description": tool.get("description", ""),
                    "feature_group": cache.get(name, "Khác"),
                    "parameters": copy.deepcopy(tool.get("parameters", {})),
                }
                continue
            current = pool[name]
            if not current["description"] and tool.get("description"):
                current["description"] = tool["description"]
            current_properties = current["parameters"].get("properties", {})
            new_properties = tool.get("parameters", {}).get("properties", {})
            if not current_properties and new_properties:
                current["parameters"] = copy.deepcopy(tool["parameters"])
    return [pool[name] for name in sorted(pool)]


def _split_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    split_names = ("train", "val", "test")
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        root = parent.setdefault(node, node)
        if root != node:
            parent[node] = find(root)
        return parent[node]

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for english, vietnamese in pairs:
        en_node = ("en", _query_key(english["query"]))
        vi_node = ("vi", _query_key(vietnamese["query"]))
        union(en_node, vi_node)

    ids_by_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    pair_by_id = {english["id"]: (english, vietnamese) for english, vietnamese in pairs}
    for sample_id in pair_by_id:
        english, vietnamese = pair_by_id[sample_id]
        root = find(("en", _query_key(english["query"])))
        ids_by_group[root].append(sample_id)

    groups_by_stratum: dict[tuple[str, bool], list[tuple[str, str]]] = defaultdict(list)
    for root, sample_ids in ids_by_group.items():
        strata = Counter(
            (pair_by_id[sample_id][0]["source"], bool(pair_by_id[sample_id][0]["function_calls"]))
            for sample_id in sample_ids
        )
        dominant_stratum = min(
            strata,
            key=lambda value: (-strata[value], value),
        )
        groups_by_stratum[dominant_stratum].append(root)

    assignments: dict[str, str] = {}
    for stratum_index, stratum in enumerate(sorted(groups_by_stratum)):
        group_roots = groups_by_stratum[stratum]
        random.Random(seed + stratum_index).shuffle(group_roots)
        total = sum(len(ids_by_group[root]) for root in group_roots)
        target = {name: total * ratios[name] for name in split_names}
        assigned = {name: 0 for name in split_names}
        for root in group_roots:
            split = max(
                split_names,
                key=lambda name: (target[name] - assigned[name], -split_names.index(name)),
            )
            for sample_id in ids_by_group[root]:
                assignments[sample_id] = split
            assigned[split] += len(ids_by_group[root])
    if len(assignments) != len(pairs):
        raise ValueError("split assignment does not cover every record")
    return assignments


def _split_records(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], assignments: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    english_splits = {name: [] for name in ("train", "val", "test")}
    vietnamese_splits = {name: [] for name in ("train", "val", "test")}
    for english, vietnamese in pairs:
        split = assignments[english["id"]]
        english_splits[split].append(english)
        vietnamese_splits[split].append(vietnamese)
    return english_splits, vietnamese_splits


def _validate_splits(
    english_splits: dict[str, list[dict[str, Any]]],
    vietnamese_splits: dict[str, list[dict[str, Any]]],
) -> None:
    for language, splits in (("en", english_splits), ("vi", vietnamese_splits)):
        split_ids: dict[str, set[str]] = {}
        split_queries: dict[str, set[str]] = {}
        split_scenarios: dict[str, set[str]] = {}
        for split, records in splits.items():
            ids: set[str] = set()
            queries: set[str] = set()
            scenarios: set[str] = set()
            for record in records:
                ok, errors = validate_record(record)
                if not ok:
                    raise ValueError(f"Invalid {language} record {record.get('id')}: {errors}")
                if record["id"] in ids:
                    raise ValueError(f"Duplicate {language} ID in {split}: {record['id']}")
                ids.add(record["id"])
                query_key = _query_key(record["query"])
                queries.add(query_key)
                scenarios.add(
                    _canonical_json(
                        {
                            "source": record["source"],
                            "query": query_key,
                            "function_calls": record["function_calls"],
                            "tools": _tool_shape(record),
                        }
                    )
                )
            split_ids[split] = ids
            split_queries[split] = queries
            split_scenarios[split] = scenarios
        query_to_split: dict[str, str] = {}
        scenario_to_split: dict[str, str] = {}
        for split, queries in split_queries.items():
            for query in queries:
                previous = query_to_split.setdefault(query, split)
                if previous != split:
                    raise ValueError(f"{language} query leakage between {previous}/{split}")
            for scenario in split_scenarios[split]:
                previous = scenario_to_split.setdefault(scenario, split)
                if previous != split:
                    raise ValueError(f"{language} scenario leakage between {previous}/{split}")
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap_ids = split_ids[left] & split_ids[right]
            overlap_queries = split_queries[left] & split_queries[right]
            overlap_scenarios = split_scenarios[left] & split_scenarios[right]
            if overlap_ids or overlap_queries or overlap_scenarios:
                raise ValueError(
                    f"{language} leakage between {left}/{right}: "
                    f"ids={len(overlap_ids)}, queries={len(overlap_queries)}, "
                    f"scenarios={len(overlap_scenarios)}"
                )
    for split in ("train", "val", "test"):
        en_ids = {record["id"] for record in english_splits[split]}
        vi_ids = {record["id"] for record in vietnamese_splits[split]}
        if en_ids != vi_ids:
            raise ValueError(f"EN/VI IDs differ in {split}: en={len(en_ids)}, vi={len(vi_ids)}")


def _counts(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    source = Counter(record["source"] for record in rows)
    label = Counter("positive" if record["function_calls"] else "negative" for record in rows)
    calls = Counter(len(record["function_calls"]) for record in rows)
    return {
        "total": len(rows),
        "source": dict(sorted(source.items())),
        "label": dict(sorted(label.items())),
        "calls_per_sample": dict(sorted((str(key), value) for key, value in calls.items())),
    }


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.endswith(".tmp") and path.name != "manifest.json"
    }


def _copy_active_export(revision_dir: Path, active_dir: Path, force: bool) -> dict[str, str]:
    if active_dir.exists():
        if not force:
            raise FileExistsError(f"Active benchmark already exists: {active_dir}")
        shutil.rmtree(active_dir)
    staging = active_dir.with_name(active_dir.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        shutil.copy2(revision_dir / "vi" / f"{split}.jsonl", staging / f"{split}.jsonl")
    shutil.copy2(revision_dir / "tool_pool.json", staging / "tool_pool.json")
    shutil.copytree(revision_dir / "tool_schema", staging / "tool_schema")
    shutil.copy2(revision_dir / "manifest.json", staging / "manifest.json")
    shutil.copy2(revision_dir / "split_manifest.json", staging / "split_manifest.json")
    shutil.copy2(revision_dir / "metadata.json", staging / "metadata.json")
    shutil.copytree(revision_dir / ".cache", staging / ".cache")
    os.replace(staging, active_dir)
    return {
        str(path.relative_to(active_dir)): _sha256(path)
        for path in sorted(active_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def build_revision(config: RevisionBuildConfig) -> dict[str, Any]:
    """Build a revision in staging, validate it, and optionally promote active VI data."""
    if config.revision.startswith(".") or "/" in config.revision or "\\" in config.revision:
        raise ValueError("revision must be a simple directory name")
    revision_dir = config.benchmark_root / config.revision
    if revision_dir.exists():
        if not config.overwrite_revision:
            raise FileExistsError(f"Revision already exists: {revision_dir}")
        shutil.rmtree(revision_dir)
    staging = config.benchmark_root / f".{config.revision}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    config.benchmark_root.mkdir(parents=True, exist_ok=True)

    en_paths = [
        (config.input_en_glaive, "glaive"),
        (config.input_en_glaive_negative, "glaive"),
        (config.input_en_xlam, "xlam"),
    ]
    vi_paths = [
        (config.input_vi_glaive, "glaive"),
        (config.input_vi_glaive_negative, "glaive"),
        (config.input_vi_xlam, "xlam"),
    ]
    english, en_input_counts = _load_source_map(en_paths)
    vietnamese, vi_input_counts = _load_source_map(vi_paths)
    pairs = _pair_records(english, vietnamese)
    valid_pairs, rejected_records = _filter_valid_pairs(pairs)
    unique_pairs, duplicate_counts = _deduplicate_pairs(valid_pairs)
    assignments = _split_pairs(
        unique_pairs,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
        config.seed,
    )
    english_splits, vietnamese_splits = _split_records(unique_pairs, assignments)

    cache: dict[str, str] = {}
    if config.feature_group_cache is not None:
        cache = load_cache(config.feature_group_cache)
    all_tool_names = {
        tool["name"]
        for _english, vietnamese_record in unique_pairs
        for tool in vietnamese_record["tools"]
    }
    cache = {name: cache.get(name, "Khác") for name in sorted(all_tool_names)}
    for split in ("train", "val", "test"):
        english_splits[split] = _attach_feature_groups(english_splits[split], cache)
        vietnamese_splits[split] = _attach_feature_groups(vietnamese_splits[split], cache)
    _validate_splits(english_splits, vietnamese_splits)

    vi_records = [record for split in ("train", "val", "test") for record in vietnamese_splits[split]]
    pool = _build_tool_pool(vi_records, cache)
    tool_names = {tool["name"] for tool in pool}
    if tool_names != all_tool_names:
        raise ValueError(f"Tool pool coverage mismatch: pool={len(tool_names)}, records={len(all_tool_names)}")

    staging.mkdir(parents=True, exist_ok=False)
    for language, splits in (("en", english_splits), ("vi", vietnamese_splits)):
        for split, records in splits.items():
            _write_jsonl(staging / language / f"{split}.jsonl", records)
    _write_json(staging / "tool_pool.json", pool)
    for tool in pool:
        _write_json(staging / "tool_schema" / f"{tool['name']}.json", tool)
    _write_json(staging / ".cache" / "feature_group.json", cache)
    _write_json(staging / "split_manifest.json", assignments)
    _write_jsonl(staging / "rejected_records.jsonl", rejected_records)

    all_en = [record for split in ("train", "val", "test") for record in english_splits[split]]
    all_vi = [record for split in ("train", "val", "test") for record in vietnamese_splits[split]]
    metadata = {
        "schema_version": "benchmark-core-v1",
        "revision": config.revision,
        "language": "vi",
        "counts": {"en": _counts(all_en), "vi": _counts(all_vi)},
        "split_counts": {
            language: {split: len(splits[split]) for split in ("train", "val", "test")}
            for language, splits in (("en", english_splits), ("vi", vietnamese_splits))
        },
        "n_unique_tools": len(pool),
        "feature_group_counts": dict(
            sorted(Counter(tool["feature_group"] for tool in pool).items())
        ),
        "input_counts": {"en": en_input_counts, "vi": vi_input_counts},
        "deduplication": {
            "input_pairs": len(pairs),
            "rejected_pairs": len(rejected_records),
            "output_pairs": len(unique_pairs),
            "removed": sum(duplicate_counts.values()),
            "reasons": duplicate_counts,
            "query_normalization": "NFKC + casefold + whitespace collapse",
        },
        "rejected_records": {
            "path": "rejected_records.jsonl",
            "count": len(rejected_records),
        },
    }
    _write_json(staging / "metadata.json", metadata)

    input_counts = {**en_input_counts, **vi_input_counts}
    source_cache = load_cache(config.feature_group_cache) if config.feature_group_cache else {}
    manifest = {
        "schema_version": "benchmark-core-v1",
        "revision": config.revision,
        "builder": {
            "module": "src.data.rebuild_benchmark",
            "git_revision": _git_revision(),
        },
        "inputs": {
            str(path): {"sha256": _sha256(path), "records": input_counts[str(path)]["records"]}
            for path in [
                config.input_en_glaive,
                config.input_en_glaive_negative,
                config.input_en_xlam,
                config.input_vi_glaive,
                config.input_vi_glaive_negative,
                config.input_vi_xlam,
            ]
        },
        "split": {
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "seed": config.seed,
            "strategy": "stratified by source and positive/negative label; paired IDs stay together",
        },
        "deduplication": metadata["deduplication"],
        "counts": metadata["counts"],
        "split_counts": metadata["split_counts"],
        "n_unique_tools": len(pool),
        "feature_group_cache": {
            "input": str(config.feature_group_cache) if config.feature_group_cache else None,
            "cached_or_fallback": {
                "cached": sum(1 for name in all_tool_names if name in source_cache),
                "fallback": sum(1 for name in all_tool_names if name not in source_cache),
            },
        },
        "feature_group_config": str(config.feature_group_config),
        "split_manifest_sha256": _sha256(staging / "split_manifest.json"),
        "files": {},
    }
    manifest["files"] = _file_hashes(staging)
    _write_json(staging / "manifest.json", manifest)
    os.replace(staging, revision_dir)

    if config.promote:
        active_hashes = _copy_active_export(revision_dir, config.active_dir, config.force_active)
        manifest["active_export"] = {
            "path": str(config.active_dir),
            "files": active_hashes,
        }
        _write_json(revision_dir / "manifest.json", manifest)
        shutil.copy2(revision_dir / "manifest.json", config.active_dir / "manifest.json")
    print(
        f"[rebuild] revision={config.revision} pairs={len(unique_pairs)} "
        f"train={len(english_splits['train'])} val={len(english_splits['val'])} "
        f"test={len(english_splits['test'])} tools={len(pool)}"
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/benchmark.yaml")
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--active-dir", type=Path, default=None)
    parser.add_argument("--feature-group-cache", type=Path, default=None)
    parser.add_argument("--feature-group-config", type=Path, default=None)
    promote = parser.add_mutually_exclusive_group()
    promote.add_argument("--promote", dest="promote", action="store_true")
    promote.add_argument("--no-promote", dest="promote", action="store_false")
    parser.set_defaults(promote=None)
    parser.add_argument("--overwrite-revision", action="store_true", default=None)
    parser.add_argument("--force-active", action="store_true", default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-en-glaive", type=Path, default=None)
    parser.add_argument("--input-en-glaive-negative", type=Path, default=None)
    parser.add_argument("--input-en-xlam", type=Path, default=None)
    parser.add_argument("--input-vi-glaive", type=Path, default=None)
    parser.add_argument("--input-vi-glaive-negative", type=Path, default=None)
    parser.add_argument("--input-vi-xlam", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    values: dict[str, Any] = {}
    if args.config.exists():
        import yaml

        with args.config.open("r", encoding="utf-8") as source:
            loaded = yaml.safe_load(source)
        if isinstance(loaded, dict):
            values = loaded
    config = RevisionBuildConfig.from_dict(values)
    cli_values = {
        "revision": args.revision,
        "benchmark_root": args.benchmark_root,
        "active_dir": args.active_dir,
        "feature_group_cache": args.feature_group_cache,
        "feature_group_config": args.feature_group_config,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "promote": args.promote,
        "overwrite_revision": args.overwrite_revision,
        "force_active": args.force_active,
        "input_en_glaive": args.input_en_glaive,
        "input_en_glaive_negative": args.input_en_glaive_negative,
        "input_en_xlam": args.input_en_xlam,
        "input_vi_glaive": args.input_vi_glaive,
        "input_vi_glaive_negative": args.input_vi_glaive_negative,
        "input_vi_xlam": args.input_vi_xlam,
    }
    replace_values = {key: value for key, value in cli_values.items() if value is not None}
    config = replace(config, **replace_values)
    build_revision(config)


if __name__ == "__main__":
    main()
