"""Build call-specific CE labels from checked dataset mention annotations."""

from __future__ import annotations

from typing import Any

from src.models.crossencoder.data_collator import iter_parameters
from src.models.crossencoder.label_generator import LabelGenerator, LabelGeneratorConfig, SkipLabel
from src.models.crossencoder.normalize import SpanNormalizer


def annotated_pairs(sample: dict[str, Any], split: str, source_key: str) -> list[dict[str, Any]]:
    query = sample["query"]
    mentions: dict[tuple[int, str], dict[str, Any]] = {}
    for mention in sample.get("metadata", {}).get("argument_mentions", []):
        key = (mention["call_index"], mention["parameter_path"])
        if key in mentions:
            raise ValueError(f"Duplicate mention: {sample['id']} {key}")
        mentions[key] = mention
    schemas = {tool["name"]: tool for tool in sample["tools"]}
    generator = LabelGenerator(LabelGeneratorConfig(tokenizer_name=""))
    rows = []
    for call_index, call in enumerate(sample["function_calls"]):
        for param in iter_parameters(schemas[call["name"]]):
            value = call["arguments"].get(param["name"])
            is_span = param["routing_type"] in ("string", "number")
            mention = mentions.get((call_index, param["name"]))
            if value is not None and is_span:
                if mention is None:
                    raise ValueError(f"Missing mention: {sample['id']} {call_index} {param['name']}")
                start, end = mention["start"], mention["end"]
                if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(query)
                        or query[start:end] != mention["surface"]
                        or type(mention["canonical_value"]) is not type(value)
                        or mention["canonical_value"] != value):
                    raise ValueError(f"Invalid mention: {sample['id']} {call_index} {param['name']}")
                labels = {"has_value": 1, "span_start": 0, "span_end": 0,
                          "enum_label": 0, "boolean_label": 0,
                          "schema_type": param["routing_type"], "char_start": start, "char_end": end}
            else:
                labels = generator.generate(query, param, value, param["required"])
                if isinstance(labels, SkipLabel):
                    raise ValueError(f"Unsupported CustomTools label: {sample['id']} {param['name']} {labels.reason}")
            row = {"sample_id": sample["id"], "call_index": call_index, "query": query,
                   "tool_name": call["name"], "param": param, "labels": labels,
                   "source": "custom_vi", "source_key": source_key, "split": split,
                   "tool_split": sample["metadata"]["tool_split"],
                   "label_origin": "argument_mentions" if value is not None and is_span else "schema_and_gold"}
            if value is not None and is_span:
                norm = SpanNormalizer().normalize(mention["surface"], param)
                row["surface_normalizes_to_gold"] = norm.ok and norm.value == value
            rows.append(row)
    return rows
