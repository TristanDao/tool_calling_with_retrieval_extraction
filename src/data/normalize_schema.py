"""Normalize xLAM/Glaive parameter schema → JSON Schema chuẩn.

Mapping:
- str / string                → "string"
- int / integer               → "integer"
- float / number              → "number"
- bool / boolean              → "boolean"
- list / List[T] / List[...]  → "array" + items: {type: T}
- ", optional" suffix         → tách thành top-level required: []
- Dict / object               → "object"

Hoạt động trên raw xLAM tools (parameters là dict[str, param_schema])
và Glaive system JSON (parameters.properties.{name}.{type,description}).
"""

from __future__ import annotations

import re
from typing import Any


_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "enum": "enum",
    "list": "array",
    "array": "array",
    "object": "object",
    "dict": "object",
}

_OPTIONAL_RE = re.compile(r",\s*optional\s*$", re.IGNORECASE)
_LIST_OF_RE = re.compile(r"^List\[\s*([A-Za-z0-9_\[\], ]+?)\s*\]$")
_UNION_RE = re.compile(r"Union\[\s*([^\]]+?)\s*\]")
_TUPLE_RE = re.compile(r"^Tuple\[\s*([^,]+?)\s*,\s*([^,]+?)\s*\]$")
_LIST_OF_LIST_RE = re.compile(r"^List\[\s*List\[\s*([A-Za-z0-9_]+)\s*\]\s*\]$")
_LIST_OF_TUPLE_RE = re.compile(r"^List\[\s*Tuple\[\s*([^,]+?)\s*,\s*([^,]+?)\s*\]\s*\]$")

_PRIMITIVE_TYPES = {"string", "integer", "number", "boolean"}


def _strip_optional(type_str: str) -> tuple[str, bool]:
    if not type_str:
        return "", False
    s = _OPTIONAL_RE.sub("", str(type_str).strip())
    return s, s != str(type_str).strip()


def _map_simple_type(type_str: str) -> str | None:
    if not type_str:
        return None
    cleaned = type_str.strip().lower()
    return _TYPE_MAP.get(cleaned)


def _parse_items_type(type_str: str) -> dict[str, Any] | None:
    """Parse `List[T]` → items dict (chỉ phần items, không wrap).

    Returns:
      - {"type": "integer"}     cho List[int]
      - {"type": "number"}      cho List[Union[int, float]]
      - {"type": "array", "items": {"type": "integer"}}  cho List[List[int]]
      - {"type": "array"}       cho List[Tuple[...]]
    Caller tự wrap thành {"type": "array", "items": <result>}.
    """
    if not type_str:
        return None
    s = type_str.strip()

    m = _LIST_OF_LIST_RE.match(s)
    if m:
        inner = _map_simple_type(m.group(1))
        if inner:
            return {"type": "array", "items": {"type": inner}}

    m = _LIST_OF_TUPLE_RE.match(s)
    if m:
        return {"type": "array"}

    m = _LIST_OF_RE.match(s)
    if m:
        inner_raw = m.group(1).strip()
        if _UNION_RE.match(inner_raw):
            um = _UNION_RE.match(inner_raw)
            inner_types = [t.strip() for t in um.group(1).split(",")]
            mapped = [_map_simple_type(t) for t in inner_types]
            valid = [m for m in mapped if m in _PRIMITIVE_TYPES]
            if valid:
                if "number" in valid and "integer" in valid:
                    valid = ["number"]
                return {"type": valid[0]}
        else:
            inner = _map_simple_type(inner_raw)
            if inner:
                return {"type": inner}

    return None


def normalize_type(raw_type: str) -> tuple[str, bool]:
    """Map raw type string to (standard_type, is_optional).

    Examples:
      "str"                  -> ("string", False)
      "int, optional"        -> ("integer", True)
      "List[int]"            -> ("array", False)
      "List[Union[int, float]]" -> ("array", False)
    """
    if not raw_type:
        return "string", False
    cleaned, is_optional = _strip_optional(str(raw_type))
    mapped = _map_simple_type(cleaned)
    if mapped:
        return mapped, is_optional
    if _LIST_OF_RE.match(cleaned) or cleaned.lower() == "list":
        return "array", is_optional
    return "string", is_optional


def normalize_parameter_schema(
    pname: str,
    pschema: dict[str, Any],
    description: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Normalize 1 parameter schema → (normalized_schema, is_optional).

    Output schema (xLAM style — flat per param):
      {
        "type": "string" | "integer" | "number" | "boolean" | "array" | "object" | "enum",
        "description": "...",
        ...(items nếu array, enum nếu enum, default nếu có)...
      }
    """
    raw_type = pschema.get("type", "string")
    standard_type, is_optional = normalize_type(raw_type)

    out: dict[str, Any] = {"type": standard_type}

    desc = description if description is not None else pschema.get("description", "")
    if desc:
        out["description"] = str(desc).strip()

    if "enum" in pschema and pschema["enum"]:
        enum_vals = pschema["enum"]
        if isinstance(enum_vals, list) and all(isinstance(v, (str, int, float, bool)) for v in enum_vals):
            out["enum"] = list(enum_vals)
            out["type"] = "enum"

    if "default" in pschema and pschema["default"] not in (None, ""):
        out["default"] = pschema["default"]

    if standard_type == "array":
        items = pschema.get("items")
        if items is None:
            items = _parse_items_type(raw_type)
        if items is not None and isinstance(items, dict):
            out["items"] = items
        else:
            out["items"] = {"type": "string"}

    return out, is_optional


def normalize_xlam_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Normalize 1 xLAM tool → standard schema.

    Input xLAM:
      {
        "name": "foo",
        "description": "...",
        "parameters": {
          "param1": {"type": "str", "description": "..."},
          "param2": {"type": "int, optional", "description": "..."}
        }
      }

    Output:
      {
        "name": "foo",
        "description": "...",
        "parameters": {
          "type": "object",
          "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "integer", "description": "..."}
          },
          "required": ["param1"]
        }
      }
    """
    raw_params = tool.get("parameters", {}) or {}
    if not isinstance(raw_params, dict):
        raw_params = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for pname, pschema in raw_params.items():
        if not isinstance(pschema, dict):
            pschema = {"type": "string"}
        normalized, is_optional = normalize_parameter_schema(pname, pschema)
        properties[pname] = normalized
        if not is_optional:
            required.append(pname)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": parameters,
    }


def normalize_glaive_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Normalize 1 Glaive tool từ system JSON.

    Glaive format:
      {
        "name": "foo",
        "description": "...",
        "parameters": {
          "type": "object",
          "properties": {
            "param1": {"type": "string", "description": "..."}
          },
          "required": ["param1"]
        }
      }
    → Trả về normalized (giữ nguyên vì đã đúng JSON Schema chuẩn).
    """
    return {
        "name": tool.get("name", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
    }


def normalize_tool(tool: dict[str, Any], dataset: str = "auto") -> dict[str, Any]:
    """Auto-detect format (xLAM flat or Glaive nested) và normalize.

    Heuristic: nếu `parameters` có key "properties" → Glaive format (đã chuẩn).
               nếu `parameters` không có "properties" → xLAM format (flat).
    """
    params = tool.get("parameters", {}) or {}
    if not isinstance(params, dict):
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": {}},
        }
    if "properties" in params:
        return normalize_glaive_tool(tool)
    return normalize_xlam_tool(tool)


def is_standard_type(t: str) -> bool:
    return t in {"string", "integer", "number", "boolean", "array", "object", "enum"}
