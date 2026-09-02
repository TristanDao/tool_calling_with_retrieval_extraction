"""Tests for the shared native output parser."""

from __future__ import annotations

from src.evaluation.native_output import parse_tool_output


def test_parse_single_native_call():
    output = """<tool_call>
<function=search_tutors>
<parameter=subject>
Toán
</parameter>
<parameter=online>
true
</parameter>
</function>
</tool_call>"""
    parsed = parse_tool_output(output, {"search_tutors"})
    assert parsed.errors == []
    assert parsed.calls[0].name == "search_tutors"
    assert parsed.calls[0].arguments == {"subject": "Toán", "online": True}


def test_parse_multiple_native_calls_and_json_values():
    output = """<tool_call><function=first><parameter=items>[1, 2]</parameter></function></tool_call>
<tool_call><function=second><parameter=enabled>false</parameter></function></tool_call>"""
    parsed = parse_tool_output(output)
    assert [call.name for call in parsed.calls] == ["first", "second"]
    assert parsed.calls[0].arguments["items"] == [1, 2]
    assert parsed.calls[1].arguments["enabled"] is False


def test_parse_negative_output():
    parsed = parse_tool_output("Tôi chưa thể thực hiện yêu cầu này.")
    assert parsed.is_negative
    assert parsed.errors == []


def test_parse_json_fallback_and_unknown_tool():
    parsed = parse_tool_output('{"name":"lookup","arguments":{"x":1}}', {"other"})
    assert parsed.calls[0].arguments == {"x": 1}
    assert parsed.errors == ["unknown tool: lookup"]


def test_malformed_native_output_is_not_counted_as_negative():
    parsed = parse_tool_output("<tool_call><function=lookup>")
    assert not parsed.is_negative
    assert "unbalanced native tool_call tags" in parsed.errors


def test_parse_json_code_fence():
    parsed = parse_tool_output("```json\n[{\"name\":\"lookup\",\"arguments\":{}}]\n```")
    assert [call.name for call in parsed.calls] == ["lookup"]
