"""Writers for machine-readable and thesis-friendly evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summary_markdown(report: dict[str, Any]) -> str:
    detection = report["metrics"]["detection"]
    retrieval = report["metrics"]["retrieval"]
    selection = report["metrics"]["selection"]
    extraction = report["metrics"]["extraction"]
    schema = report["metrics"]["schema_validity"]
    end_to_end = report["metrics"]["end_to_end"]
    efficiency = report["metrics"]["efficiency"]
    oracle_extraction = report["metrics"]["oracle_extraction"]
    rows = [
        ("Call F1", detection["f1"]),
        ("Recall@5", retrieval.get("recall_at_5")),
        ("Tool Set Accuracy", selection["tool_set_accuracy_positive"]),
        ("Normalized ArgEM given tool", extraction["normalized_arg_em_given_correct_tool"]),
        (
            "Oracle-tool ArgEM",
            oracle_extraction["normalized_arg_em_given_correct_tool"]
            if oracle_extraction is not None
            else None,
        ),
        ("Argument Pair F1", extraction["argument_pair"]["f1"]),
        ("Schema Validity", schema["call_schema_validity"]),
        ("N-FCEM-positive", end_to_end["n_fcem_positive"]),
        ("Overall Success", end_to_end["overall_success"]),
        ("p95 latency (ms)", efficiency["latency_ms"]["p95"]),
        ("Cost/correct call (USD)", efficiency["cost"]["usd_per_correct_call"]),
    ]
    lines = [
        "# Evaluation Summary",
        "",
        f"Samples: {report['dataset']['sample_count']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {_format(value)} |" for name, value in rows)
    lines.extend(["", "## Output quality", ""])
    lines.append(
        f"Prediction parse coverage: {_format(schema['prediction_parse_validity'])}; "
        f"retrieval coverage: {_format(retrieval['coverage'])}."
    )
    return "\n".join(lines) + "\n"


def write_evaluation_outputs(
    output_dir: Path,
    report: dict[str, Any],
    per_sample: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with (output_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_sample:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
