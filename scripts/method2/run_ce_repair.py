"""Train repaired CE supervision and select a checkpoint using validation only."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.sources import sha256_file


def verify_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in data["files"].items():
        if not Path(name).is_file() or sha256_file(Path(name)) != digest:
            raise ValueError(f"Missing or modified artifact: {name}")
    return data


def selection_score(gate: dict[str, Any]) -> tuple[float, float]:
    argument_em = gate["inference_diagnostics"]["argument_em"]
    enum_accuracy = gate["components"]["by_source"]["custom_vi"]["enum_accuracy"]
    if argument_em is None:
        raise ValueError("Validation Argument EM must be measured before checkpoint selection")
    return float(argument_em), float(enum_accuracy)


def run() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CE repair training requires a CUDA session")
    verify_manifest(Path("completion_manifest.json"))
    patch = verify_manifest(Path("ce_repair_manifest.json"))
    if patch["parent_completion_sha256"] != sha256_file(Path("completion_manifest.json")):
        raise ValueError("CE repair must use its recorded parent bundle")
    root = Path("results/method2/ce_repair")
    candidate = Path("artifacts/method2/crossencoder/repair_v1/final")
    baseline = Path("artifacts/method2/crossencoder/baseline_for_repair/final")
    selected = Path("artifacts/method2/crossencoder/run01/final")
    marker = Path("results/method2/completion/validation_completed.json")
    if root.exists() or candidate.parent.exists() or selected.exists() or marker.exists():
        raise FileExistsError("Use a fresh working session; preserve all earlier CE results")
    config = "configs/method2/ce_repair_v1.yaml"

    def command(*args: str) -> None:
        subprocess.run([sys.executable, *args], check=True)

    def validate(model: Path, output: Path) -> dict[str, Any]:
        command("scripts/method2/validation_gate.py", "--config", config,
                "--model", str(model), "--output", str(output))
        return json.loads((output / "validation_gate.json").read_text(encoding="utf-8"))

    old_gate = validate(baseline, root / "baseline")
    command("-m", "src.models.crossencoder.train", "--config", config)
    new_gate = validate(candidate, root / "candidate")
    winner = "candidate" if selection_score(new_gate) > selection_score(old_gate) else "baseline"
    chosen = candidate if winner == "candidate" else baseline
    shutil.copytree(chosen, selected)
    selected_gate = validate(selected, marker.parent / "validation")
    comparison = {"selection_rule": "validation argument_em, then enum accuracy; baseline on ties",
                  "winner": winner, "baseline_score": selection_score(old_gate),
                  "candidate_score": selection_score(new_gate), "baseline_gate_passed": old_gate["passed"],
                  "candidate_gate_passed": new_gate["passed"], "selected_gate_passed": selected_gate["passed"],
                  "selected_source": str(chosen), "selected_checkpoint": str(selected),
                  "test_used_for_selection": False,
                  "repair_manifest_sha256": sha256_file(Path("ce_repair_manifest.json")),
                  "next_action": "review_validation_results_before_notebook04"}
    (root / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    hashes = {p.as_posix(): sha256_file(p) for p in selected.rglob("*") if p.is_file()}
    marker.write_text(json.dumps({"stage": "validation", "completed": True,
                      "bundle_manifest_sha256": sha256_file(Path("completion_manifest.json")),
                      "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "checkpoint_hashes": hashes,
                      "results": {"validation_gate": selected_gate, "ce_repair": comparison}},
                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
