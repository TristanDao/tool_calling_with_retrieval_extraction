"""CE repair selects on validation and retains checkpoint provenance without GPU training."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.models.sources import sha256_file


def _module():
    spec = importlib.util.spec_from_file_location("ce_repair_runner", "scripts/method2/run_ce_repair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("candidate_score,expected", [(0.6, "candidate"), (0.4, "baseline"), (0.5, "baseline")])
def test_validation_selection_and_marker(tmp_path: Path, monkeypatch, candidate_score: float, expected: str) -> None:
    module = _module()
    monkeypatch.chdir(tmp_path)
    baseline = Path("artifacts/method2/crossencoder/baseline_for_repair/final")
    baseline.mkdir(parents=True)
    (baseline / "model.safetensors").write_bytes(b"baseline")
    Path("completion_manifest.json").write_text(json.dumps({"files": {}}))
    Path("ce_repair_manifest.json").write_text(json.dumps({
        "files": {}, "parent_completion_sha256": sha256_file(Path("completion_manifest.json"))}))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda index: "mock GPU")
    validated = []

    def execute(args, check):
        if "src.models.crossencoder.train" in args:
            final = Path("artifacts/method2/crossencoder/repair_v1/final")
            final.mkdir(parents=True)
            (final / "model.safetensors").write_bytes(b"candidate")
        else:
            model = Path(args[args.index("--model") + 1])
            output = Path(args[args.index("--output") + 1])
            validated.append(model)
            output.mkdir(parents=True)
            score = candidate_score if (model / "model.safetensors").read_bytes() == b"candidate" else 0.5
            gate = {"passed": False, "checkpoint": str(model), "inference_diagnostics": {"argument_em": score},
                    "components": {"by_source": {"custom_vi": {"enum_accuracy": 0.8}}}}
            (output / "validation_gate.json").write_text(json.dumps(gate))

    monkeypatch.setattr(module.subprocess, "run", execute)
    module.run()
    comparison = json.loads(Path("results/method2/ce_repair/comparison.json").read_text())
    assert comparison["winner"] == expected
    assert comparison["test_used_for_selection"] is False
    marker = json.loads(Path("results/method2/completion/validation_completed.json").read_text())
    assert marker["completed"] and not marker["results"]["validation_gate"]["passed"]
    assert len(validated) == 3
    for name, digest in marker["checkpoint_hashes"].items():
        assert sha256_file(Path(name)) == digest


def test_addon_hash_mismatch_is_rejected(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.chdir(tmp_path)
    Path("manifest.json").write_text(json.dumps({"files": {"missing": "digest"}}))
    with pytest.raises(ValueError, match="Missing or modified artifact"):
        module.verify_manifest(Path("manifest.json"))
