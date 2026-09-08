"""Completion notebooks preserve the full bundle and explicit stage dependencies."""

import json
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from src.models.sources import sha256_file

spec = importlib.util.spec_from_file_location("completion_generator", "scripts/method2/make_completion_notebooks.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
SETUP, STAGES, make_notebooks = generator.SETUP, generator.STAGES, generator.make_notebooks


def test_generated_completion_notebooks_match_and_compile(tmp_path: Path) -> None:
    for stage, path in zip(STAGES, make_notebooks(tmp_path), strict=True):
        assert path.read_bytes() == (Path("notebooks") / path.name).read_bytes()
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
        assert stage in "".join(notebook["cells"][3]["source"])


def test_ce_repair_notebook_matches_and_compiles(tmp_path: Path) -> None:
    path = generator.make_ce_repair_notebook(tmp_path)
    assert path.read_bytes() == (Path("notebooks") / path.name).read_bytes()
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")


def test_setup_restores_every_hashed_bundle_file(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "input" / "bundle"
    work = tmp_path / "work"
    work.mkdir()
    for name in ["src", "scripts", "configs", "data", "docs", "notebooks"]:
        (bundle / name).mkdir(parents=True)
        (bundle / name / "example.txt").write_text(name)
    pins = bundle / "configs/method2/pinned_versions.json"
    pins.parent.mkdir()
    pins.write_text('{"pinned": {}}')
    (bundle / "same_domain_feasibility.json").write_text("{}")
    manifest = {"files": {p.relative_to(bundle).as_posix(): sha256_file(p)
                          for p in bundle.rglob("*") if p.is_file()}}
    (bundle / "completion_manifest.json").write_text(json.dumps(manifest))
    source = SETUP.replace("/kaggle/input", (tmp_path / "input").as_posix()).replace("/kaggle/working", work.as_posix())
    monkeypatch.chdir(tmp_path)
    with patch("subprocess.run") as run:
        exec(compile(source, "setup", "exec"), {})
    commands = [call.args[0] for call in run.call_args_list]
    assert commands[0][1:4] == ["-m", "pip", "install"]
    assert commands[1][1:] == ["-m", "pip", "uninstall", "-y", "torchao"]
    assert commands[2][1] == "-c"
    assert "model.add_adapter" in commands[2][2]
    assert "backward()" in commands[2][2]
    compile(commands[2][2], "lora_smoke", "exec")
    for name, digest in manifest["files"].items():
        assert sha256_file(work / name) == digest


def _helpers() -> dict:
    namespace = {}
    exec(compile(generator.INPUT_HELPERS, "input_helpers", "exec"), namespace)
    return namespace


def _stage(root: Path, weights: bytes = b"adapter", include_weights: bool = True) -> Path:
    import hashlib

    relative = "artifacts/method2/biencoder/strict_round1/final/adapter_model.safetensors"
    marker = root / "results/method2/completion/round1_completed.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"stage": "round1", "completed": True,
                                  "bundle_manifest_sha256": "bundle",
                                  "checkpoint_hashes": {relative: hashlib.sha256(weights).hexdigest()}}))
    if include_weights:
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes(weights)
    return marker


def test_reports_archive_duplicate_does_not_hide_full_output(tmp_path: Path) -> None:
    expected = _stage(tmp_path / "dataset")
    _stage(tmp_path / "dataset/reports_extracted", include_weights=False)
    assert _helpers()["choose_stage"](tmp_path, "round1", "bundle") == expected


def test_duplicate_identical_checkpoints_are_accepted(tmp_path: Path) -> None:
    first = _stage(tmp_path / "a")
    _stage(tmp_path / "b")
    assert _helpers()["choose_stage"](tmp_path, "round1", "bundle") == first


def test_distinct_checkpoints_are_not_chosen_arbitrarily(tmp_path: Path) -> None:
    _stage(tmp_path / "a")
    _stage(tmp_path / "b", weights=b"another adapter")
    with pytest.raises(RuntimeError, match="one complete round1 checkpoint identity"):
        _helpers()["choose_stage"](tmp_path, "round1", "bundle")


def test_corrupt_or_foreign_checkpoint_cannot_be_selected(tmp_path: Path) -> None:
    marker = _stage(tmp_path / "a")
    helpers = _helpers()
    with pytest.raises(RuntimeError):
        helpers["choose_stage"](tmp_path, "round1", "another bundle")
    metadata = json.loads(marker.read_text())
    (marker.parents[3] / next(iter(metadata["checkpoint_hashes"]))).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError):
        helpers["choose_stage"](tmp_path, "round1", "bundle")


def test_duplicate_bundles_require_same_manifest(tmp_path: Path) -> None:
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        (root / "file").write_text("content")
        (root / "completion_manifest.json").write_text(json.dumps({"files": {"file": "digest"}}))
    helpers = _helpers()
    assert helpers["choose_bundle"](tmp_path) == tmp_path / "a"
    (tmp_path / "b/completion_manifest.json").write_text(json.dumps({"files": {"file": "changed"}}))
    with pytest.raises(RuntimeError):
        helpers["choose_bundle"](tmp_path)
