"""Test cổng fail-closed trước training.

Điểm mấu chốt: thiếu artefact hoặc SHA-256 lệch thì **dừng**, không được âm thầm
build lại. Nếu job training tự dựng lại index từ dữ liệu đang có trên máy thì ta
mất đúng thứ cần đảm bảo — bằng chứng model train trên đúng split đã kiểm định.
"""

import json

import pytest

from src.models.preflight import (
    PreflightReport,
    check_artifacts_match_manifest,
    check_split_integrity,
    check_two_stages_agree,
    check_versions,
)
from src.models.sources import (
    MissingDecontaminationIndex,
    load_decontamination,
    load_or_build_decontamination,
    sha256_file,
)


@pytest.fixture
def workspace(tmp_path):
    """Tạo bộ artefact hợp lệ tối thiểu."""
    decon = tmp_path / "decontamination.json"
    decon.write_text(json.dumps({"effective_split": {"a": "test"}, "stats": {}}), encoding="utf-8")

    files = {"data/method2/decontamination.json": decon}
    for name in (
        "data/method2/tool_pool.json",
        "data/method2/biencoder/train.jsonl",
        "data/method2/crossencoder/train.jsonl",
    ):
        path = tmp_path / name.replace("/", "_")
        path.write_text("{}", encoding="utf-8")
        files[name] = path

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "derived": {
                    name: {"present": True, "sha256": sha256_file(path)}
                    for name, path in files.items()
                }
            }
        ),
        encoding="utf-8",
    )
    return {"tmp": tmp_path, "manifest": manifest, "decon": decon, "files": files}


def _run_artifacts_check(workspace, monkeypatch) -> PreflightReport:
    import src.models.preflight as preflight

    monkeypatch.setattr(
        preflight, "REQUIRED_ARTIFACTS", tuple(workspace["files"]), raising=False
    )
    monkeypatch.chdir(workspace["tmp"])
    # REQUIRED_ARTIFACTS là đường dẫn tương đối; ánh xạ về file thật trong tmp.
    for name, path in workspace["files"].items():
        target = workspace["tmp"] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    report = PreflightReport()
    check_artifacts_match_manifest(
        report, workspace["manifest"], workspace["tmp"] / "data/method2/decontamination.json"
    )
    return report


def test_passes_when_every_artifact_matches(workspace, monkeypatch):
    report = _run_artifacts_check(workspace, monkeypatch)

    assert report.passed, report.failures


def test_fails_when_decontamination_missing(workspace, monkeypatch):
    report = _run_artifacts_check(workspace, monkeypatch)
    assert report.passed

    (workspace["tmp"] / "data/method2/decontamination.json").unlink()
    report2 = PreflightReport()
    ok = check_artifacts_match_manifest(
        report2, workspace["manifest"], workspace["tmp"] / "data/method2/decontamination.json"
    )

    assert not ok
    assert "decontamination.json tồn tại" in report2.failures


def test_fails_when_sha256_drifts(workspace, monkeypatch):
    _run_artifacts_check(workspace, monkeypatch)
    target = workspace["tmp"] / "data/method2/tool_pool.json"
    target.write_text('{"changed": true}', encoding="utf-8")

    report = PreflightReport()
    ok = check_artifacts_match_manifest(
        report, workspace["manifest"], workspace["tmp"] / "data/method2/decontamination.json"
    )

    assert not ok
    assert any("tool_pool" in name for name in report.failures)


def test_load_decontamination_is_fail_closed(tmp_path):
    with pytest.raises(MissingDecontaminationIndex, match="decontaminate"):
        load_decontamination(tmp_path / "nope.json")


def test_load_or_build_refuses_to_build_by_default(tmp_path):
    with pytest.raises(MissingDecontaminationIndex):
        load_or_build_decontamination(tmp_path / "nope.json", (), None)


def test_load_or_build_builds_only_when_explicitly_allowed(tmp_path):
    path = tmp_path / "decon.json"

    index = load_or_build_decontamination(path, (), None, allow_build=True)

    assert path.exists()
    assert index.effective_split == {}


def _stats(overlap: dict, leaked: list | None = None, overlapping=None) -> dict:
    return {
        "split_overlap_after": overlap,
        "unseen_tools_leaked_into_train_positives": leaked if leaked is not None else [],
        "decontamination": {"overlapping_queries": overlapping or {"test∩val": 30}},
    }


def test_split_integrity_detects_residual_overlap(tmp_path):
    clean = tmp_path / "pairs.json"
    dirty = tmp_path / "label.json"
    clean.write_text(json.dumps(_stats({"test∩val": 0, "test∩train": 0, "train∩val": 0})), encoding="utf-8")
    dirty.write_text(json.dumps(_stats({"test∩val": 7, "test∩train": 0, "train∩val": 0})), encoding="utf-8")

    report = PreflightReport()
    ok = check_split_integrity(report, clean, dirty)

    assert not ok
    assert "overlap Cross-Encoder == 0" in report.failures


def test_split_integrity_detects_unseen_tool_leakage(tmp_path):
    path = tmp_path / "pairs.json"
    zeros = {"test∩val": 0, "test∩train": 0, "train∩val": 0}
    path.write_text(json.dumps(_stats(zeros, leaked=["vi_unseen_tool"])), encoding="utf-8")

    report = PreflightReport()
    ok = check_split_integrity(report, path, path)

    assert not ok
    assert "unseen positive leakage == 0" in report.failures


def test_two_stages_must_share_the_same_index(tmp_path):
    bi = tmp_path / "bi.json"
    cross = tmp_path / "cross.json"
    zeros = {"test∩val": 0, "test∩train": 0, "train∩val": 0}
    bi.write_text(json.dumps(_stats(zeros, overlapping={"test∩val": 30})), encoding="utf-8")
    cross.write_text(json.dumps(_stats(zeros, overlapping={"test∩val": 999})), encoding="utf-8")

    report = PreflightReport()

    assert not check_two_stages_agree(report, bi, cross)
    assert "hai stage dùng chung index" in report.failures


def test_version_mismatch_fails_in_strict_mode(tmp_path):
    pinned = tmp_path / "pinned.json"
    pinned.write_text(json.dumps({"pinned": {"transformers": "0.0.0-không-tồn-tại"}}), encoding="utf-8")

    strict = PreflightReport()
    lenient = PreflightReport()

    assert not check_versions(strict, pinned, strict=True)
    assert check_versions(lenient, pinned, strict=False)
