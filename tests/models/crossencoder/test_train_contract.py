"""Ràng buộc trên source của `crossencoder/train.py` — chạy được khi không có torch.

Ba lỗi dưới đây chỉ lộ ra sau nhiều giờ GPU trên Kaggle (lúc session đứt và
phải resume), nên phải bắt được từ máy local nơi không cài torch. Đọc file dạng
text thay vì import module.
"""

from pathlib import Path

import pytest

TRAIN = Path("src/models/crossencoder/train.py")
EVALUATE = Path("src/models/crossencoder/evaluate.py")


@pytest.fixture(scope="module")
def source() -> str:
    return TRAIN.read_text(encoding="utf-8")


def test_resume_rebuilds_model_from_checkpoint(source):
    """Dựng model mới rồi chỉ nạp optimizer state = mất sạch trọng số đã train.

    Optimizer state khi đó khớp với một bộ trọng số không còn tồn tại, nên
    resume còn tệ hơn train lại từ đầu.
    """
    assert "CrossEncoderForExtraction.from_pretrained(resume_path)" in source
    # Phải nạp TRƯỚC vòng lặp stage, nếu không model đã bị dựng lại mất rồi.
    assert source.index("from_pretrained(resume_path)") < source.index(
        "for stage_index, stage in enumerate(stages)"
    )


def test_checkpoint_stores_curriculum_position(source):
    """Không ghi stage_index/epoch thì resume quay về đầu warm-up."""
    assert '"stage_index": stage_index' in source
    assert '"epoch": epoch' in source
    assert "stage_index=stage_index, epoch=epoch" in source


def test_resume_skips_finished_stages_and_epochs(source):
    assert "if stage_index < resume_stage:" in source
    assert "first_epoch = resume_epoch if stage_index == resume_stage else 0" in source
    assert "for epoch in range(first_epoch, stage.epochs):" in source


def test_report_has_every_field_run_manifest_requires(source):
    """`run_manifest._audit_complete` đọc đúng ba khoá này.

    Thiếu là `audit_complete.missing` báo thiếu và assert trong notebook nổ —
    sau khi đã tiêu hết giờ GPU.
    """
    manifest_src = Path("src/models/run_manifest.py").read_text(encoding="utf-8")
    for key in ("final_checkpoint", "training_duration_hours", "checkpoint_selection"):
        assert f'train_report.get("{key}")' in manifest_src, f"run_manifest không đọc {key}"

    for key in ('"final_checkpoint"', '"training_duration_hours"', '"checkpoint_selection"'):
        assert key in source, f"train_report thiếu {key}"


def test_smoke_mode_is_isolated(source):
    """Smoke phải ghi ra thư mục riêng, không đè checkpoint của run dài."""
    assert "config.max_steps = args.smoke" in source
    assert "smoke_run01" in source


def test_find_last_checkpoint_reads_real_step_not_name(source):
    """'checkpoint-1000' < 'checkpoint-500' theo chuỗi; `stage-warmup` không parse được."""
    assert 'output_dir.glob("stage-*")' in source
    assert "progress.json" in source


def test_gate_is_measured_on_custom_vi_not_overall():
    """Plan §Phase 3 chốt gate trên custom val.

    `overall` bị xLAM chi phối (14,452/17,769 cặp val) nên đo ở đó là đo nhầm
    tập — gate có thể đạt trong khi custom_vi trượt.
    """
    source = EVALUATE.read_text(encoding="utf-8")

    assert 'report["by_source"].get("custom_vi")' in source
    assert '"measured_on"' in source
