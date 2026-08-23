"""Test preset smoke run (Run 1) của Bi-Encoder.

Smoke run phải kiểm chứng đúng cấu hình sẽ dùng ở run thật, nên **không được**
đổi các tham số quyết định bộ nhớ và ngữ nghĩa — batch_size, mini_batch_size,
fp16, LoRA, max_seq_length. Đổi những thứ đó thì smoke run không còn trả lời
được câu hỏi "có OOM không, effective batch có đúng 256 không".
"""

import pytest

pytest.importorskip("torch")

from src.models.biencoder.train import (
    BiEncoderTrainConfig,
    apply_smoke_preset,
    select_best_checkpoint,
)


def test_smoke_preserves_memory_and_semantics_settings():
    base = BiEncoderTrainConfig()

    smoke = apply_smoke_preset(base, 200)

    assert smoke.batch_size == base.batch_size == 256
    assert smoke.mini_batch_size == base.mini_batch_size == 8
    assert smoke.fp16 is base.fp16
    assert smoke.gradient_checkpointing is base.gradient_checkpointing
    assert smoke.max_seq_length == base.max_seq_length
    assert smoke.lora == base.lora
    assert smoke.scale == base.scale


def test_smoke_limits_steps_and_thickens_checkpoints():
    smoke = apply_smoke_preset(BiEncoderTrainConfig(), 200)

    assert smoke.max_steps == 200
    assert smoke.epochs == 1
    # Cần ít nhất 2 checkpoint để kiểm tra resume.
    assert smoke.max_steps // smoke.save_steps >= 2
    assert smoke.eval_steps <= smoke.max_steps


def test_smoke_writes_to_a_separate_output_dir():
    base = BiEncoderTrainConfig()

    smoke = apply_smoke_preset(base, 200)

    assert smoke.output_dir != base.output_dir
    assert smoke.output_dir.name.startswith("smoke")
    # Áp lại preset không lồng thêm tiền tố.
    assert apply_smoke_preset(smoke, 200).output_dir == smoke.output_dir


def test_smoke_loads_enough_samples_for_requested_steps():
    smoke = apply_smoke_preset(BiEncoderTrainConfig(), 300)

    assert smoke.max_train_samples >= 300 * smoke.batch_size


@pytest.mark.parametrize("steps", [100, 200, 300])
def test_smoke_range_from_the_plan(steps):
    smoke = apply_smoke_preset(BiEncoderTrainConfig(), steps)

    assert smoke.max_steps == steps
    assert smoke.save_steps >= 10


def test_select_best_checkpoint_resolves_metric_name_across_versions():
    log = [
        {"loss": 1.2, "step": 10},
        {"eval_custom_val_cosine_ndcg@10": 0.71, "step": 100},
        {"eval_custom_val_cosine_ndcg@10": 0.83, "step": 200},
    ]

    result = select_best_checkpoint(log, metric_name=None)

    assert result["resolved_metric"] == "eval_custom_val_cosine_ndcg@10"
    assert result["best_value"] == 0.83
    assert result["best_step"] == 200
    assert "eval_custom_val_cosine_ndcg@10" in result["available_metrics"]


def test_select_best_checkpoint_reports_when_metric_absent():
    log = [{"eval_something_else": 0.5, "step": 10}]

    result = select_best_checkpoint(log, metric_name="eval_không_có")

    assert result["resolved_metric"] is None
    assert result["available_metrics"] == ["eval_something_else"]


def test_select_best_checkpoint_handles_no_evaluation():
    result = select_best_checkpoint([{"loss": 1.0, "step": 1}], metric_name=None)

    assert result["resolved_metric"] is None


def test_checkpoint_step_reads_global_step(tmp_path):
    import json

    from src.models.biencoder.train import checkpoint_step

    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 100}), encoding="utf-8")

    assert checkpoint_step(ckpt) == 100
    assert checkpoint_step(tmp_path / "không-có") == 0
    assert checkpoint_step(None) == 0


def test_smoke_preset_logs_loss_often_enough_to_be_visible():
    # Với logging_steps mặc định 50, smoke run 100 step kết thúc mà
    # `last_train_loss` vẫn None — đúng thứ cần báo cáo lại bị thiếu.
    smoke = apply_smoke_preset(BiEncoderTrainConfig(), 100)

    assert smoke.logging_steps <= smoke.max_steps // 2
