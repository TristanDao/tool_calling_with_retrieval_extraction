"""Resume của Cross-Encoder trên Kaggle — nơi session bị ngắt là chuyện thường.

Ba lỗi từng có, mỗi lỗi một test:

1. `train()` dựng model mới từ `xlm-roberta-base` rồi chỉ nạp optimizer state —
   toàn bộ trọng số đã train bị vứt, resume còn tệ hơn train lại từ đầu vì
   optimizer state không còn khớp trọng số nào.
2. `trainer_state.pt` không ghi vị trí trong curriculum, nên resume luôn quay
   về giai đoạn 0 epoch 0 và train lại cả phần warm-up đã xong.
3. `find_last_checkpoint` sắp theo tên: `checkpoint-1000` < `checkpoint-500`,
   và `stage-warmup` thì `int()` ném ValueError.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from src.models.crossencoder.train import (  # noqa: E402
    _read_trainer_state,
    find_last_checkpoint,
)


def _make_checkpoint(root, name: str, step: int, stage_index: int = 0, epoch: int = 0):
    path = root / name
    path.mkdir(parents=True)
    (path / "progress.json").write_text(
        json.dumps({"step": step, "stage_index": stage_index, "epoch": epoch}), encoding="utf-8"
    )
    torch.save({"step": step, "optimizer": {}}, path / "trainer_state.pt")
    return path


def test_checkpoint_records_curriculum_position(tmp_path):
    """Thiếu stage_index/epoch thì resume train lại toàn bộ warm-up."""
    _make_checkpoint(tmp_path, "checkpoint-4200", 4200, stage_index=1, epoch=1)

    position = _read_trainer_state(str(tmp_path / "checkpoint-4200"))

    assert position == {"step": 4200, "stage_index": 1, "epoch": 1}


def test_old_checkpoint_without_progress_still_readable(tmp_path):
    """Checkpoint sinh trước khi vá không có progress.json — không được vỡ."""
    path = tmp_path / "checkpoint-100"
    path.mkdir()
    torch.save({"step": 100, "optimizer": {}}, path / "trainer_state.pt")

    assert _read_trainer_state(str(path)) == {"step": 100, "stage_index": 0, "epoch": 0}


def test_find_last_checkpoint_sorts_numerically_not_lexically(tmp_path):
    """'checkpoint-1000' < 'checkpoint-500' theo thứ tự chữ."""
    _make_checkpoint(tmp_path, "checkpoint-500", 500)
    _make_checkpoint(tmp_path, "checkpoint-1000", 1000)

    assert find_last_checkpoint(tmp_path).endswith("checkpoint-1000")


def test_find_last_checkpoint_sees_stage_tagged_checkpoint(tmp_path):
    """Checkpoint cuối giai đoạn mang tên `stage-warmup`, không có số.

    Bỏ sót nó thì sau khi warm-up xong mà session đứt, resume quay về giữa
    warm-up và train lại phần đã xong.
    """
    _make_checkpoint(tmp_path, "checkpoint-4000", 4000, stage_index=0, epoch=1)
    _make_checkpoint(tmp_path, "stage-warmup", 4066, stage_index=0, epoch=2)

    last = find_last_checkpoint(tmp_path)

    assert last.endswith("stage-warmup")
    assert _read_trainer_state(last)["epoch"] == 2


def test_checkpoint_without_trainer_state_is_ignored(tmp_path):
    """Thư mục `final/` chỉ có trọng số, không resume được từ đó."""
    (tmp_path / "final").mkdir()
    _make_checkpoint(tmp_path, "checkpoint-10", 10)

    assert find_last_checkpoint(tmp_path).endswith("checkpoint-10")


def test_train_loads_weights_from_checkpoint_when_resuming():
    """`train()` phải nạp trọng số qua `from_pretrained`, không dựng model mới.

    Kiểm bằng source vì chạy thật cần GPU: nhánh resume phải gọi
    `CrossEncoderForExtraction.from_pretrained`, và phải nằm TRƯỚC vòng lặp
    stage (dựng model xong mới nạp thì trọng số mới bị ghi đè lung tung).
    """
    import inspect

    from src.models.crossencoder import train as train_module

    source = inspect.getsource(train_module.train)

    assert "CrossEncoderForExtraction.from_pretrained(resume_path)" in source
    assert source.index("from_pretrained(resume_path)") < source.index("for stage_index, stage in")


def test_smoke_flag_writes_to_a_separate_output_dir():
    """Smoke ghi đè lên run thật thì mất checkpoint của run dài."""
    import inspect

    from src.models.crossencoder import train as train_module

    source = inspect.getsource(train_module.main)

    assert "smoke_run01" in source
    assert "config.max_steps = args.smoke" in source
