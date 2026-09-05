"""Notebook Kaggle phải đủ khối, không chỉ đúng cú pháp.

`SMOKE_CELLS` từng được định nghĩa nhưng quên nối vào `BIENCODER_CELLS`:
notebook sinh ra thiếu hẳn phần smoke/resume mà vẫn hợp lệ cú pháp, nên lọt qua
mọi kiểm tra cho tới khi chạy thật trên Kaggle. Test này chặn đúng lớp lỗi đó.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("method2_*.ipynb"))


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_notebooks", "scripts/method2/make_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_shell(source: str) -> str:
    r"""Bỏ dòng `!`/`%` của IPython, kể cả dòng nối bằng `\`."""
    out, skipping = [], False
    for line in source.split("\n"):
        stripped = line.strip()
        if skipping or stripped.startswith(("!", "%")):
            out.append("")
            skipping = stripped.endswith("\\")
            continue
        out.append(line)
    return "\n".join(out)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_every_required_block(path):
    module = _load_generator()

    assert module.check_markers(path) == []


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        ast.parse(_strip_shell("".join(cell["source"])), filename=f"{path.name}:cell{index}")


def test_notebooks_match_generator(tmp_path):
    """File .ipynb phải là bản sinh ra từ script, không sửa tay."""
    module = _load_generator()
    module.NOTEBOOK_DIR = tmp_path
    for path in module.build_notebooks():
        committed = Path("notebooks") / path.name
        assert committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), (
            f"{path.name} lệch với make_notebooks.py — chạy lại script rồi commit"
        )


def test_marker_check_catches_a_dropped_block(tmp_path):
    module = _load_generator()
    cells = module._setup("t", "s") + [module._cell(k, s) for k, s in module.PREFLIGHT_CELLS]
    incomplete = tmp_path / "method2_kaggle_biencoder.ipynb"
    incomplete.write_text(
        json.dumps(module._notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8"
    )

    missing = module.check_markers(incomplete)

    assert "Full training" in missing
    assert "biencoder.index" in missing


def _subprocess_train_blocks(source: str, module: str) -> list[str]:
    """Gom lệnh dạng list `[sys.executable, '-m', <module>, 'train', ...]`.

    Round 1/Round 2 gọi bằng `subprocess.run` chứ không phải `!python`, nên nếu
    chỉ quét dạng shell thì test tưởng là "không có lệnh train nào" và pass rỗng.
    """
    needle = f"'{module}', 'train'"
    blocks, lines = [], source.split(chr(10))
    for i, line in enumerate(lines):
        if needle not in line:
            continue
        window = []
        for nxt in lines[i:]:
            window.append(nxt)
            if "]" in nxt:
                break
        blocks.append(chr(10).join(window))
    return blocks


def _train_commands(source: str, module: str) -> list[str]:
    r"""Gom lệnh `!python -m <module>`, ghép cả dòng nối bằng `\`."""
    blocks, current = [], None
    for line in source.split("\n"):
        if module in line and line.strip().startswith("!"):
            current = [line]
        elif current is not None:
            current.append(line)
            if not line.rstrip().endswith("\\"):
                blocks.append("\n".join(current))
                current = None
    if current:
        blocks.append("\n".join(current))
    return blocks


def test_every_biencoder_train_command_forces_single_gpu():
    """Benchmark đo ở chế độ 1 GPU; để DataParallel bật lại thì số đo vô nghĩa.

    sentence-transformers tự bọc DataParallel khi thấy >1 GPU, và với GradCache
    gọi model hàng trăm lần mỗi step thì phí đồng bộ đẩy từ ~36 lên ~475 s/step.
    """
    source = "".join(
        "".join(c["source"])
        for c in json.loads(
            (Path("notebooks") / "method2_kaggle_biencoder.ipynb").read_text(encoding="utf-8")
        )["cells"]
    )

    commands = _train_commands(source, "src.models.biencoder.train train")
    commands += _subprocess_train_blocks(source, "src.models.biencoder.train")

    assert commands, "không tìm thấy lệnh train nào"
    missing = [c for c in commands if "--single-gpu" not in c]
    assert not missing, f"{len(missing)} lệnh train thiếu --single-gpu"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_shell_commands_never_interpolate_a_maybe_none_variable(path):
    """`!python ... --resume-from {resume}` nội suy None thành chuỗi "None".

    HF Trainer coi "None" là đường dẫn checkpoint rồi đi tải
    `sentence-transformers/None` từ Hub — chết ngay vì đang chạy offline. Cách
    an toàn là dựng sẵn `resume_arg` (rỗng khi không có checkpoint) rồi nội suy
    biến đó, nên chỉ kiểm tra bên trong lệnh shell.
    """
    source = "".join(
        "".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
    )

    for module in ("src.models.biencoder.train", "src.models.crossencoder.train"):
        for command in _train_commands(source, module):
            assert "{resume}" not in command, f"nội suy biến có thể None: {command[:80]}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_checkpoints_sorted_numerically(path):
    """`sorted()` theo tên cho 'checkpoint-1000' < 'checkpoint-500'."""
    source = "".join(
        "".join(c["source"]) for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
    )

    assert "sorted(glob.glob(f'{RUN}/checkpoint-*'))[-1]" not in source


def test_clean_resume_treats_none_string_as_no_resume():
    from src.models.biencoder.train import clean_resume

    assert clean_resume("None") is None
    assert clean_resume("none") is None
    assert clean_resume("") is None
    assert clean_resume("  ") is None
    assert clean_resume(None) is None
    assert clean_resume("artifacts/run01/checkpoint-500") == "artifacts/run01/checkpoint-500"


def test_required_markers_khong_co_muc_trung():
    """Mỗi marker khai một lần.

    Trùng lặp đã xảy ra ba lần khi chèn khối cell bằng script: helper chạy lại
    sau một lần hỏng giữa chừng và cộng thêm dòng marker cũ. Vô hại về chức năng
    nhưng là dấu hiệu file đã bị sửa hai lần chồng nhau — chỗ đó đáng để bắt.
    """
    module = _load_generator()

    for name, markers in module.REQUIRED_MARKERS.items():
        assert len(markers) == len(set(markers)), (
            f"{name}: marker trùng {sorted({m for m in markers if markers.count(m) > 1})}"
        )
