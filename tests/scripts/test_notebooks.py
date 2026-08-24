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

    assert "--smoke" in missing
    assert "resume_verified" in missing
