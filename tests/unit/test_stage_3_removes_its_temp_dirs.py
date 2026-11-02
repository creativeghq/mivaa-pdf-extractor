"""Every temp folder Stage 3 creates is removed when the stage ends, success or failure."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _code(path: str) -> str:
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    return budget.blank_comments((_ROOT / path).read_text(encoding="utf-8"))


def test_every_stage_3_temp_dir_is_registered_for_removal():
    src = _code("app/api/pdf_processing/stage_3_images.py")
    made = src.count("tempfile.mkdtemp(")
    assert made >= 1
    assert src.count("temp_dirs.append(") == made, "a folder made here must be appended to temp_dirs"
    wrapper = src[src.index("async def process_product_images("):src.index("async def _process_product_images(")]
    finally_block = wrapper[wrapper.index("finally:"):]
    assert "shutil.rmtree(" in finally_block and "for d in temp_dirs" in finally_block
