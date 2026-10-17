"""Register the stdlib-only `app.utils.cross_loop_lock` by file path.

Many unit tests replace `app` with stub packages and load the module under test by path; a
dotted import of a module already in `sys.modules` resolves without its parent packages.
"""
import importlib.util
import sys
from pathlib import Path

_NAME = "app.utils.cross_loop_lock"
if _NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _NAME, Path(__file__).resolve().parents[2] / "app" / "utils" / "cross_loop_lock.py")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_NAME] = _module
    _spec.loader.exec_module(_module)
