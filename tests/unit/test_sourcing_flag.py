import contextvars
import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "app/services/images/sourcing_flag.py"
_spec = importlib.util.spec_from_file_location("sourcing_flag_under_test", _PATH)
flag = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = flag
_spec.loader.exec_module(flag)


def test_sourcing_runs_unless_switched_off(monkeypatch):
    monkeypatch.delenv("IMAGE_SOURCING_ENABLED", raising=False)
    assert contextvars.copy_context().run(flag.sourcing_enabled)
    monkeypatch.setenv("IMAGE_SOURCING_ENABLED", "false")
    assert not contextvars.copy_context().run(flag.sourcing_enabled)


def test_a_task_can_force_it_on_when_the_service_has_it_off(monkeypatch):
    monkeypatch.setenv("IMAGE_SOURCING_ENABLED", "false")

    def forced():
        flag.force_sourcing_for_this_task()
        return flag.sourcing_enabled()

    assert contextvars.copy_context().run(forced)
    assert not contextvars.copy_context().run(flag.sourcing_enabled)
