"""The rollup says whether it ran, and a free success is marked (mivaa#30 M16-5, M16-6).

M16-5. `stamp_refresh_cost` and `recompute_lifetime_cost` existed twice — once per
subject domain — and both copies had the same three properties:

    if not tracked_x_id or not refresh_run_id:
        return                                   # a bug, shaped exactly like "nothing to do"
    try:
        ...rpc...
    except Exception as e:
        logger.warning(...)                      # swallowed
                                                 # -> None, so no caller could react

`stamp_job_refresh_cost` is the platform's CANONICAL silent zero. It referenced a column
that did not exist, the exception was swallowed, and per-subject billing read 0 for
months while every health signal stayed green. The column was fixed. The shape that hid
it was still in both files, waiting for the next one.

The rollup is what makes spend visible. When it stops, no number goes wrong — the numbers
stop moving, which is indistinguishable from a quiet month.

M16-6. A model call that succeeded consumed tokens. Zero of both is an accounting
failure, and it fails silently: `int(x or 0)` turns a missing usage block into 0, 0
tokens produce 0 raw cost, and the row is written `success=True`. That is the input to
M16-1 — a zero cost becomes a zero amount becomes a free operation reporting success.

`success` is deliberately NOT flipped. What is marked is the usage, not the call: the
provider may genuinely have answered, and turning a wrong cost into a wrong outcome is
not an improvement.

WATCHED TO FAIL: the source cases were run against the pre-fix tree and fired. The
behaviour cases exercise `usage_anomaly` and `stamp_rollup`, which are new.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

JOB = APP / "services" / "integrations" / "job_cost_logger.py"
MENTION = APP / "services" / "integrations" / "mention_cost_logger.py"
ROLLUP = APP / "services" / "integrations" / "cost_rollup.py"
CORE = APP / "modules" / "_core" / "cost_logger.py"

LOGGERS = [JOB, MENTION]


def _load(path: Path, name: str):
    """Both modules under test are stdlib-only at import time, which is the point —
    MIVAA's own CI installs pytest and nothing else, so a test that needs supabase to
    import cannot run there at all."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rollup = _load(ROLLUP, "cost_rollup_under_test")


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


# -------------------------------------------------------------------------
# M16-5 -- behaviour of the shared rollup
# -------------------------------------------------------------------------

def test_a_missing_id_is_reported_rather_than_returning_quietly():
    """The bare `return` is the whole finding. A caller that has just written an
    ai_usage_logs row and cannot name the subject it belongs to has spend it cannot
    attribute — which looked exactly like having nothing to do."""
    assert rollup.stamp_rollup(
        rpc="stamp_job_refresh_cost",
        params={"p_tracked_job_id": None},
        module="job-cost",
        required={"tracked_job_id": None, "refresh_run_id": "r1"},
    ) is False


def test_a_failing_rpc_returns_false_rather_than_none():
    """It must not RAISE — the provider call already happened and was already billed, so
    a bookkeeping failure must not destroy the result it is bookkeeping for. But the
    caller has to be able to tell."""
    import types

    fake = types.ModuleType("app.services.core.supabase_client")

    class _Client:
        def rpc(self, *_a, **_k):
            raise RuntimeError("relation does not exist")

    class _Wrapper:
        client = _Client()

    fake.get_supabase_client = lambda: _Wrapper()
    sys.modules["app.services.core.supabase_client"] = fake
    try:
        assert rollup.stamp_rollup(
            rpc="stamp_job_refresh_cost",
            params={"p_tracked_job_id": "j1"},
            module="job-cost",
            required={"tracked_job_id": "j1"},
        ) is False
    finally:
        del sys.modules["app.services.core.supabase_client"]


def test_a_successful_rollup_returns_true():
    import types

    fake = types.ModuleType("app.services.core.supabase_client")

    class _Exec:
        def execute(self):
            return {"data": []}

    class _Client:
        def rpc(self, *_a, **_k):
            return _Exec()

    class _Wrapper:
        client = _Client()

    fake.get_supabase_client = lambda: _Wrapper()
    sys.modules["app.services.core.supabase_client"] = fake
    try:
        assert rollup.stamp_rollup(
            rpc="recompute_job_cost",
            params={"p_tracked_job_id": "j1"},
            module="job-cost",
            required={"tracked_job_id": "j1"},
        ) is True
    finally:
        del sys.modules["app.services.core.supabase_client"]


def test_a_lost_rollup_is_logged_at_error_not_warning():
    """Severity is not decoration here. The DB log sink drops sub-WARNING records from
    noisy libraries, and a lost rollup is spend that already happened and is now
    unattributed."""
    src = _strip_comments(_read(ROLLUP))
    assert "logger.warning" not in src, (
        "the shared rollup is back to warning about a lost rollup"
    )
    assert src.count("logger.error") >= 2


# -------------------------------------------------------------------------
# M16-5 -- one implementation, not two
# -------------------------------------------------------------------------

def test_neither_logger_hand_rolls_the_rollup_again():
    for path in LOGGERS:
        src = _strip_comments(_read(path))
        for fn in ("stamp_refresh_cost", "recompute_lifetime_cost"):
            body = ast.get_source_segment(src, _node(src, fn)) or ""
            assert "stamp_rollup(" in body, (
                f"{path.name}::{fn} no longer delegates to cost_rollup (#30 M16-5) — "
                "two copies is how one of them drifts, and this exact pair is where the "
                "platform's canonical silent zero lived"
            )
            assert ".rpc(" not in body, f"{path.name}::{fn} calls the RPC directly again"


def test_both_rollups_return_a_verdict():
    for path in LOGGERS:
        src = _read(path)
        for fn in ("stamp_refresh_cost", "recompute_lifetime_cost"):
            node = _node(src, fn)
            returns = ast.unparse(node.returns) if node.returns else ""
            assert returns == "bool", (
                f"{path.name}::{fn} returns {returns or 'nothing'} again, so no caller "
                "can tell a rollup that ran from one that did not"
            )


def test_the_required_ids_are_named_so_the_log_says_which_one_was_missing():
    for path in LOGGERS:
        body = ast.get_source_segment(
            _read(path), _node(_read(path), "stamp_refresh_cost")
        ) or ""
        assert "required=" in body, (
            f"{path.name}: the rollup no longer declares which ids it needs, so a "
            "missing one is anonymous again"
        )


# -------------------------------------------------------------------------
# M16-6 -- a free success is marked
# -------------------------------------------------------------------------

def _usage_anomaly():
    """Imported by source rather than by module, because `_core.cost_logger` pulls in
    supabase at import time."""
    src = _read(CORE)
    seg = ast.get_source_segment(src, _node(src, "usage_anomaly"))
    ns: dict = {"Optional": None}
    marker = re.search(r'ZERO_USAGE_MARKER = "(.*?)"', src).group(1)
    ns["ZERO_USAGE_MARKER"] = marker
    exec(seg.replace("-> Optional[str]", ""), ns)
    return ns["usage_anomaly"], marker


def test_a_successful_call_with_no_tokens_is_marked():
    fn, marker = _usage_anomaly()
    assert fn(0, 0, True) == marker


def test_a_failed_call_with_no_tokens_is_not_marked():
    """A call that failed is ALLOWED to have no usage — that is the ordinary shape of a
    failure, and marking it would bury the real ones."""
    fn, _ = _usage_anomaly()
    assert fn(0, 0, False) is None


def test_a_real_call_is_not_marked():
    fn, _ = _usage_anomaly()
    assert fn(120, 40, True) is None
    assert fn(0, 40, True) is None, (
        "output-only is a real answer to a cached-prompt call; only zero of BOTH is the "
        "accounting failure"
    )


def test_both_loggers_mark_it_and_neither_flips_success():
    for path in LOGGERS:
        body = ast.get_source_segment(_read(path), _node(_read(path), "log_haiku_call")) or ""
        assert "usage_anomaly(" in body, (
            f"{path.name}::log_haiku_call writes a zero-cost success unmarked again "
            "(#30 M16-6)"
        )
        assert "error_message=error_message or anomaly" in body
        assert "success=False" not in body, (
            "success is being overwritten — the CALL may genuinely have succeeded, and "
            "trading a wrong cost for a wrong outcome is not a fix"
        )


def test_the_marker_is_one_named_constant():
    """Two loggers writing two spellings of the same fact is a cost view that finds half
    of them."""
    src = _read(CORE)
    assert "ZERO_USAGE_MARKER" in src
    for path in LOGGERS:
        assert "usage_missing:" not in _strip_comments(_read(path)), (
            f"{path.name} restates the marker text instead of using the constant"
        )
