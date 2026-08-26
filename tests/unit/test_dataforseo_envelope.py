"""Behavioral tests for the one place that decides whether DataForSEO actually succeeded.

Real tests, not a source scan — `app.services.integrations.dataforseo_envelope` imports
nothing beyond `typing`, so it can be exercised directly under CI's pytest-only install.

CI CONSTRAINT (see test_workspace_resolution.py): CI installs pytest and NOTHING else.
A third-party import here makes the module uncollectable and takes the ENTIRE suite down.
This file therefore imports only stdlib plus the module under test.

The two defects these pin, both found by the 2026-08-26 agent tool sweep:

  1. `20100 Task Created` was treated as a FAILURE. It is the success code for the async
     endpoints — the task was accepted and queued. `seo_trustpilot_search` reported
     "dataforseo task 20100: Task Created." as its error, on a call that had worked and
     been paid for. `job_search_service` had always tolerated 20100 at its own call site;
     when three copies of this check were consolidated into this module, that case was the
     one left behind — the same shape as the consolidation it was fixing.

  2. `tasks_error` was checked BEFORE the per-task loop and returned a bare count. Eleven
     tools in that sweep came back with the identical sentence "dataforseo reported 1
     failed task(s)", none of them saying what went wrong — while the reason sat one field
     away in the task the loop would have read.
"""

import importlib.util
import os

# Loaded BY PATH, not as `app.services.integrations.dataforseo_envelope`.
#
# The package import pulls `app/services/__init__.py`, which imports the supabase client,
# which needs the `supabase` package — absent under CI's pytest-only install. That would
# not fail this file alone; an uncollectable module takes the whole suite down. The module
# under test imports nothing but `typing`, so loading the file directly exercises the real
# code with no package side effects.
_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "app", "services", "integrations", "dataforseo_envelope.py"
)
_spec = importlib.util.spec_from_file_location("dataforseo_envelope_under_test", _PATH)
env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(env)


def test_plain_success():
    ok, reason = env.check({"status_code": 20000, "tasks": [{"status_code": 20000}]})
    assert ok is True
    assert reason is None


def test_task_created_is_success_not_failure():
    """20100 at task level means queued, not broken."""
    ok, reason = env.check(
        {"status_code": 20000, "tasks": [{"status_code": 20100, "status_message": "Task Created."}]}
    )
    assert ok is True, f"20100 must be accepted, got: {reason}"
    assert reason is None


def test_task_created_is_success_at_envelope_level_too():
    ok, reason = env.check(
        {"status_code": 20100, "status_message": "Task Created.", "tasks": [{"status_code": 20100}]}
    )
    assert ok is True, f"envelope 20100 must be accepted, got: {reason}"


def test_real_task_failure_reports_the_providers_own_words():
    """The reason must be DataForSEO's message, not a count of how many failed."""
    ok, reason = env.check(
        {
            "status_code": 20000,
            "tasks_error": 1,
            "tasks": [{"status_code": 40501, "status_message": "Invalid Field: 'target'"}],
        }
    )
    assert ok is False
    assert "Invalid Field" in reason, reason
    assert "failed task(s)" not in reason, (
        "the per-task message must win over the count — the count is what made every "
        f"failure read identically. got: {reason}"
    )


def test_count_survives_only_when_no_task_gives_a_reason():
    ok, reason = env.check({"status_code": 20000, "tasks_error": 2, "tasks": [{"status_code": 20000}]})
    assert ok is False
    assert "2 failed task(s)" in reason
    assert "no per-task reason" in reason, reason


def test_envelope_failure_still_reported():
    ok, reason = env.check({"status_code": 40101, "status_message": "Auth error", "tasks": []})
    assert ok is False
    assert "40101" in reason and "Auth error" in reason


def test_no_tasks_is_a_failure_not_an_empty_result():
    ok, reason = env.check({"status_code": 20000, "tasks": []})
    assert ok is False
    assert "no tasks" in reason


def test_non_object_body():
    ok, reason = env.check(None)
    assert ok is False
    assert "non-object" in reason


def test_assert_ok_raises_with_the_reason():
    try:
        env.assert_ok({"status_code": 20000, "tasks": [{"status_code": 40501, "status_message": "Nope"}]})
    except RuntimeError as e:
        assert "Nope" in str(e)
    else:
        raise AssertionError("assert_ok must raise on a failed task")


def test_success_codes_are_the_single_source():
    """Both success codes live in one frozenset, so a caller cannot disagree with this module."""
    assert env.DFS_OK in env.DFS_SUCCESS_CODES
    assert env.DFS_TASK_CREATED in env.DFS_SUCCESS_CODES
    assert 40501 not in env.DFS_SUCCESS_CODES
