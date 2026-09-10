"""The one place that decides whether a DataForSEO response actually succeeded."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

#: DataForSEO's "everything worked" code, at both the envelope and task level.
DFS_OK = 20000

#: "Task Created" — the success code for the ASYNC endpoints (task_post). The task was
#: accepted and is queued; the results arrive from task_get or a webhook.
DFS_TASK_CREATED = 20100

#: Codes that mean the call succeeded.
DFS_SUCCESS_CODES = frozenset({DFS_OK, DFS_TASK_CREATED})

#: "Not Found" at the ENVELOPE level — the endpoint URL itself does not exist. Ours, not theirs.
DFS_PATH_NOT_FOUND = 40400


def _task_failure(tasks: Any) -> Optional[str]:
    """The first task-level failure, as DataForSEO's own words. None when all tasks are fine."""
    for task in tasks or []:
        code = (task or {}).get("status_code")
        if code is not None and code not in DFS_SUCCESS_CODES:
            return f"dataforseo task {code}: {(task or {}).get('status_message')}"
    return None


def check(data: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """(ok, reason). `reason` is None when the response really did succeed."""
    if not isinstance(data, dict):
        return False, "dataforseo returned a non-object body"

    envelope_code = data.get("status_code")
    if envelope_code == DFS_PATH_NOT_FOUND:
        # 40400 at the ENVELOPE is not "your query matched nothing" — it is DataForSEO saying the
        # URL does not exist, i.e. OUR path is wrong. That is a code defect, and it hides
        # perfectly: a 404 costs nothing, so the tool is both permanently empty and invisible in
        # the spend, which reads exactly like a tool nobody uses.
        return False, (
            f"dataforseo envelope {envelope_code}: {data.get('status_message')} "
            "— this endpoint PATH does not exist at DataForSEO. Fix the URL in "
            "dataforseo_unified_client; it is not a query or a credentials problem."
        )
    if envelope_code is not None and envelope_code not in DFS_SUCCESS_CODES:
        return False, f"dataforseo envelope {envelope_code}: {data.get('status_message')}"

    tasks = data.get("tasks") or []
    if not tasks:
        # An empty task list is not an empty result set — a successful call always
        # echoes at least the task it ran.
        return False, "dataforseo returned no tasks"

    # The PER-TASK message first, because it is the only diagnostic in the response.
    task_reason = _task_failure(tasks)
    if task_reason:
        return False, task_reason

    tasks_error = data.get("tasks_error")
    if isinstance(tasks_error, int) and tasks_error > 0:
        return False, (
            f"dataforseo reported {tasks_error} failed task(s) and gave no per-task reason"
        )

    return True, None


def is_ok(data: Optional[Dict[str, Any]]) -> bool:
    """True when the envelope reports success. Prefer `assert_ok` — a bare bool
    discards the reason, and the reason is what makes a failed run diagnosable."""
    ok, _ = check(data)
    return ok


def assert_ok(data: Optional[Dict[str, Any]]) -> None:
    """Raise RuntimeError unless the DataForSEO envelope reports success."""
    ok, reason = check(data)
    if not ok:
        raise RuntimeError(reason)
