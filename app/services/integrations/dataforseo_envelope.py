"""The one place that decides whether a DataForSEO response actually succeeded.

**HTTP 200 means "the API received your request", not "your request worked."** Auth
failures, unknown locations, out-of-credit accounts and rejected tasks all come back
200 with a non-20000 `status_code` in the JSON body. Iterating `tasks` then yields
nothing, and the caller records `hits_returned=0, success=True` — indistinguishable
from a query that genuinely matched nothing.

That defect was found and fixed independently three times, in three subsystems:
the shared client (#14), `job_search_service` (#16 M3-10), and `mention_search_service`
(#18 M5-8). Three copies of one defect is the argument for one validator, so the
working implementation from `job_search_service` moved here and all three call it.

Raising vs. returning: `assert_ok` raises so the existing `except Exception` around
each provider call converts a rejected task into the module's own explicit failure
marker (`success=False` + `error_message`), which is what the cost logs and the
`ops.silent_zero` probes read. `is_ok` exists for the one caller that already returns
a structured result object instead of raising.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

#: DataForSEO's "everything worked" code, at both the envelope and task level.
DFS_OK = 20000

#: "Task Created" — the success code for the ASYNC endpoints (task_post). The task was
#: accepted and is queued; the results arrive from task_get or a webhook.
#:
#: This module accepted only 20000, so every async submission was reported as a failure.
#: `seo_trustpilot_search` came back "dataforseo task 20100: Task Created." — a success
#: message delivered as an error, on a call that had worked and been paid for.
#: `job_search_service` has always tolerated 20100 at its own call site; when the three
#: copies of this check were consolidated here, that case was the one left behind. Same
#: shape as the consolidation it was fixing.
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
        #
        # `domain_technologies` was posting to `/domain_analytics/technologies/technologies/live`
        # — a doubled segment — and had therefore never returned a single result. Note the rule
        # cannot be "reject doubled segments": `/backlinks/backlinks/live` is a real endpoint.
        # Only calling it tells you, which is why this says so in the message.
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
    #
    # `tasks_error` used to be checked ahead of this loop and returned a bare count —
    # "dataforseo reported 1 failed task(s)". That is the count of a thing whose
    # description was sitting one field away, and it made every DataForSEO failure read
    # identically: eleven tools in the 2026-08-26 sweep returned that exact sentence, and
    # not one of them said what had actually gone wrong. The count survives below, but only
    # when no task carries a reason of its own.
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
