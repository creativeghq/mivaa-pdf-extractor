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


def check(data: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """(ok, reason). `reason` is None when the response really did succeed."""
    if not isinstance(data, dict):
        return False, "dataforseo returned a non-object body"

    envelope_code = data.get("status_code")
    if envelope_code is not None and envelope_code != DFS_OK:
        return False, f"dataforseo envelope {envelope_code}: {data.get('status_message')}"

    tasks_error = data.get("tasks_error")
    if isinstance(tasks_error, int) and tasks_error > 0:
        return False, f"dataforseo reported {tasks_error} failed task(s)"

    tasks = data.get("tasks") or []
    if not tasks:
        # An empty task list is not an empty result set — a successful call always
        # echoes at least the task it ran.
        return False, "dataforseo returned no tasks"

    for task in tasks:
        code = (task or {}).get("status_code")
        if code is not None and code != DFS_OK:
            return False, f"dataforseo task {code}: {(task or {}).get('status_message')}"

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
