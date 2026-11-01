"""Whether image sourcing (#525) is on for the work running in THIS task.

Off unless IMAGE_SOURCING_ENABLED=true for the service, or a caller such as the image-redo
endpoint turns it on for its own task only; a process-wide switch would also flip it for
every other ingestion running in the same service.
"""

import os
from contextvars import ContextVar

_forced: ContextVar[bool] = ContextVar("image_sourcing_forced", default=False)


def sourcing_enabled() -> bool:
    return _forced.get() or os.getenv("IMAGE_SOURCING_ENABLED", "false").lower() == "true"


def force_sourcing_for_this_task() -> None:
    _forced.set(True)
