"""Whether image sourcing (#525) is on for the work running in THIS task.

On by default: an ingestion that skips it keeps every low-res crop and never looks at the maker's
site (DOCKS shipped 51 px swatches while cereuro.it held every colour). IMAGE_SOURCING_ENABLED=false
turns it off for the service; a caller such as image-redo can still force it on for its own task.
"""

import os
from contextvars import ContextVar

_forced: ContextVar[bool] = ContextVar("image_sourcing_forced", default=False)


def sourcing_enabled() -> bool:
    return _forced.get() or os.getenv("IMAGE_SOURCING_ENABLED", "true").lower() == "true"


def force_sourcing_for_this_task() -> None:
    _forced.set(True)
