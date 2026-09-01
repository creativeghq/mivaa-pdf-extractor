"""Pull complete objects out of a JSON array that is still being written.

WHY
---
A forced tool call returns one object — `{"zones": [ ... ]}` — and the whole object is
only parseable once the last byte arrives. For segmentation that is ~40 seconds during
which the caller has every zone the model has already decided on and can show none of
them, because `json.loads` on a truncated array raises.

Anthropic streams a forced tool's input as `input_json_delta` fragments of exactly that
text. This scanner consumes those fragments and emits each array ELEMENT the moment its
closing brace arrives, so a consumer can render zone 1 while the model is still writing
zone 9.

WHAT IT IS NOT
--------------
Not a JSON parser. It finds the named array, then tracks string/escape state and brace
depth to know where one element ends — and hands the element itself to `json.loads`,
which is the only thing that decides whether it is valid. Anything before the array and
anything after it is ignored: this answers one question, and a scanner that tried to
answer more would be a second JSON implementation to keep correct.

Deliberately stdlib-only and app-import-free: MIVAA's CI installs pytest and nothing
else, so a derivation that matters has to be loadable by a test on its own.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

_WHITESPACE = " \t\r\n"


class ArrayItemStreamer:
    """Feed it JSON text in arbitrary chunks; get back finished array elements.

    Chunk boundaries are meaningless to the caller — the key name, a `:` and the
    opening `[` may each arrive in a different fragment, and an element may be split
    mid-string. State is carried across `feed()` calls so none of that matters.

        streamer = ArrayItemStreamer("zones")
        for fragment in fragments:
            for zone in streamer.feed(fragment):
                ...   # a complete zone dict, as soon as it closed
    """

    def __init__(self, array_key: str) -> None:
        self._key_token = f'"{array_key}"'
        self._buf = ""
        #: Have we found `"<key>" : [` yet? Until then `_buf` holds the unmatched head.
        self._started = False
        #: Did we see the array's closing `]`? Everything after it is ignored.
        self._finished = False
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._item: List[str] = []
        self._dropped = 0

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def dropped(self) -> int:
        """Elements that closed but did not parse.

        Should be zero. It is counted rather than swallowed because a silently
        discarded zone is indistinguishable from a zone the model never emitted, and
        that is the failure this whole file exists to avoid on the other axis.
        """
        return self._dropped

    def feed(self, chunk: str) -> List[Any]:
        """Return every array element completed by this chunk (possibly none)."""
        if self._finished:
            return []
        self._buf += chunk
        if not self._started and not self._locate_array():
            return []
        return self._scan()

    def _locate_array(self) -> bool:
        """Advance past `"<key>" <ws> : <ws> [`. False = not enough text yet."""
        search_from = 0
        while True:
            idx = self._buf.find(self._key_token, search_from)
            if idx < 0:
                # Keep only what could still be a PREFIX of the token — the key may be
                # split across two fragments, and dropping the tail would lose it.
                keep = len(self._key_token) - 1
                self._buf = self._buf[-keep:] if keep else ""
                return False

            cursor = idx + len(self._key_token)
            cursor = self._skip_ws(cursor)
            if cursor is None:
                self._buf = self._buf[idx:]  # incomplete — wait for more text
                return False
            if self._buf[cursor] != ":":
                search_from = idx + 1  # the token was a value, not our key
                continue

            cursor = self._skip_ws(cursor + 1)
            if cursor is None:
                self._buf = self._buf[idx:]
                return False
            if self._buf[cursor] != "[":
                search_from = idx + 1
                continue

            self._buf = self._buf[cursor + 1:]
            self._started = True
            return True

    def _skip_ws(self, cursor: int) -> int | None:
        while cursor < len(self._buf) and self._buf[cursor] in _WHITESPACE:
            cursor += 1
        return None if cursor >= len(self._buf) else cursor

    def _scan(self) -> List[Any]:
        out: List[Any] = []
        consumed = 0
        for ch in self._buf:
            consumed += 1

            if self._depth == 0:
                # Between elements: commas and whitespace only, until `{` or `]`.
                if ch == "{":
                    self._depth = 1
                    self._item = ["{"]
                elif ch == "]":
                    self._finished = True
                    break
                continue

            self._item.append(ch)

            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif ch == "\\":
                    self._escaped = True
                elif ch == '"':
                    self._in_string = False
                continue

            if ch == '"':
                self._in_string = True
            elif ch == "{":
                self._depth += 1
            elif ch == "}":
                self._depth -= 1
                if self._depth == 0:
                    text = "".join(self._item)
                    self._item = []
                    try:
                        out.append(json.loads(text))
                    except ValueError:
                        self._dropped += 1
                        logger.warning(
                            "streamed array element did not parse (%d chars): %.120s",
                            len(text), text,
                        )

        self._buf = self._buf[consumed:]
        return out
