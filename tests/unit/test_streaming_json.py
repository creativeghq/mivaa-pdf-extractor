"""`ArrayItemStreamer` emits every element, once, whole — whatever the chunking.

WHY THIS IS TESTED AND NOT EYEBALLED
------------------------------------
The failure mode is the one this codebase keeps being audited for: a DROPPED element is
indistinguishable from an element the model never emitted. If a `}` inside a string
("wall {left of window}") is counted as a closing brace, or a chunk boundary splits an
escape sequence, the streamer silently yields eleven zones where twelve were written —
and the caller renders eleven, confidently, with nothing raising anywhere.

So the sweep here is exhaustive on the axis that actually varies at runtime: the SAME
payload re-fed at every chunk size from 1 byte upward must produce a byte-identical
result. Anthropic's `input_json_delta` fragments have no relationship to JSON structure,
so any size is a real case.

Loads the module by path — MIVAA's CI installs pytest and nothing else, so a test that
imported `app.*` would not run.
"""

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _ROOT / "app" / "services" / "core" / "streaming_json.py"


def _load():
    spec = importlib.util.spec_from_file_location("streaming_json", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


streaming_json = _load()
ArrayItemStreamer = streaming_json.ArrayItemStreamer


#: A reply shaped like a real one: nested bbox object, a brace and a quote inside a
#: string value, a unicode escape, an empty-ish zone, and a key AFTER the array.
ZONES = [
    {"bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}, "label": "floor", "confidence": 0.91},
    {"bbox": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.5}, "label": 'wall {left of "window"}'},
    {"bbox": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}, "label": "πλακάκι", "finish": "matte"},
    {"bbox": {"x": 0.9, "y": 0.9, "w": 0.05, "h": 0.05}},
]
PAYLOAD = json.dumps({"zones": ZONES, "note": "trailing key"})


def _drain(chunks):
    streamer = ArrayItemStreamer("zones")
    out = []
    for chunk in chunks:
        out.extend(streamer.feed(chunk))
    return streamer, out


def _split(text, size):
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_every_chunk_size_yields_the_same_zones():
    """The chunk boundary is Anthropic's, not ours. One byte at a time is a real case."""
    for size in range(1, len(PAYLOAD) + 1):
        streamer, out = _drain(_split(PAYLOAD, size))
        assert out == ZONES, f"chunk size {size} produced {len(out)} zones, not {len(ZONES)}"
        assert streamer.dropped == 0, f"chunk size {size} dropped an element"
        assert streamer.finished, f"chunk size {size} never saw the closing bracket"


def test_a_brace_inside_a_string_does_not_end_an_element():
    """`wall {left of "window"}` is a label, not structure. Counting its braces splits
    one zone into two malformed halves — and the halves do not parse, so they vanish."""
    _, out = _drain([PAYLOAD])
    labels = [z.get("label") for z in out]
    assert 'wall {left of "window"}' in labels
    assert len(out) == len(ZONES)


def test_zones_arrive_before_the_payload_is_complete():
    """The whole point: element 1 is available while element 4 is still unwritten."""
    cut = PAYLOAD.index('"label": "wall')
    streamer, early = _drain([PAYLOAD[:cut]])
    assert early == ZONES[:1], "the first zone should already be out"
    assert not streamer.finished
    assert streamer.feed(PAYLOAD[cut:]) == ZONES[1:]


def test_an_escaped_quote_split_across_chunks_is_not_a_string_end():
    payload = json.dumps({"zones": [{"bbox": {}, "label": 'a \\" b'}]})
    for size in range(1, len(payload) + 1):
        _, out = _drain(_split(payload, size))
        assert len(out) == 1, f"chunk size {size} mis-tracked the escape"
        assert out[0]["label"] == 'a \\" b'


def test_text_after_the_array_is_ignored():
    """`note` sits after `zones` and is not an element. A scanner that kept going would
    emit it as a zone with no bbox, which validation would drop — quietly turning a
    parser bug into a missing zone."""
    _, out = _drain([PAYLOAD])
    assert all("bbox" in z for z in out)


def test_a_key_that_is_not_ours_is_not_mistaken_for_the_array():
    """`"zones"` appearing as a VALUE, or a different key ending in the same letters,
    must not start the scan — otherwise the real array is missed entirely and the
    caller sees zero zones from a reply that had twelve."""
    payload = json.dumps({"source_zones": "zones", "zones": ZONES[:2]})
    for size in (1, 3, 7, len(payload)):
        _, out = _drain(_split(payload, size))
        assert out == ZONES[:2], f"chunk size {size} locked onto the wrong key"


def test_an_unparseable_element_is_counted_not_hidden():
    """A `dropped` counter that stayed at 0 while elements went missing would make this
    module's own failure invisible, which is the bug it exists to prevent."""
    streamer = ArrayItemStreamer("zones")
    out = streamer.feed('{"zones": [{"bbox": {,}}, {"bbox": {"x": 1}}]}')
    assert streamer.dropped == 1
    assert out == [{"bbox": {"x": 1}}]


def test_an_empty_array_finishes_cleanly():
    streamer, out = _drain(['{"zones": []}'])
    assert out == []
    assert streamer.finished
