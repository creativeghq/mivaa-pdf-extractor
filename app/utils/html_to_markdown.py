"""Job-ad HTML to light markdown, stdlib only.

Keeps what a reader of a job body needs (headings, paragraphs, lists, bold) and
drops everything else, including script/style/iframe content.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser

_DROP = {"script", "style", "iframe", "noscript", "svg", "head", "template"}
_BLOCK = {"p", "div", "section", "article", "header", "footer", "table", "tr", "blockquote", "ul", "ol"}
_HEADING = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "####", "h6": "####"}


class _Converter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._drop_depth = 0

    def _newline(self, n: int = 2) -> None:
        self.out.append("\n" * n)

    def handle_starttag(self, tag, attrs):
        if tag in _DROP:
            self._drop_depth += 1
            return
        if self._drop_depth:
            return
        if tag in _HEADING:
            self._newline()
            self.out.append(_HEADING[tag] + " ")
        elif tag == "li":
            self._newline(1)
            self.out.append("- ")
        elif tag == "br":
            self._newline(1)
        elif tag in ("strong", "b"):
            self.out.append("**")
        elif tag in _BLOCK:
            self._newline()

    def handle_endtag(self, tag):
        if tag in _DROP:
            self._drop_depth = max(0, self._drop_depth - 1)
            return
        if self._drop_depth:
            return
        if tag in _HEADING or tag in _BLOCK:
            self._newline()
        elif tag in ("strong", "b"):
            self.out.append("**")

    def handle_data(self, data):
        if not self._drop_depth:
            self.out.append(re.sub(r"\s+", " ", data))


def html_to_markdown(value: str | None, *, unescape_first: bool = False) -> str:
    """Convert job-ad HTML to markdown. `unescape_first` for entity-encoded HTML (Greenhouse)."""
    if not value:
        return ""
    src = html.unescape(value) if unescape_first else value
    conv = _Converter()
    conv.feed(src)
    conv.close()
    text = "".join(conv.out)
    text = re.sub(r"\*\*\s*\*\*", "", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
