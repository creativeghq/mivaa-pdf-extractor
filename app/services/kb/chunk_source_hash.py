"""Hash of the text a kb_doc's chunks were built from. Import-free so tests can load it by path."""

import hashlib
from typing import Optional


def source_hash(title: Optional[str], content: Optional[str]) -> str:
    """Mirror of SQL public.kb_doc_chunk_source_hash(title, content); the two must agree byte for byte."""
    return hashlib.md5(f"{title or ''}\n{content or ''}".encode("utf-8")).hexdigest()
