"""
Guard: Voyage is the ONLY embedding provider, and nothing may quietly add a second.

WHY THIS EXISTS
---------------
Until 2026-08-08 this service fell back to OpenAI `text-embedding-3-small` when Voyage
failed. That is not a resilience feature — it is a correctness hazard wearing one.

`text-embedding-3-small` at 1024D and `voyage-4` at 1024D are the same SHAPE and a
different SPACE. A fallback vector is therefore accepted everywhere a real one is:
Postgres stores it, HNSW ranks it, cosine returns a confident number. No typecheck, no
dimension check and no integrity probe can see it, because every artifact involved is
individually well-formed. On write paths the damage is durable — mixed-space rows sit
in the collection forever, ranking wrongly.

The old design tried to contain this with a per-call `allow_openai_fallback=False`, and
SEVEN call sites passed it. `generate_understanding_query_embedding` did not — so the
collections most carefully purified on the write side could still be QUERIED with an
OpenAI vector. That is what an opt-out always decays into: it holds until someone adds
the eighth call site. This test replaces the discipline with a structural check.

Source/AST based: imports no app module and touches no DB, so it runs in CI in about a
second.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

#: Every module that produces a vector destined for a shared collection.
_EMBEDDING_MODULES = [
    _APP / "services" / "embeddings" / "real_embeddings_service.py",
    _APP / "services" / "embeddings" / "aspect_backfill.py",
    _APP / "services" / "embeddings" / "understanding_backfill.py",
    _APP / "services" / "embeddings" / "page_embedding_service.py",
]

#: `SearchDeduplicationService` is deliberately absent. It embeds a query only to
#: compare it against OTHER queries in its own dedup cache — it never touches a VECS
#: collection, so it is not in the same latent space at all. If that ever changes it
#: belongs in the list above.
_KNOWN_NON_COLLECTION_EMBEDDER = "search_deduplication_service.py"


def _executable_source(path: Path, text: str | None = None) -> str:
    """Module source with comments and DOCSTRINGS stripped — but other strings kept."""
    tree = ast.parse(text if text is not None else path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body[0].value.value = ""
    # Comments never survive the AST, so unparse drops those for free.
    return ast.unparse(tree)


@pytest.mark.parametrize("path", _EMBEDDING_MODULES, ids=lambda p: p.name)
def test_no_embedding_module_calls_a_second_provider(path):
    """No embedding module may reference another embedding provider in live code."""
    assert path.exists(), f"{path.name} moved — update this guard rather than deleting it"
    source = _executable_source(path).lower()
    for forbidden in ("openai", "api.openai.com", "text-embedding-3", "cohere", "gemini"):
        assert forbidden not in source, (
            f"{path.name} references '{forbidden}' in executable code. Voyage is the "
            f"only embedding provider; a same-dimension vector from a different model "
            f"is accepted silently and ranks as confident nonsense."
        )


def test_the_per_call_opt_out_is_gone():
    """`allow_openai_fallback` must not come back.

    A per-call opt-out is the wrong shape for this: it is correct only while every
    author remembers it, and it already failed that way once.
    """
    offenders = []
    for path in _APP.rglob("*.py"):
        try:
            # Executable source only. The modules that REMOVED this flag explain the
            # removal in prose naming the flag, so a raw-text scan flags the very
            # comment that documents the fix.
            if "allow_openai_fallback" in _executable_source(path):
                offenders.append(path.relative_to(_ROOT).as_posix())
        except SyntaxError:  # pragma: no cover - a broken file is another test's job
            continue
    assert not offenders, (
        f"`allow_openai_fallback` reappeared in {offenders}. The fallback was removed "
        f"outright precisely so no call site has to opt out of it."
    )


def test_nothing_calls_an_openai_embedding_endpoint():
    """The check this file's NAME implies, which it did not previously make."""
    offenders = []
    for path in _APP.rglob("*.py"):
        try:
            src = _executable_source(path)
        except SyntaxError:  # pragma: no cover - a broken file is another test's job
            continue
        rel = path.relative_to(_ROOT).as_posix()
        # The PRICE TABLE is reference data, not a call. `ai_pricing` carries rows for
        # models this platform does not call — that is what a price table is for, and
        # deleting them would only mean an unrecognised model gets costed by a guess.
        if rel.endswith("app/config/ai_pricing.py"):
            continue
        if re.search(r"embeddings\s*\.\s*create\s*\(", src):
            offenders.append(f"{rel}: openai embeddings.create(")
        if re.search(r"text-embedding-3-(small|large)", src):
            offenders.append(f"{rel}: names an OpenAI embedding model")
        if re.search(r"api\.openai\.com/v1/embeddings", src):
            offenders.append(f"{rel}: raw OpenAI embeddings endpoint")
    assert not offenders, (
        "OpenAI embedding call(s) found: " + "; ".join(offenders) + ". Voyage is the only "
        "embedding provider on this platform — two models at one dimension are the same "
        "SHAPE and a different SPACE, so a substituted vector is stored, indexed and "
        "ranked without anything raising."
    )


def test_no_dead_fallback_config_remains():
    """A setting nobody reads is an invitation to wire it back up."""
    config = _executable_source(_APP / "config.py")
    assert "voyage_fallback_to_openai" not in config
    assert "openai_embedding_model" not in config, (
        "a config key naming an embedding model this platform will not call reads as "
        "permission to call it"
    )


#: The ONE place OpenAI may be called, and the only thing it may be called for: the
#: mention probe's chat completions ("what does ChatGPT say about this brand"). The
#: 2026-08-23 removal recorded that answer as the single most valuable one the probe
#: could give; it was restored on 2026-09-05 as a deliberate provider decision — a
#: price row for gpt-5-mini / gpt-5, a key under OPENAI_API_KEY, and this allowlist.
_OPENAI_CHAT_ALLOWED = ("app/services/integrations/llm_mention_probe_service.py",)


def test_openai_is_chat_only_and_lives_in_one_file():
    """OpenAI is a chat provider for the mention probe and NOTHING else."""
    offenders = []
    for path in _APP.rglob("*.py"):
        try:
            src = _executable_source(path)
        except SyntaxError:  # pragma: no cover
            continue
        rel = path.relative_to(_ROOT).as_posix()
        # Reference data, and the guards that REJECT a gpt model, are not usage.
        if rel.endswith(("app/config/ai_pricing.py", "app/utils/supabase_logging_handler.py")):
            continue
        allowed_chat = rel in _OPENAI_CHAT_ALLOWED
        for pattern, what in (
            (r"^\s*import openai", "imports the openai package"),
            (r"from openai import", "imports from the openai package"),
            (r"api\.openai\.com/v1/embeddings", "calls the OpenAI embeddings endpoint"),
        ):
            if re.search(pattern, src, re.M):
                offenders.append(f"{rel}: {what}")
        if not allowed_chat:
            for pattern, what in (
                (r"api\.openai\.com", "calls an OpenAI endpoint outside the mention probe"),
                (r"OPENAI_API_KEY", "reads the OpenAI key outside the mention probe"),
            ):
                if re.search(pattern, src, re.M):
                    offenders.append(f"{rel}: {what}")
    assert not offenders, (
        "OpenAI outside its allowed shape: " + "; ".join(offenders) + ". It is a chat "
        "provider for the mention probe only — httpx, chat completions, one file. Anything "
        "else is the fallback-embedder trap coming back."
    )


def test_the_openai_package_is_not_a_dependency():
    """A package in requirements is a package someone will import."""
    reqs = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    offenders = [
        line for line in reqs.splitlines()
        if line.strip().lower().startswith("openai")
    ]
    assert not offenders, f"openai is back in requirements.txt: {offenders}"


def test_failure_returns_nothing_rather_than_something_else():
    """The contract that replaced the fallback: absence, never a substitute."""
    for path in _EMBEDDING_MODULES:
        tree = ast.parse(_executable_source(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            rendered = ast.unparse(node.value).replace(" ", "")
            # Matched ANYWHERE, not as a prefix: the batch form is
            # `[[0.0] * 1024] * len(texts)`, which does not start with `[0.0]*`.
            # A prefix check passed a hand-run mutation and would have missed the
            # real thing — the batch path is the one where a fabricated vector
            # actually gets persisted.
            assert not re.search(r"\[0(\.0)?\]\*", rendered), (
                f"{path.name} returns a fabricated zero vector ({rendered}). On "
                f"provider failure return None — absence is recoverable and visible; "
                f"a plausible-looking vector is neither."
            )

    tree = ast.parse(
        (_APP / "services" / "embeddings" / "real_embeddings_service.py").read_text(
            encoding="utf-8"
        )
    )
    for name, expected in (
        ("_generate_text_embedding", "None"),
        ("generate_batch_embeddings", "[None] * len(texts)"),
    ):
        node = next(
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        )
        returns = {
            ast.unparse(r.value) for r in ast.walk(node)
            if isinstance(r, ast.Return) and r.value is not None
        }
        assert expected in returns, (
            f"{name} must be able to return {expected} on provider failure; "
            f"found returns: {sorted(returns)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# One latent space: both text defaults must be 4-series (#400 W3)
# ─────────────────────────────────────────────────────────────────────────────

def _field_default(source: str, field: str) -> str:
    """The `default=` literal of a pydantic Field declaration, by AST."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != field or not isinstance(node.value, ast.Call):
            continue
        for kw in node.value.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
    raise AssertionError(f"config.py declares no {field} with a literal default")


def test_both_text_embedding_defaults_are_four_series():
    """voyage-4-large indexes and voyage-4 queries — and that ONLY works because the 4
    series is one shared latent space. Any other prefix is a different space at the same
    1024D, so the wrong vector is stored, indexed and ranked with nothing raising."""
    source = (_APP / "config.py").read_text(encoding="utf-8")
    document = _field_default(source, "voyage_document_model")
    query = _field_default(source, "voyage_query_model")
    alias = _field_default(source, "voyage_model")

    for label, value in (("document", document), ("query", query), ("alias", alias)):
        assert value.startswith("voyage-4"), (
            f"the {label} embedding model defaults to {value!r}. Only the voyage-4 series "
            f"shares a latent space with the corpus; anything else needs a full re-embed, "
            f"not a config change."
        )
    assert alias == document, (
        "voyage_model is the back-compat alias for the document model and every provenance "
        "stamp in real_embeddings_service reads it — letting the two differ makes those "
        "stamps name a model that never ran"
    )


def test_the_two_models_are_priced_apart():
    """voyage-4-large is 2x voyage-4. One price for both understates every document
    embedding by half, which is a wrong number that nothing can raise on."""
    pricing = (_APP / "config" / "ai_pricing.py").read_text(encoding="utf-8")
    assert '"voyage-4-large"' in pricing, "voyage-4-large has no pricing entry"
    service = (_APP / "services" / "embeddings" / "real_embeddings_service.py").read_text(
        encoding="utf-8"
    )
    assert "cost_per_million = 0.06" not in service, (
        "a literal Voyage price is back in the embeddings service; resolve it from "
        "AIPricingConfig so the document model is not billed at the query model's rate"
    )


def test_the_model_is_chosen_by_input_type():
    """A query embedded with the document model is not wrong — same space — but it is
    billed at 2x and recorded under the wrong provenance."""
    service = (_APP / "services" / "embeddings" / "real_embeddings_service.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(service)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "voyage_model_for"
    )
    body = ast.unparse(node)
    assert "voyage_query_model" in body and "voyage_document_model" in body
    assert service.count('"model": self.voyage_model,') == 0, (
        "a Voyage request still pins the document model regardless of input_type"
    )
