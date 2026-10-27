import asyncio

import importlib.util
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app" / "services" / "products" / "enrichment_promote.py"
_spec = importlib.util.spec_from_file_location("enrichment_promote", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

claims_to_patch, finished_status, promote_attempt = _mod.claims_to_patch, _mod.finished_status, _mod.promote_attempt


class FakeIO:
    def __init__(self, status="extracted", product_status="draft", claims=None, empty=None, canon=True, embed=True):
        self.attempt = {
            "id": "a1", "product_id": "p1", "workspace_id": "w1", "status": status,
            "claims": claims if claims is not None else [{"field": "slip_rating", "value": "R10"}],
            "source_url": "https://x.example/p", "confidence": 0.9, "promoted_fields": [],
        }
        self.product = {"id": "p1", "workspace_id": "w1", "status": product_status}
        self.empty = empty if empty is not None else ["slip_rating"]
        self.canon, self.embed_ok = canon, embed
        self.calls = []

    async def load_attempt(self, _id): return dict(self.attempt)
    async def load_product(self, _id): return dict(self.product)
    async def still_empty(self, *_): return list(self.empty)

    async def merge(self, _id, patch, source):
        self.calls.append(("merge", dict(patch), source["source"]))
        return list(patch)

    async def save_attempt(self, _id, fields):
        self.attempt.update(fields)
        self.calls.append(("save", fields.get("status")))

    async def canonicalize(self, _p):
        self.calls.append(("canon",))
        return self.canon

    async def embed(self, _p):
        self.calls.append(("embed",))
        if not self.embed_ok:
            raise RuntimeError("voyage down")
        return True


def run(io):
    return asyncio.run(promote_attempt(io, "a1"))


def test_promotes_then_rebuilds_then_is_done():
    io = FakeIO()
    out = run(io)
    assert out["status"] == "done" and out["promoted_fields"] == ["slip_rating"]
    assert [c[0] for c in io.calls if c[0] != "save"] == ["merge", "canon", "embed"]
    assert io.attempt["canonicalization_state"] == "ok" and io.attempt["embedding_state"] == "ok"


def test_a_failed_embedding_keeps_the_promotion_and_a_retry_repeats_only_the_derived_steps():
    io = FakeIO(embed=False)
    assert run(io)["status"] == "promoted" and io.attempt["embedding_state"] == "failed"
    io.embed_ok = True
    io.calls.clear()
    assert run(io)["status"] == "done"
    assert [c[0] for c in io.calls if c[0] != "save"] == ["canon", "embed"]


def test_degraded_canonicalization_is_not_done():
    io = FakeIO(canon=False)
    assert run(io)["status"] == "promoted" and io.attempt["canonicalization_state"] == "degraded"


def test_a_field_filled_since_extraction_is_not_overwritten():
    io = FakeIO(empty=[])
    out = run(io)
    assert out["status"] == "done" and out["promoted_fields"] == []
    assert not any(c[0] in ("merge", "canon", "embed") for c in io.calls)


def test_a_published_product_is_left_alone():
    io = FakeIO(product_status="active")
    assert run(io)["status"] == "failed"
    assert not any(c[0] == "merge" for c in io.calls)


def test_a_finished_attempt_is_returned_as_is():
    io = FakeIO(status="done")
    assert run(io)["status"] == "done" and io.calls == []


def test_patch_and_status_rules():
    assert claims_to_patch([{"field": "a", "value": 1}, {"field": "b", "value": 2}], ["b"]) == {"b": 2}
    assert finished_status("ok", "ok") == "done" and finished_status("ok", "failed") == "promoted"
