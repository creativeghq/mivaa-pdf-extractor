"""What /docs, /redoc and GET / say about the API comes from the code, not from a hand-kept list."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _code(path: str) -> str:
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    return budget.blank_comments((_ROOT / path).read_text(encoding="utf-8"))


def test_the_root_endpoint_lists_no_paths_by_hand():
    src = _code("app/main.py")
    root = src[src.index("async def root("):src.index("from app.api.search import router as search_router")]
    assert "/api/" not in root
    assert "request.app.openapi()" in root


def test_the_schema_version_is_the_one_setting():
    src = _code("app/main.py")
    app = src[src.index("app = FastAPI("):src.index("lifespan=lifespan")]
    assert "version=settings.app_version" in app
    assert "localhost" not in app


def test_a_module_router_keeps_its_own_tag_instead_of_gaining_a_second():
    src = _code("app/modules/__init__.py")
    assert "tags=None if router.tags else" in src


def test_the_schema_carries_no_hand_written_counts_or_claims():
    src = _code("app/main.py")
    for block in ("x-api-features", "x-endpoint-categories", "x-platform-stats"):
        assert block not in src
    assert "dict.fromkeys(operation[\"tags\"])" in src
