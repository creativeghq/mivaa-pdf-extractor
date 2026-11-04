"""A monitored subject reports into its agent conversation, and a deleted chat is replaced."""
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "monitoring_conversation", _ROOT / "app/services/utilities/monitoring_conversation.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)


class _Query:
    def __init__(self, db, table):
        self.db, self.table_name, self.op, self.payload = db, table, "select", None

    def select(self, _cols):
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def eq(self, *_):
        return self

    def limit(self, _n):
        return self

    def execute(self):
        rows = self.db.setdefault(self.table_name, [{}])
        if self.op == "insert":
            row = {**self.payload, "id": f"new-{self.table_name}"}
            self.db.setdefault("inserted", []).append((self.table_name, row))
            return type("R", (), {"data": [row]})
        if self.op == "update":
            rows[0].update(self.payload)
        return type("R", (), {"data": rows})


class _Client:
    def __init__(self, db):
        self.db = db

    def table(self, name):
        return _Query(self.db, name)


def _ensure(db):
    return mc.ensure_conversation(
        _Client(db), table="tracked_queries", tracked_id="t1", user_id="u1",
        title_prefix="Price monitoring", label_columns=("search_query",),
    )


def test_reuses_the_existing_thread():
    db = {"tracked_queries": [{"source_conversation_id": "c1", "search_query": "tile"}]}
    assert _ensure(db) == "c1"
    assert "inserted" not in db


def test_a_deleted_chat_opens_a_new_thread_and_relinks():
    db = {"tracked_queries": [{"source_conversation_id": None, "search_query": "tile"}]}
    assert _ensure(db) == "new-agent_chat_conversations"
    assert db["tracked_queries"][0]["source_conversation_id"] == "new-agent_chat_conversations"
    table, row = db["inserted"][0]
    assert table == "agent_chat_conversations"
    assert row["agent_id"] == "kai"
    assert row["title"] == "Price monitoring — tile"


def test_alert_link_carries_what_a_replacement_chat_needs():
    path = mc.alert_path("c1", "price", "t1")
    assert path.startswith("/agent-hub?")
    assert "conversation=c1" in path
    assert "alert=price%3At1" in path


def test_posted_alert_carries_the_details_and_only_a_real_source_link():
    db = {"agent_chat_conversations": [{"message_count": 2}]}
    assert mc.post_alert(
        _Client(db), conversation_id="c1", title="Price drop", body="Was 10, now 8",
        kind="price", tracked_id="t1", source_url="/agent-hub?agent=kai",
    )
    _, msg = db["inserted"][0]
    assert msg["role"] == "assistant"
    assert "Price drop" in msg["content"] and "Was 10, now 8" in msg["content"]
    assert "Open source" not in msg["content"]
    assert db["agent_chat_conversations"][0]["message_count"] == 3


def test_email_links_are_absolute():
    assert mc.absolute_url("/agent-hub?agent=kai").startswith("http")
    assert mc.absolute_url("https://outlet.example/a") == "https://outlet.example/a"
