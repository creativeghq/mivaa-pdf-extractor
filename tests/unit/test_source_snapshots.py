import asyncio
import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("source_snapshots_under_test", _ROOT / "app/services/pdf/source_snapshots.py")
ss = importlib.util.module_from_spec(spec)
sys.modules["source_snapshots_under_test"] = ss
spec.loader.exec_module(ss)

HTML = (
    "<html><head><style>.a{color:red}</style><script>var x = 99999;</script></head><body>"
    "<h1>Comfort 9</h1><p>Panel 2780 x 600 x 9 mm</p><table><tr><td>Weight</td><td>12,5</td></tr></table>"
    '<a href="/catalog/acoustic_panels_1/mila_9/">Mila</a><a href="https://other.example/x">out</a>'
    '<a href="/catalog/acoustic_panels_1/?sort=PRICE&amp;order=asc">sorted</a><a href="/catalog/img/a.jpg">image</a><a href="/about/">about</a><a href="#top">top</a></body></html>'
)


def test_visible_text_skips_scripts_and_styles_and_keeps_table_cells():
    page = ss.parse_page(HTML)
    assert "99999" not in page["text"] and "color:red" not in page["text"]
    assert "2780 x 600 x 9 mm" in page["text"] and "12,5" in page["text"]


def test_only_pages_inside_the_root_tree_are_followed_and_images_are_not():
    links = ss.same_tree_links("https://stella-ist.com/catalog/", ss.parse_page(HTML)["links"], "https://stella-ist.com/catalog/")
    assert links == ["https://stella-ist.com/catalog/acoustic_panels_1/mila_9/"]


def test_the_storage_path_is_content_addressed_under_the_workspace():
    assert ss.storage_path("w1", "abc") == "source-snapshots/w1/abc.html"
    assert ss.sha256(b"x") == ss.sha256(b"x") != ss.sha256(b"y")


class Storage:
    def __init__(self, fail=None):
        self.uploads, self.fail = [], fail

    def from_(self, bucket):
        outer = self

        class Bucket:
            def upload(self, path, body, options):
                if outer.fail:
                    raise RuntimeError(outer.fail)
                outer.uploads.append((bucket, path, options["upsert"]))

        return Bucket()


class Db:
    def __init__(self, fail=None):
        self.storage, self.rows = Storage(fail), []

    def table(self, name):
        db = self

        class Query:
            def upsert(self, row, on_conflict):
                db.rows.append((name, row, on_conflict))
                return self

            def execute(self):
                return types.SimpleNamespace(data=[{"id": "snap-1"}])

        return Query()


def run(db):
    return asyncio.run(ss.store_snapshot(db, workspace_id="w1", document_id="d1", url="https://stella-ist.com/catalog/", final_url="https://stella-ist.com/catalog/",
                                         status=200, content_type="text/html", body=HTML.encode()))


def test_a_snapshot_stores_the_body_privately_and_one_row_keyed_by_content():
    db = Db()
    out = run(db)
    assert out["id"] == "snap-1" and out["sha256"] in db.storage.uploads[0][1]
    assert db.storage.uploads[0][0] == "pdf-documents" and db.storage.uploads[0][2] == "false"
    name, row, conflict = db.rows[0]
    assert name == "source_snapshots" and conflict == "workspace_id,url,content_sha256"
    assert row["document_id"] == "d1" and row["http_status"] == 200 and row["visible_text_chars"] > 0


def test_an_identical_body_already_stored_is_not_an_error_but_any_other_failure_is():
    assert run(Db(fail="The resource already exists"))["id"] == "snap-1"
    try:
        run(Db(fail="bucket is not reachable"))
    except RuntimeError as exc:
        assert "not reachable" in str(exc)
    else:
        raise AssertionError("a real storage failure must not be swallowed")
