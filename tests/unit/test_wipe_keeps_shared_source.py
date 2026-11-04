from types import SimpleNamespace

import pytest

cleanup = pytest.importorskip("app.services.utilities.cleanup_service")

ROW = {"storage_bucket": "pdf-documents", "storage_object_path": "catalogs/x.pdf", "metadata": {}}


class _Query:
    def __init__(self, data=None, error=None):
        self.data, self.error = data, error

    def select(self, *a):
        return self

    def eq(self, *a):
        return self

    def neq(self, *a):
        return self

    def limit(self, *a):
        return self

    def execute(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=self.data)


def _wipe(query):
    removed = []
    bucket = SimpleNamespace(remove=lambda paths: removed.extend(paths))
    client = SimpleNamespace(table=lambda name: query, storage=SimpleNamespace(from_=lambda name: bucket))
    service = cleanup.CleanupService.__new__(cleanup.CleanupService)
    service.logger = SimpleNamespace(info=print, warning=print, error=print)
    service.cleanup_document_storage("doc", SimpleNamespace(client=client), document_row=ROW, source_only=True)
    return removed


def test_a_source_file_another_document_uses_survives_the_wipe():
    assert _wipe(_Query(data=[{"id": "other"}])) == []


def test_an_unshared_source_file_is_removed():
    assert _wipe(_Query(data=[])) == ["catalogs/x.pdf"]


def test_a_failed_lookup_keeps_the_file():
    assert _wipe(_Query(error=RuntimeError("db down"))) == []
