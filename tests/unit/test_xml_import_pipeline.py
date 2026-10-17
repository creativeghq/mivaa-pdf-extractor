"""Guards for the XML import pipeline (#494 B16/B17 and the comment-5 XML findings).

Source-based plus the pure helpers exec'd in isolation: CI installs pytest and nothing else,
so nothing here imports `app`.
"""

import ast
import importlib.util
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _ROOT / "app" / "services" / "integrations" / "data_import_service.py"
_ROUTES = _ROOT / "app" / "api" / "data_import_routes.py"
_TEXT_FOLD = _ROOT / "app" / "utils" / "text_fold.py"


def _blank_comments(src: str) -> str:
    tree = ast.parse(src)
    spans = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            spans.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        "" if i in spans or line.lstrip().startswith("#") else line
        for i, line in enumerate(src.split("\n"), start=1)
    )


def _func(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _blank_comments(ast.get_source_segment(src, node) or "")
    raise AssertionError(f"{name} not found — re-point this guard")


_SRC = _SERVICE.read_text(encoding="utf-8")
_ROUTES_SRC = _ROUTES.read_text(encoding="utf-8")


def _load_text_fold():
    spec = importlib.util.spec_from_file_location("text_fold", _TEXT_FOLD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_parse_price(raw, hint_currency=None):
    hint = hint_currency.strip().upper() if hint_currency and len(hint_currency.strip()) == 3 else None
    if raw is None:
        return None, hint
    stated = 'EUR' if '€' in raw else 'USD' if '$' in raw else 'JPY' if '¥' in raw else None
    digits = re.sub(r'[^\d.]', '', raw.replace(',', '.'))
    try:
        amount = Decimal(digits) if digits else None
    except Exception:
        amount = None
    return amount, stated or hint


def _helpers() -> Dict[str, Any]:
    tree = ast.parse(_SRC)
    wanted = {'ImportRowError', 'source_type_for', 'import_identity_for', 'resolve_money',
              'derive_import_outcome', '_SOURCE_TYPES'}
    ns: Dict[str, Any] = {
        'Decimal': Decimal, 'Optional': Optional, 'Tuple': Tuple, 'Any': Any,
        'fold_model_token': _load_text_fold().fold_model_token,
        'parse_price': _fake_parse_price,
    }
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], 'id', None)
        if name in wanted:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SERVICE), 'exec'), ns)
    return ns


H = _helpers()


class TestSourceTypeStaysInsideTheCheck:
    def test_feed_sources_keep_their_value(self):
        assert H['source_type_for']('xml_import') == 'xml_import'

    @pytest.mark.parametrize('source', ['dealer_manual', 'warehouse_intake', 'manual'])
    def test_manual_paths_record_provenance_elsewhere(self, source):
        # products_source_type_check rejects these; every manual "Add product" was a 500 (B17).
        assert H['source_type_for'](source) is None

    def test_the_insert_writes_provenance_into_created_from_type(self):
        body = _func(_SRC, 'create_product_from_payload')
        assert '"created_from_type": source' in body
        assert '"source_type": source_type_for(source)' in body
        assert "product_record['source_type'] = source" not in body


class TestImportIdentity:
    def test_two_suppliers_with_one_sku_are_two_products(self):
        f = H['import_identity_for']
        assert f('Brand A', 'X1', 'Tile', allow_name_identity=True) != \
            f('Brand B', 'X1', 'Tile', allow_name_identity=True)

    def test_a_greek_lookalike_brand_and_sku_fold_to_one_identity(self):
        f = H['import_identity_for']
        fold = _load_text_fold().fold_model_token
        assert f('KEROS', fold('7012ΜΤ'), None, allow_name_identity=True) == \
            f('Keros', fold('7012-MT'), None, allow_name_identity=True)

    def test_no_sku_in_a_feed_still_has_a_stable_identity(self):
        f = H['import_identity_for']
        assert f('Brand', None, 'Oak Plank 20mm', allow_name_identity=True) == \
            f('Brand', None, 'oak-plank 20 mm', allow_name_identity=True)

    def test_a_manual_add_without_a_sku_never_matches_by_name(self):
        assert H['import_identity_for']('Brand', None, 'Oak', allow_name_identity=False) is None

    def test_the_lookup_uses_the_identity_and_folded_sku_is_stored(self):
        body = _func(_SRC, 'create_product_from_payload')
        assert re.search(r"eq\('import_identity',\s*import_identity\)", body)
        assert "product_record['external_sku_folded'] = external_sku_folded" in body
        assert "product_record['import_identity'] = import_identity" in body


class TestMoney:
    def test_feed_currency_is_a_hint(self):
        assert H['resolve_money']('12.50', 'eur') == (Decimal('12.50'), 'EUR')

    def test_conflicting_currencies_reject_the_row(self):
        with pytest.raises(H['ImportRowError']):
            H['resolve_money']('$12', 'EUR')

    def test_no_currency_anywhere_rejects_rather_than_assuming_eur(self):
        with pytest.raises(H['ImportRowError']):
            H['resolve_money']('12.50', None)

    def test_cost_never_reaches_list_price(self):
        body = _func(_SRC, 'create_product_from_payload')
        assert "product_record['cost'] = float(cost[0])" in body
        price_body = _func(_SRC, '_upsert_import_price')
        assert 'supplier_cost' not in price_body

    def test_a_failed_price_write_is_not_swallowed(self):
        assert 'except' not in _func(_SRC, '_upsert_import_price')


class TestOutcome:
    def test_all_failed_is_failed(self):
        assert H['derive_import_outcome'](0, 5, 0) == ('failed', 'failed')

    def test_some_failed_is_partial(self):
        assert H['derive_import_outcome'](3, 2, 0) == ('completed', 'partial')

    def test_a_quota_skip_is_not_a_success(self):
        assert H['derive_import_outcome'](3, 0, 1) == ('completed', 'partial')

    def test_batch_counts_skips_separately(self):
        body = _func(_SRC, '_process_batch')
        assert 'skipped += 1' in body
        assert "status='skipped'" in body

    def test_resume_seeds_the_counters_from_the_checkpoint(self):
        body = _func(_SRC, 'process_import_job')
        for key in ('batch_index', 'processed', 'failed', 'skipped'):
            assert f"checkpoint.get('{key}')" in body


class TestWritesAreScoped:
    def test_ownership_is_checked_before_the_failure_handler(self):
        body = _func(_SRC, 'process_import_job')
        assert body.index('_get_job(job_id, workspace_id)') < body.index('try:')

    def test_job_updates_carry_the_workspace(self):
        body = _func(_SRC, '_update_job_status')
        assert re.search(r"eq\('workspace_id',\s*workspace_id\)", body)

    def test_images_are_not_linked_through_a_fake_document(self):
        body = _func(_SRC, '_link_images_to_product')
        assert "'document_id': None" in body
        assert 'document_id=product_id' not in body
        assert "'workspace_id': workspace_id" in body

    def test_canonicalizer_gets_the_sync_wrapper_and_degraded_is_marked(self):
        body = _func(_SRC, 'create_product_from_payload')
        assert 'canonicalize_product_attributes(\n            self.supabase_wrapper' in body
        assert "product_metadata['facet_canonicalization']" in body

    def test_metadata_values_survive_absent_top_level_keys(self):
        body = _func(_SRC, 'create_product_from_payload')
        meta_block = body[body.index('product_metadata = {'):]
        meta_block = meta_block[:meta_block.index('}')]
        assert "product_data.get('color')" not in meta_block

    def test_feed_unit_wins_over_the_category_default(self):
        body = _func(_SRC, 'create_product_from_payload')
        guard = body.index('if not feed_unit:')
        assert guard < body.index('feed_unit = _default_unit_for(')

    def test_an_unreadable_category_registry_is_marked_not_resolved_to_pcs(self):
        body = _func(_SRC, 'create_product_from_payload')
        assert 'except CategoryRegistryUnavailable' in body
        assert "product_metadata['_category_registry']" in body

    def test_reimport_never_writes_status(self):
        assert "'status'" not in _SRC[_SRC.index('_SUPPLIER_OWNED_COLUMNS = frozenset'):][:600]


class TestProcessRoute:
    def test_requires_catalog_import(self):
        body = _func(_ROUTES_SRC, 'process_import_job')
        assert '_require_catalog_import(claims, request.workspace_id)' in body

    def test_feed_import_accepts_pricing_manage(self):
        assert '_FEED_IMPORT_CAPABILITIES = frozenset({"catalog.import", "pricing.manage"})' in _ROUTES_SRC
        assert 'r.get("capability") in _FEED_IMPORT_CAPABILITIES' in _func(_ROUTES_SRC, '_require_catalog_import')

    def test_claims_pending_once(self):
        body = _func(_ROUTES_SRC, 'process_import_job')
        assert '.eq("status", "pending")' in body
        assert 'status_code=409' in body
