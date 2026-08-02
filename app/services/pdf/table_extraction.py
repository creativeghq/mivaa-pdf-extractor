"""
Table Extraction Service — VLM-native.

Tables are recognized by PaddleOCR-VL during the Stage 1 structural pass and
persisted as markdown/HTML on every TABLE layout element
(``document_layout_analysis.layout_elements[].metadata.html``). This module
parses that cached content into rows and writes it to ``product_tables``, which
:class:`~app.services.metadata.table_metadata_extractor.TableMetadataExtractor`
then mines for dimensions / packaging / performance specs during Stage 5.

Layering: silver → gold. The VLM already read every table off the page, so this
NEVER re-derives tables from the PDF. The previous pdfplumber implementation did
exactly that, was never called by any stage (dead since #248), and failed on the
borderless spec tables that dominate material catalogs.

Features:
- Parses both markdown pipe tables and ``<table>`` HTML (the VLM emits either)
- Classifies table type (multilingual — catalogs are IT/ES/FR as often as EN)
- Stores in product_tables, idempotently per product
"""

import logging
import re
from html import unescape
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

#: A grid needs at least a header row + one data row, and 2 columns, to carry
#: anything TableMetadataExtractor can use.
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

#: Markdown alignment row: |---|:--:|---:|
_MD_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")

#: Header keywords, shared with TableMetadataExtractor so classification and
#: parsing agree on what a dimensions / packaging table looks like. Multilingual
#: on purpose — material catalogs are as often IT/ES/FR as EN, and an
#: English-only classifier routed those tables to the generic parser, throwing
#: away column mappings the parser already understood.
DIMENSION_KEYWORDS = (
    'dimension', 'dimensions', 'dimensioni', 'dimensione',
    'size', 'sizes', 'misura', 'misure', 'medida', 'medidas', 'taille',
    'format', 'formato', 'formats', 'formati',
    'thickness', 'spessore', 'espesor', 'épaisseur', 'epaisseur', 'stärke', 'starke',
    'width', 'height', 'length', 'diameter',
    'larghezza', 'altezza', 'lunghezza', 'ancho', 'alto', 'largo',
)

PACKAGING_KEYWORDS = (
    'pcs/box', 'pcs/', 'pieces', 'pezzi', 'piezas', 'pièces',
    'box', 'boxes', 'carton', 'scatola', 'scatole', 'caja', 'cajas',
    'pallet', 'bancale', 'palet',
    'coverage', 'mq', 'm2', 'm²', 'sqm',
    'weight', 'peso', 'poids', 'gewicht', 'kg',
)

PRICING_KEYWORDS = (
    'price', 'prezzo', 'precio', 'prix', 'preis',
    'cost', 'costo', 'rate', 'pricing', 'quote', 'msrp', 'listino',
)


class _TableHTMLParser(HTMLParser):
    """Collect ``<tr>``/``<td>``/``<th>`` text into a row grid.

    Uses the stdlib parser rather than a regex sweep (or a new bs4/lxml
    dependency) — VLM-generated table HTML is well-formed but does use nested
    inline tags, which a naive regex would mangle.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None:
            if self._row is None:
                self._row = []
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        # Tolerate an unclosed trailing <tr>.
        if self._row:
            self.rows.append(self._row)
            self._row = None


def parse_html_table(content: str) -> Optional[List[List[str]]]:
    """Parse ``<table>`` HTML into a row grid. ``None`` when nothing usable."""
    try:
        parser = _TableHTMLParser()
        parser.feed(content)
        parser.close()
    except Exception as e:  # noqa: BLE001 — malformed VLM output must not abort a job
        logger.debug(f"   HTML table parse failed: {e}")
        return None
    return _normalize_grid(parser.rows)


def parse_markdown_table(content: str) -> Optional[List[List[str]]]:
    """Parse a markdown pipe table into a row grid. ``None`` when nothing usable."""
    rows: List[List[str]] = []
    for line in content.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        cells = [unescape(c.strip()) for c in line.strip("|").split("|")]
        non_empty = [c for c in cells if c]
        # Drop the |---|:--:| alignment row — it is syntax, not data.
        if non_empty and all(_MD_SEPARATOR_CELL.match(c) for c in non_empty):
            continue
        rows.append(cells)
    return _normalize_grid(rows)


def parse_table_content(content: str) -> Optional[List[List[str]]]:
    """Parse VLM-recognized table content (markdown OR HTML) into a row grid.

    The layout element field is named ``html`` but PaddleOCR-VL emits markdown
    for most tables, so dispatch on the content itself rather than the name.
    Returns ``None`` when the content yields nothing a consumer could use —
    callers skip the region rather than persisting an empty table.
    """
    if not content or not content.strip():
        return None
    lowered = content.lower()
    if "<table" in lowered or "<tr" in lowered:
        return parse_html_table(content)
    if "|" in content:
        return parse_markdown_table(content)
    return None


def _normalize_grid(rows: List[List[str]]) -> Optional[List[List[str]]]:
    """Drop blank rows, pad ragged rows to a rectangle, enforce the minimums."""
    cleaned = [
        [str(c).strip() for c in row]
        for row in rows
        if any(str(c).strip() for c in row)
    ]
    if len(cleaned) < MIN_TABLE_ROWS:
        return None
    width = max(len(r) for r in cleaned)
    if width < MIN_TABLE_COLS:
        return None
    return [r + [""] * (width - len(r)) for r in cleaned]


class TableExtractor:
    """Persist PaddleOCR-VL–recognized tables into ``product_tables``."""

    def __init__(self):
        """Initialize table extractor."""
        self.logger = logger

    async def persist_tables_from_layout_cache(
        self,
        document_id: str,
        product_id: str,
        physical_pages: List[int],
        supabase: Any,
        logger: Optional[logging.Logger] = None,
    ) -> int:
        """Parse this product's cached TABLE regions and store them.

        Reads the same ``document_layout_analysis`` cache Stage 2 chunking uses,
        so no PDF is reopened and no inference is re-run — the table content was
        recognized once, during the Stage 1 structural pass.

        Idempotent: every row is fully derived from the cache, so this product's
        existing rows are cleared first. Safe to call on resume, and it MUST be
        called on resume — a job that ran before this stage existed has chunks
        but no tables.

        Returns the number of tables stored.
        """
        log = logger or self.logger
        if not document_id or not product_id or not physical_pages:
            return 0

        # Local import: keeps app.services.pdf free of an app.api dependency at
        # module load (the codebase's established pattern for stage helpers).
        from app.api.pdf_processing.stage_1_layout_precompute import (
            get_layout_from_document_cache_with_status,
        )

        try:
            cached = await get_layout_from_document_cache_with_status(
                document_id=document_id,
                physical_pages=sorted({int(p) for p in physical_pages}),
                supabase=supabase,
                logger=log,
            )
        except Exception as e:  # noqa: BLE001 — never fail the product on this
            log.debug(f"   layout cache read for tables failed (non-fatal): {e}")
            return 0

        regions_seen = 0
        tables: List[Dict[str, Any]] = []
        for page_number, entry in sorted(cached.items()):
            for region in (entry.get("regions") or []):
                if not isinstance(region, dict):
                    continue
                if str(region.get("region_type") or "").upper() != "TABLE":
                    continue
                regions_seen += 1
                content = ((region.get("metadata") or {}).get("html") or "")
                grid = parse_table_content(content)
                if not grid:
                    continue
                tables.append(
                    self._convert_table_to_dict(
                        table=grid,
                        page_number=int(page_number),
                        confidence=float(region.get("confidence") or 0.0),
                    )
                )

        if regions_seen and not tables:
            # Explicit marker, not a silent empty return: the detector found
            # tables and every one of them failed to parse — that is a bug in
            # the parser or a change in the VLM's output format, not "no tables".
            log.warning(
                f"   ⚠️ {regions_seen} TABLE region(s) detected but none parsed "
                f"for product {product_id} — check the VLM table content format"
            )
            return 0
        if not tables:
            log.debug(f"   No TABLE regions on this product's pages ({product_id})")
            return 0

        try:
            supabase.client.table('product_tables') \
                .delete() \
                .eq('product_id', product_id) \
                .execute()
        except Exception as e:  # noqa: BLE001
            log.debug(f"   product_tables pre-clean failed (non-fatal): {e}")

        stored = await self.store_tables_in_database(product_id, tables, supabase)
        log.info(
            f"   📊 Stored {stored}/{len(tables)} table(s) from "
            f"{regions_seen} TABLE region(s) for product {product_id}"
        )
        return stored

    def _convert_table_to_dict(
        self,
        table: List[List[str]],
        page_number: int,
        layout_region_id: Optional[str] = None,
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Convert a parsed row grid to the structured dictionary format.

        Args:
            table: Row grid (list of lists), first row treated as headers
            page_number: Page number
            layout_region_id: Optional layout region ID. Always None for the
                VLM path — layout elements are JSONB array entries, not rows,
                so there is no UUID to reference (the column is nullable).
            confidence: Region detection confidence

        Returns:
            Structured table dictionary
        """
        # First row is the header row
        headers = table[0] if len(table) > 0 else []

        # Extract data rows (skip header row)
        data_rows = table[1:] if len(table) > 1 else []

        # Create structured table data
        table_data = {
            'headers': headers,
            'rows': data_rows,
            'num_rows': len(data_rows),
            'num_cols': len(headers)
        }

        return {
            'page_number': page_number,
            'layout_region_id': layout_region_id,
            'table_data': table_data,
            'headers': headers,
            'confidence': confidence,
            'extractor': 'paddleocr-vl',
            'metadata': {}
        }

    def classify_table_type(self, headers: List[str], table_data: Dict[str, Any]) -> str:
        """
        Classify table type based on headers and content.

        Keywords are multilingual on purpose: material catalogs are as often
        Italian / Spanish / French as English, and TableMetadataExtractor's
        column mappings already understand `formato` / `spessore` / `espesor`.
        An English-only classifier routed those tables to the generic parser and
        threw the column mapping away.

        Args:
            headers: Table headers
            table_data: Table data dictionary

        Returns:
            Table type: 'specifications', 'packaging', 'pricing', 'comparison',
            'dimensions', or 'other'
        """
        # Convert headers to lowercase for matching
        headers_lower = [str(h).lower() for h in headers]
        headers_text = ' '.join(headers_lower)

        # Pricing first — a price column is unambiguous and disqualifying.
        if any(keyword in headers_text for keyword in PRICING_KEYWORDS):
            return 'pricing'

        # Dimensions before packaging: the canonical catalog table carries BOTH
        # (Formato | Spessore | Pz/Scatola | Mq/Scatola | Kg/Scatola), and the
        # product's own dimensions are the more specific label. The consumer
        # runs the packaging parser independently of this label, so nothing is
        # lost — see TableMetadataExtractor.extract_metadata_from_tables.
        if any(keyword in headers_text for keyword in DIMENSION_KEYWORDS):
            return 'dimensions'

        if any(keyword in headers_text for keyword in PACKAGING_KEYWORDS):
            return 'packaging'

        # Specifications table keywords
        if any(keyword in headers_text for keyword in [
            'specification', 'specifiche', 'especificacion', 'spécification',
            'property', 'properties', 'proprieta', 'proprietà', 'propiedad',
            'feature', 'caratteristica', 'caracteristica',
            'characteristic', 'parameter', 'parametro', 'norm', 'norma', 'standard',
        ]):
            return 'specifications'

        # Comparison table keywords
        if any(keyword in headers_text for keyword in [
            'comparison', 'versus', 'compare', 'confronto', 'model', 'modello',
        ]):
            return 'comparison'

        # Default to 'other'
        return 'other'

    async def store_tables_in_database(
        self,
        product_id: str,
        tables: List[Dict[str, Any]],
        supabase_client: Any
    ) -> int:
        """
        Store extracted tables in product_tables database table.

        Args:
            product_id: Product ID
            tables: List of extracted table dictionaries
            supabase_client: Supabase client instance

        Returns:
            Number of tables stored
        """
        if not tables or len(tables) == 0:
            return 0

        try:
            stored_count = 0

            for table in tables:
                # Classify table type
                table_type = self.classify_table_type(
                    headers=table.get('headers', []),
                    table_data=table.get('table_data', {})
                )

                # Prepare record for database
                table_record = {
                    'product_id': product_id,
                    'page_number': table['page_number'],
                    'layout_region_id': table.get('layout_region_id'),
                    'table_data': table['table_data'],
                    'headers': table['headers'],
                    'table_type': table_type,
                    'confidence': table.get('confidence', 0.0),
                    'extractor': table.get('extractor', 'paddleocr-vl'),
                    'metadata': table.get('metadata', {})
                }

                # Insert into database
                result = supabase_client.client.table('product_tables')\
                    .insert(table_record)\
                    .execute()

                if result.data:
                    stored_count += 1
                    self.logger.debug(f"   ✅ Stored {table_type} table from page {table['page_number']}")

            return stored_count

        except Exception as e:
            self.logger.error(f"❌ Failed to store tables in database: {e}")
            return 0
