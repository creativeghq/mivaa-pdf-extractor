"""Category -> default selling unit.

`material_categories.default_unit` is the source. Two hardcoded copies of the coarse map
lived here and in the import service, and both stopped at the original ten categories, so a
category added in admin silently sold in `pcs`. Unit spelling follows src/lib/units.ts,
where `m2` is canonical and `sqm` is only a tolerated alias on read.
"""
from typing import Any, Dict, Optional, Tuple

DEFAULT_UNIT = 'pcs'

#: A vocabulary value whose unit differs from its own category's default. Everything else
#: takes the category default straight from the table, so this stays short by construction:
#: a worktop is sold by area although `general_materials` is `pcs`, a wall panel by the piece
#: although `paint_wall_decor` is `m2`.
FINE_UNIT_OVERRIDES: Dict[str, str] = {
    'wall_panel': 'pcs',
    'countertop': 'm2',
    'kitchen_worktop': 'm2',
    'stone_slab': 'm2',
    'metal_panel': 'm2',
    'glass_panel': 'm2',
    'glass_partition': 'm2',
    'cable': 'm',
    'conduit': 'm',
    'trunking': 'm',
    'cable_tray': 'm',
    'led_profile': 'm',
    'pipe': 'm',
    'pipe_insulation': 'm',
    'edge_tape': 'm',
    'hanging_rail': 'm',
    'rebar': 'm',
    'metal_profile': 'm',
    'ceiling_channel': 'm',
    'corner_bead': 'm',
    'plasterboard': 'm2',
    'melamine_board': 'm2',
    'mdf': 'm2',
    'chipboard': 'm2',
    'plywood': 'm2',
    'osb': 'm2',
    'hpl': 'm2',
    'cladding': 'm2',
    'concrete': 'm2',
    'terrazzo': 'm2',
    'quartz': 'm2',
}


def resolve_default_unit(
    material_category: Optional[str],
    category_units: Optional[Dict[str, str]] = None,
    vocab_to_category: Optional[Dict[str, str]] = None,
    default: str = DEFAULT_UNIT,
) -> str:
    """Resolve a coarse key OR a fine vocabulary value onto a selling unit.

    Args:
        material_category: category key, controlled_vocab value or alias.
        category_units: category_key -> default_unit, from material_categories.
        vocab_to_category: controlled_vocab value / alias -> category_key.
        default: unit when nothing resolves.

    Returns:
        The unit string, or `default`. Never guesses by substring: the fuzzy match this
        replaced resolved on any shared substring, so an unrecognised value could take a
        neighbour's unit rather than admitting it did not know.
    """
    if not material_category:
        return default
    cat = str(material_category).lower().strip()
    if not cat:
        return default

    if cat in FINE_UNIT_OVERRIDES:
        return FINE_UNIT_OVERRIDES[cat]

    units = category_units or {}
    if cat in units:
        return units[cat]

    owner = (vocab_to_category or {}).get(cat)
    if owner and owner in units:
        return units[owner]

    return default


def load_category_units(supabase: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Read the category registry. Returns (category_key -> unit, vocab value -> category_key).

    Fails soft to two empty maps: an unavailable registry means `resolve_default_unit` falls
    back to its default rather than to a stale copy of the table.
    """
    units: Dict[str, str] = {}
    vocab: Dict[str, str] = {}
    try:
        client = getattr(supabase, 'client', supabase)
        resp = client.table('material_categories') \
            .select('category_key, default_unit, controlled_vocab, vocab_aliases') \
            .eq('is_active', True).execute()
        for row in (resp.data or []):
            key = (row.get('category_key') or '').lower().strip()
            if not key:
                continue
            unit = row.get('default_unit')
            if unit:
                units[key] = unit
            for term in (row.get('controlled_vocab') or []) + (row.get('vocab_aliases') or []):
                t = str(term or '').lower().strip()
                if t:
                    vocab.setdefault(t, key)
    except Exception:
        return {}, {}
    return units, vocab
