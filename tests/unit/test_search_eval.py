import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "app/evaluation/search_eval.py"
_spec = importlib.util.spec_from_file_location("search_eval_under_test", _PATH)
se = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = se
_spec.loader.exec_module(se)


def _cand(pid, **scores):
    return {"id": pid, "name": pid, **{c: scores.get(c, 0.0) for c in se.CHANNELS}}


W = {"visual": 0.5, "understanding": 0.3, "keyword": 0.2}


def test_active_mean_lets_a_weak_extra_match_drag_a_strong_product_down():
    strong = _cand("strong", visual=0.9, understanding=0.85, keyword=0.1)
    single = _cand("single", understanding=0.8)
    assert se.fuse_active_mean([strong, single], W)[0] == "single"
    assert se.fuse_weighted_sum([strong, single], W)[0] == "strong"


def test_rrf_ranks_by_position_in_each_channel():
    a = _cand("a", visual=0.30, understanding=0.20, keyword=0.9)
    b = _cand("b", visual=0.31, understanding=0.21, keyword=0.1)
    assert se.fuse_rrf([a, b], W)[0] == "b"


def test_metrics_count_ranks_and_misses():
    m = se.metrics([1, 3, None, 12])
    assert m["hit@1"] == 0.25 and m["hit@5"] == 0.5 and m["hit@10"] == 0.5
    assert m["mrr"] == round((1 + 1 / 3 + 1 / 12) / 4, 3)


def test_first_hit_uses_any_expected_id():
    assert se.first_hit(["x", "y", "z"], ["z", "y"]) == 2
    assert se.first_hit(["x"], ["q"]) is None
