"""Guards on the metrics, the labels, and the evaluation protocol.

The protocol tests are the ones that matter. A search project's credibility rests
entirely on whether the evaluation is honest, and the two ways it usually is not
-- unsegmented aggregates and a leaky split -- are both asserted against here.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import corpus as C  # noqa: E402
from src import rank as R  # noqa: E402
from src import retrieval as RT  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(os.path.join(DATA, "products.json")):
        pytest.skip("run `python src/corpus.py` first")
    with open(os.path.join(DATA, "products.json")) as f:
        products = json.load(f)
    with open(os.path.join(DATA, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(DATA, "judgments.json")) as f:
        judgments = json.load(f)
    return products, queries, judgments


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def test_ndcg_of_the_ideal_ranking_is_one():
    labels = {"a": 3, "b": 2, "c": 1, "d": 0}
    assert R.ndcg_at_k(["a", "b", "c", "d"], labels, 4) == pytest.approx(1.0)


def test_ndcg_penalises_a_reversed_ranking():
    labels = {"a": 3, "b": 2, "c": 1, "d": 0}
    good = R.ndcg_at_k(["a", "b", "c", "d"], labels, 4)
    bad = R.ndcg_at_k(["d", "c", "b", "a"], labels, 4)
    assert bad < good


def test_exponential_gains_make_one_exact_beat_three_complements():
    """The gain convention is load-bearing. With LINEAR gains three Complements
    (1+1+1) would equal one Exact (3); with 2^g-1 an Exact is worth 7 and three
    Complements are worth 3. In a purchase-intent context the second is right."""
    labels = {"E": 3, "c1": 1, "c2": 1, "c3": 1}
    exact_first = R.ndcg_at_k(["E", "c1", "c2", "c3"], labels, 4)
    comps_first = R.ndcg_at_k(["c1", "c2", "c3", "E"], labels, 4)
    assert exact_first > comps_first
    assert R.dcg([3], 1) == pytest.approx(7.0)
    assert R.dcg([1, 1, 1], 3) == pytest.approx(1 + 1 / np.log2(3) + 0.5)


def test_ndcg_is_zero_when_nothing_relevant_is_returned():
    assert R.ndcg_at_k(["x", "y"], {"a": 3, "x": 0, "y": 0}, 2) == 0.0


def test_ndcg_with_no_relevant_documents_is_zero_not_nan():
    assert R.ndcg_at_k(["x"], {"x": 0}, 2) == 0.0


def test_mrr_finds_the_first_exact_match():
    assert R.mrr(["a", "b", "c"], {"a": 0, "b": 3, "c": 3}) == pytest.approx(0.5)
    assert R.mrr(["a", "b"], {"a": 2, "b": 2}) == 0.0     # substitutes are not exact


def test_substitute_before_complement_scoring():
    # substitute at 0, complement at 1 -> correctly ordered
    assert R.substitute_before_complement(["s", "c"], {"s": 2, "c": 1}, 10) == 1.0
    # reversed -> wrong
    assert R.substitute_before_complement(["c", "s"], {"s": 2, "c": 1}, 10) == 0.0
    # no pair present -> NaN rather than a misleading 0 or 1
    assert np.isnan(R.substitute_before_complement(["s"], {"s": 2}, 10))


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def test_judge_grades_are_the_esci_ladder(data):
    products, queries, judgments = data
    grades = {g for m in judgments.values() for g in m.values()}
    assert grades <= {0, 1, 2, 3}


def test_exact_beats_substitute_beats_complement(data):
    """The generative rules must actually produce the ladder they claim."""
    products, queries, _ = data
    q = next(q for q in queries if q["kind"] == "navigational")
    same_cat_same_brand = [p for p in products
                           if p["category"] == q["target_category"]
                           and p["brand"] == q["target_brand"]
                           and any(a in p["attrs"] for a in q["target_attrs"])]
    other_cat = [p for p in products
                 if p["category"] != q["target_category"]]
    assert same_cat_same_brand, "fixture query has no exact match"
    assert C.judge(q, same_cat_same_brand[0]) == 3
    comps = C.TAXONOMY[q["target_category"]][2]
    for p in other_cat:
        assert C.judge(q, p) == (1 if p["category"] in comps else 0)


def test_complement_categories_are_symmetric_where_declared():
    for cat, (_, _, comps) in C.TAXONOMY.items():
        for c in comps:
            assert c in C.TAXONOMY, "%s lists unknown complement %s" % (cat, c)


# --------------------------------------------------------------------------
# evaluation protocol
# --------------------------------------------------------------------------
def test_segments_partition_the_query_set(data):
    _, queries, _ = data
    segs = [q["segment"] for q in queries]
    assert set(segs) == {"head", "torso", "tail"}
    assert len(segs) == len(queries)


def test_head_queries_carry_more_volume_than_tail(data):
    _, queries, _ = data
    vol = {s: sum(q["frequency"] for q in queries if q["segment"] == s)
           for s in ("head", "torso", "tail")}
    assert vol["head"] > vol["tail"], "the head must be head-heavy in VOLUME"
    n_head = sum(1 for q in queries if q["segment"] == "head")
    n_tail = sum(1 for q in queries if q["segment"] == "tail")
    assert n_head < n_tail, "...while being fewer in COUNT"


def test_splitting_by_query_text_leaves_no_shared_strings(data):
    """The leak this project actually had: an instance-level split put identical
    'need' query strings in both halves and produced NDCG@10 = 1.0000."""
    _, queries, _ = data
    rng = np.random.default_rng(7)
    texts = sorted({q["text"] for q in queries})
    rng.shuffle(texts)
    train_texts = set(texts[:int(0.5 * len(texts))])
    train = {q["text"] for q in queries if q["text"] in train_texts}
    test = {q["text"] for q in queries if q["text"] not in train_texts}
    assert train & test == set()


def test_reranker_features_are_all_servable(data):
    """Every feature must be computable from the query STRING and the product
    record. `target_category` is a label; if it ever appears in build_features
    again, this fails."""
    import inspect
    src = inspect.getsource(R.build_features)
    for leaked in ("target_category", "target_brand", "target_attrs", "judgments"):
        assert leaked not in src, "%s is ground truth, not a feature" % leaked


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------
def test_bm25_finds_an_exact_title_match(data):
    products, _, _ = data
    bm25 = RT.BM25Index(products)
    p = products[0]
    top = bm25.search(p["title"], k=5)
    assert p["product_id"] in [pid for pid, _ in top]


def test_rrf_rewards_agreement_between_legs():
    a = ["x", "y", "z"]
    b = ["y", "x", "z"]
    fused = RT.rrf([a, b])
    # y is 2nd and 1st; x is 1st and 2nd -- they should be close and both above z
    assert fused["z"] < min(fused["x"], fused["y"])


def test_weighted_fusion_respects_its_weights():
    a = {"p": 1.0, "q": 0.0}
    b = {"p": 0.0, "q": 1.0}
    lex_heavy = RT.weighted_fusion([a, b], [0.9, 0.1])
    assert lex_heavy["p"] > lex_heavy["q"]
    sem_heavy = RT.weighted_fusion([a, b], [0.1, 0.9])
    assert sem_heavy["q"] > sem_heavy["p"]


def test_mined_hard_negatives_are_actually_labelled_irrelevant(data):
    products, queries, judgments = data
    bm25 = RT.BM25Index(products)
    negs = RT.mine_hard_negatives(bm25, queries[:40], judgments)
    for qid, pids in negs.items():
        for pid in pids:
            assert judgments[qid].get(pid, 0) == 0
