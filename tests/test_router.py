"""Tests for the second tranche: router, boosting, spelling, interleaving, debug."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import rank as R  # noqa: E402
from src import router as RO  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture(scope="module")
def data():
    if not os.path.exists(os.path.join(DATA, "products.json")):
        pytest.skip("run `python src/corpus.py` first")
    with open(os.path.join(DATA, "products.json")) as f:
        products = json.load(f)
    with open(os.path.join(DATA, "queries.json")) as f:
        queries = json.load(f)
    return products, queries


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------
def test_router_recognises_a_brand_query(data):
    products, _ = data
    r = RO.QueryRouter(products)
    brand = products[0]["brand"]
    noun = products[0]["noun"]
    assert r.classify("%s %s" % (brand, noun)) == "navigational"


def test_router_recognises_a_need_query(data):
    products, _ = data
    r = RO.QueryRouter(products)
    for text in ("shoes for standing all day", "help with lower back pain sitting",
                 "gift for a coffee lover"):
        assert r.classify(text) == "need", text


def test_router_beats_chance_on_the_generators_true_kind(data):
    products, queries = data
    r = RO.QueryRouter(products)
    correct = sum(1 for q in queries if r.classify(q["text"]) == q["kind"])
    assert correct / len(queries) > 0.75


def test_router_is_deterministic(data):
    products, queries = data
    r = RO.QueryRouter(products)
    for q in queries[:50]:
        assert r.classify(q["text"]) == r.classify(q["text"])


def test_weights_are_bounded_and_per_type(data):
    products, _ = data
    r = RO.QueryRouter(products)
    for t in ("navigational", "attribute", "need"):
        assert 0.0 <= r.weights[t] <= 1.0
    assert r.weights["navigational"] != r.weights["need"], \
        "a router whose weights are all equal is not routing"


def test_route_returns_a_ranking(data):
    products, _ = data
    r = RO.QueryRouter(products)
    ids = [p["product_id"] for p in products[:20]]
    bm = {pid: float(i) for i, pid in enumerate(ids)}
    ds = {pid: float(len(ids) - i) for i, pid in enumerate(ids)}
    out = r.route("waterproof work boot", bm, ds, k=5)
    assert len(out) == 5
    assert len(set(out)) == 5


def test_lexical_weight_one_reproduces_the_lexical_ordering(data):
    products, _ = data
    r = RO.QueryRouter(products, weights={"navigational": 1.0, "attribute": 1.0,
                                          "need": 1.0})
    ids = [p["product_id"] for p in products[:10]]
    bm = {pid: float(10 - i) for i, pid in enumerate(ids)}
    ds = {pid: float(i) for i, pid in enumerate(ids)}
    assert r.route("anything at all here", bm, ds, k=3) == ids[:3]


# --------------------------------------------------------------------------
# boosting
# --------------------------------------------------------------------------
def test_boost_promotes_the_brand_and_audits_it(data):
    products, _ = data
    prod_by_id = {p["product_id"]: p for p in products}
    brand = products[0]["brand"]
    rule = RO.BoostRule("R1", brand, "shoe", target_positions=2)
    branded = [p["product_id"] for p in products
               if p["brand"] == brand][:3]
    others = [p["product_id"] for p in products
              if p["brand"] != brand][:10]
    ranked = others + branded
    out, audit = rule.apply("running shoe", ranked, prod_by_id)
    assert out[:2] == branded[:2]
    assert audit["fired"] is True
    assert audit["rule_id"] == "R1"
    assert audit["promoted"] == branded[:2]
    assert set(out) == set(ranked), "boosting must reorder, never add or drop"


def test_boost_does_not_fire_on_non_matching_queries(data):
    products, _ = data
    prod_by_id = {p["product_id"]: p for p in products}
    rule = RO.BoostRule("R1", products[0]["brand"], "shoe")
    ranked = [p["product_id"] for p in products[:5]]
    out, audit = rule.apply("coffee maker", ranked, prod_by_id)
    assert out == ranked
    assert audit == {}


def test_boost_records_a_reason_when_the_brand_is_absent(data):
    products, _ = data
    prod_by_id = {p["product_id"]: p for p in products}
    rule = RO.BoostRule("R1", "NoSuchBrand", "shoe")
    ranked = [p["product_id"] for p in products[:5]]
    out, audit = rule.apply("running shoe", ranked, prod_by_id)
    assert out == ranked
    assert audit["fired"] is False and audit["reason"]


# --------------------------------------------------------------------------
# spelling
# --------------------------------------------------------------------------
def test_speller_leaves_in_vocabulary_tokens_alone(data):
    products, queries = data
    sp = RO.SpellCorrector(products, queries)
    text = products[0]["title"].lower()
    corrected, changes = sp.correct(text)
    assert changes == []
    assert corrected == text


def test_speller_fixes_a_single_deletion(data):
    products, queries = data
    sp = RO.SpellCorrector(products, queries)
    # find a long vocabulary word and delete a character
    word = next(w for w in sp.vocab if len(w) >= 8)
    typo = word[:3] + word[4:]
    corrected, changes = sp.correct(typo)
    if changes:
        assert changes[0][0] == typo
        assert sp._within_one(typo, changes[0][1])


def test_speller_refuses_ambiguous_corrections():
    """Two in-vocabulary candidates at distance one means no correction. An
    aggressive corrector that guesses changes the user's intent and produces
    confidently wrong results."""
    fake_products = [
        dict(title="cat", category="a", attrs=[], brand="b", color="c",
             noun="cat", product_id="1"),
        dict(title="bat", category="a", attrs=[], brand="b", color="c",
             noun="bat", product_id="2"),
    ]
    sp = RO.SpellCorrector(fake_products)
    corrected, changes = sp.correct("xat")
    assert changes == []
    assert corrected == "xat"


def test_edit_distance_helper():
    f = RO.SpellCorrector._within_one
    assert f("cat", "cat")
    assert f("cat", "bat")          # substitution
    assert f("cat", "cats")         # insertion
    assert f("cats", "cat")         # deletion
    assert not f("cat", "dog")
    assert not f("cat", "catsup")


# --------------------------------------------------------------------------
# interleaving
# --------------------------------------------------------------------------
def test_interleaving_draws_from_both_rankers():
    a = ["a1", "a2", "a3", "a4", "a5"]
    b = ["b1", "b2", "b3", "b4", "b5"]
    il, src = RO.team_draft_interleave(a, b, k=6, rng=np.random.default_rng(0))
    assert len(il) == 6
    assert "A" in src and "B" in src
    assert len(set(il)) == len(il), "no duplicates in an interleaved list"


def test_interleaving_is_balanced_within_one():
    """Team draft alternates, so the two teams' counts cannot diverge by more
    than one at any point -- that balance is what makes clicks attributable."""
    a = ["a%d" % i for i in range(20)]
    b = ["b%d" % i for i in range(20)]
    for seed in range(15):
        _il, src = RO.team_draft_interleave(a, b, k=10,
                                            rng=np.random.default_rng(seed))
        assert abs(src.count("A") - src.count("B")) <= 1


def test_shared_documents_are_not_duplicated():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    il, _src = RO.team_draft_interleave(a, b, k=4, rng=np.random.default_rng(1))
    assert len(il) == len(set(il))


def test_a_strictly_better_ranker_wins_the_interleaving():
    """End-to-end sanity: if A puts the relevant documents first and B does not,
    simulated clicks must prefer A."""
    labels = {"good1": 3, "good2": 3, "bad1": 0, "bad2": 0, "bad3": 0}
    a = ["good1", "good2", "bad1"]
    b = ["bad2", "bad3", "bad1"]
    rng = np.random.default_rng(4)
    wins = []
    for _ in range(200):
        il, src = RO.team_draft_interleave(a, b, k=4, rng=rng)
        clicks = RO.simulate_clicks(il, labels, rng)
        wins.append(sum(1 for c in clicks if src[c] == "A")
                    - sum(1 for c in clicks if src[c] == "B"))
    v = RO.interleaving_verdict(wins)
    assert v["preference"] > 0.5
    assert v["p_value"] < 0.05


def test_verdict_handles_all_ties():
    v = RO.interleaving_verdict([0, 0, 0])
    assert v["ties"] == 3
    assert v["p_value"] == 1.0


def test_simulated_clicks_have_position_bias():
    """Position bias is the reason interleaving is needed at all. If the
    simulator has none, the demonstration proves nothing."""
    labels = {"p%d" % i: 3 for i in range(10)}
    ranked = ["p%d" % i for i in range(10)]
    rng = np.random.default_rng(7)
    counts = np.zeros(10)
    for _ in range(2000):
        for c in RO.simulate_clicks(ranked, labels, rng):
            counts[c] += 1
    assert counts[0] > counts[5] > counts[9]


# --------------------------------------------------------------------------
# debug view
# --------------------------------------------------------------------------
def test_occlusion_contributions_cover_every_feature():
    from src import debug as DBG

    class Dummy:
        def predict(self, X):
            return X[:, 0] * 2.0 + X[:, 1]

    # Construct X so features 0 and 1 sit the SAME distance from their medians,
    # isolating the model weight. Occlusion attributes weight x deviation, so a
    # test that ignores the deviation is testing the fixture, not the method --
    # which is what the first version of this test did.
    X = np.zeros((5, len(R.FEATURES)))
    X[:, 0] = [0.0, 0.0, 0.0, 0.0, 1.0]      # median 0, row 4 deviates by 1
    X[:, 1] = [0.0, 0.0, 0.0, 0.0, 1.0]
    contrib = DBG.occlusion_contributions(Dummy(), X, 4)
    assert set(contrib) == set(R.FEATURES)
    assert contrib[R.FEATURES[0]] == pytest.approx(2.0)
    assert contrib[R.FEATURES[1]] == pytest.approx(1.0)
