"""Metrics and the learning-to-rank re-ranker.

NDCG is computed with the graded gains the corpus defines (E=3, S=2, C=1, I=0),
using the 2^g - 1 gain convention -- which is not a detail. Linear gains would
make three Complements worth as much as one Exact, and in a purchase-intent
context that is the wrong trade.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def dcg(gains: list[int], k: int) -> float:
    g = np.asarray(gains[:k], float)
    if len(g) == 0:
        return 0.0
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    return float(((2 ** g - 1) * disc).sum())


def ndcg_at_k(ranked_ids: list[str], labels: dict[str, int], k: int = 10) -> float:
    gains = [labels.get(pid, 0) for pid in ranked_ids[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    denom = dcg(ideal, k)
    return dcg(gains, k) / denom if denom > 0 else 0.0


def mrr(ranked_ids: list[str], labels: dict[str, int], threshold: int = 3) -> float:
    """Reciprocal rank of the first EXACT match. Threshold 3 by default because
    'did we find the thing they asked for' is a different question from 'did we
    find something vaguely related'."""
    for i, pid in enumerate(ranked_ids, start=1):
        if labels.get(pid, 0) >= threshold:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: list[str], labels: dict[str, int], k: int,
                threshold: int = 2) -> float:
    rel = {pid for pid, g in labels.items() if g >= threshold}
    if not rel:
        return float("nan")
    return len(rel & set(ranked_ids[:k])) / len(rel)


def substitute_before_complement(ranked_ids: list[str], labels: dict[str, int],
                                 k: int = 10) -> float | float:
    """Retail-specific: for a purchase-intent query, a Substitute must outrank a
    Complement. Someone searching for running shoes wants a different running
    shoe, not socks.

    Returns the fraction of (substitute, complement) pairs in the top-k that are
    correctly ordered; NaN when the top-k contains no such pair.
    """
    subs = [i for i, pid in enumerate(ranked_ids[:k]) if labels.get(pid, 0) == 2]
    comps = [i for i, pid in enumerate(ranked_ids[:k]) if labels.get(pid, 0) == 1]
    if not subs or not comps:
        return float("nan")
    good = sum(1 for s in subs for c in comps if s < c)
    return good / (len(subs) * len(comps))


# --------------------------------------------------------------------------
# re-ranker
# --------------------------------------------------------------------------
# Every feature here must be computable AT SERVING TIME from the query string and
# the product record. An earlier version carried a `category_match` feature built
# from the query's ground-truth target_category -- which is a label, not a
# feature. It would have inflated every number in the report and would not exist
# in production. It is replaced by `category_token_overlap`, which asks the
# legitimate version of the same question: do the category's words appear in the
# query the user actually typed?
FEATURES = ["bm25", "dense", "rrf", "popularity", "price_z", "rating",
            "category_token_overlap", "brand_in_query", "title_token_overlap",
            "exact_noun_match", "query_len"]


class Reranker:
    """LambdaMART is the retail workhorse and LightGBM is not installed here, so
    this is a gradient-boosted REGRESSOR fitted to the graded label.

    That is a pointwise objective standing in for a listwise one, and it is a real
    difference: pointwise fits the label, listwise fits the ORDER, and NDCG only
    cares about order. The substitution is named at every use site and the README
    lists it, because "LambdaMART" would be a nicer word to write and would be
    false.
    """

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingRegressor(
            max_iter=260, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def permutation_importance(self, X: np.ndarray, y: np.ndarray,
                               rng: np.random.Generator, n_repeats: int = 3):
        """Permutation importance rather than split counts: split counts reward
        high-cardinality features regardless of whether they help."""
        base = float(np.mean((self.predict(X) - y) ** 2))
        out = {}
        for j, name in enumerate(FEATURES):
            losses = []
            for _ in range(n_repeats):
                Xp = X.copy()
                rng.shuffle(Xp[:, j])
                losses.append(float(np.mean((self.predict(Xp) - y) ** 2)))
            out[name] = float(np.mean(losses) - base)
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def build_features(query: dict, cand_ids: list[str], bm25_scores: dict,
                   dense_scores: dict, rrf_scores: dict,
                   prod_by_id: dict, price_mean: float, price_sd: float
                   ) -> np.ndarray:
    q_tokens = set(query["text"].lower().split())
    rows = []
    for pid in cand_ids:
        p = prod_by_id[pid]
        cat_tokens = set(p["category"].replace("_", " ").lower().split())
        p_tokens = set(p["title"].lower().split()) | cat_tokens
        rows.append([
            bm25_scores.get(pid, 0.0),
            dense_scores.get(pid, 0.0),
            rrf_scores.get(pid, 0.0),
            p["popularity"],
            (p["price"] - price_mean) / (price_sd or 1.0),
            p["rating"],
            len(q_tokens & cat_tokens) / max(len(cat_tokens), 1),
            1.0 if p["brand"].lower() in q_tokens else 0.0,
            len(q_tokens & p_tokens) / max(len(q_tokens), 1),
            1.0 if p["noun"].split()[-1].lower() in q_tokens else 0.0,
            float(len(q_tokens)),
        ])
    return np.asarray(rows, dtype=np.float32)
