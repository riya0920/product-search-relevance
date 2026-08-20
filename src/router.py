"""Query-type routing, merchandiser boosting, spelling, and interleaving.

Four things the first pass named and did not build. Three of them are the
answers to the spec's grill questions, which is why they were worth building
rather than describing.

THE ROUTER exists because of a measured result, not a hunch: BM25 beat the dense
leg outright on navigational queries (0.894 vs 0.858) and lost catastrophically
on need queries (0.280 vs 0.372). A single fusion weight tuned on aggregate NDCG
gets tuned by the tail -- most of the QUERIES and little of the TRAFFIC -- so it
is wrong for the head, which is where the volume is.

The router classifies the query and picks the weight. It is deliberately a
CHEAP, INTERPRETABLE classifier rather than a model: it runs on every query
inside the latency budget, a search engineer has to be able to explain why a
query routed the way it did, and the features (does the query contain a brand?
a product noun? how long is it?) are exactly the ones a human would use.
"""
from __future__ import annotations

import numpy as np

from .retrieval import product_text, rrf, weighted_fusion


# --------------------------------------------------------------------------
# query classification
# --------------------------------------------------------------------------
class QueryRouter:
    """Classify a query, then choose the lexical/semantic blend for it."""

    # Fusion weight on the LEXICAL leg, per predicted query type. Fitted on the
    # training half in run_search.py rather than asserted here.
    DEFAULT_WEIGHTS = {"navigational": 0.85, "attribute": 0.70, "need": 0.20}

    def __init__(self, products: list[dict], weights: dict | None = None):
        self.brands = {p["brand"].lower() for p in products}
        self.nouns = set()
        for p in products:
            for tok in p["noun"].lower().split():
                self.nouns.add(tok)
        self.attrs = set()
        for p in products:
            for a in p["attrs"]:
                for tok in a.lower().split():
                    self.attrs.add(tok)
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

    def features(self, query: str) -> dict:
        toks = query.lower().split()
        tokset = set(toks)
        return dict(
            has_brand=bool(tokset & self.brands),
            has_noun=bool(tokset & self.nouns),
            has_attr=bool(tokset & self.attrs),
            n_tokens=len(toks),
            has_stopwordy=bool(tokset & {"for", "with", "my", "me", "a", "the",
                                         "all", "every", "help", "gift"}))

    def classify(self, query: str) -> str:
        """Rules, in priority order, each of which a search engineer can defend.

        A brand token is the strongest signal there is: nobody types a brand name
        unless they have one in mind. Absence of any product noun with a
        function word present is the 'need' signature -- the user described a
        problem rather than a product, which is exactly the case a lexical index
        has nothing to match on.
        """
        f = self.features(query)
        if f["has_brand"]:
            return "navigational"
        # A function word plus length is the 'need' signature even when a product
        # noun IS present: "gift for a coffee lover" contains 'coffee' but is not
        # a query for coffee. Requiring the absence of a noun misrouted exactly
        # that shape, which the confusion matrix showed as 21 need queries landing
        # in 'attribute'.
        if f["has_stopwordy"] and f["n_tokens"] >= 4:
            return "need"
        if not f["has_noun"] and (f["has_stopwordy"] or f["n_tokens"] >= 4):
            return "need"
        if f["has_attr"] and f["has_noun"]:
            return "attribute"
        return "attribute" if f["has_noun"] else "need"

    def weight_for(self, query: str) -> float:
        return self.weights[self.classify(query)]

    def route(self, query: str, bm25_scores: dict, dense_scores: dict,
              k: int = 100) -> list[str]:
        w = self.weight_for(query)
        fused = weighted_fusion([bm25_scores, dense_scores], [w, 1.0 - w])
        return [pid for pid, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:k]


def fit_weights(router: QueryRouter, train_queries, judgments, score_fn,
                ndcg_fn, grid=(0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0), k=10):
    """Choose the lexical weight PER QUERY TYPE on the training half.

    Fitting per type rather than globally is the entire point: the global optimum
    is a compromise nobody's query wants.
    """
    by_type: dict[str, list] = {}
    for q in train_queries:
        by_type.setdefault(router.classify(q["text"]), []).append(q)

    out = {}
    for qtype, qs in by_type.items():
        best_w, best_score = 0.5, -1.0
        for w in grid:
            tot = 0.0
            for q in qs:
                bm, ds = score_fn(q)
                fused = weighted_fusion([bm, ds], [w, 1.0 - w])
                ranked = [pid for pid, _ in sorted(fused.items(),
                                                   key=lambda kv: -kv[1])][:k]
                tot += ndcg_fn(ranked, judgments[q["query_id"]], k)
            avg = tot / max(len(qs), 1)
            if avg > best_score:
                best_w, best_score = w, avg
        out[qtype] = dict(weight=best_w, train_ndcg=best_score, n=len(qs))
    return out


# --------------------------------------------------------------------------
# merchandiser boosting -- the governed business rule
# --------------------------------------------------------------------------
class BoostRule:
    """A merchandiser's demand, implemented as a GOVERNED, MEASURED rule.

    The spec's question is what you do when a merchandiser demands their brand
    ranks top-3 for a generic query. Refusing outright and obeying blindly both
    fail. The systems answer is: implement it, scope it, log it, and quantify
    what it costs in relevance -- so the conversation becomes "this boost costs
    0.03 NDCG, is it worth it" instead of "search is broken".
    """

    def __init__(self, rule_id: str, brand: str, query_pattern: str,
                 target_positions: int = 3, owner: str = "merchandising"):
        self.rule_id = rule_id
        self.brand = brand
        self.query_pattern = query_pattern.lower()
        self.target_positions = target_positions
        self.owner = owner
        self.fired = 0

    def applies(self, query: str) -> bool:
        return self.query_pattern in query.lower()

    def apply(self, query: str, ranked: list[str], prod_by_id: dict) -> tuple[list[str], dict]:
        """Pull up to `target_positions` of the brand's products into the top.

        Returns the new ranking AND an audit record. The audit record is the
        difference between a governed rule and a hack: it names the rule, the
        owner, and exactly which documents moved.
        """
        if not self.applies(query):
            return ranked, {}
        brand_hits = [pid for pid in ranked
                      if prod_by_id[pid]["brand"].lower() == self.brand.lower()]
        if not brand_hits:
            return ranked, dict(rule_id=self.rule_id, fired=False,
                                reason="no_brand_products_in_candidate_set")
        promote = brand_hits[:self.target_positions]
        moved = [(pid, ranked.index(pid)) for pid in promote]
        rest = [pid for pid in ranked if pid not in set(promote)]
        self.fired += 1
        return promote + rest, dict(
            rule_id=self.rule_id, fired=True, owner=self.owner,
            brand=self.brand, promoted=[p for p, _ in moved],
            from_positions=[i for _, i in moved])


# --------------------------------------------------------------------------
# spelling correction
# --------------------------------------------------------------------------
class SpellCorrector:
    """Edit-distance-1 correction against the corpus vocabulary.

    Deliberately conservative: a token is only corrected if it is NOT in the
    vocabulary and exactly one in-vocabulary candidate is within edit distance
    one. Aggressive correction is worse than none -- "corrected" queries that
    change the user's intent produce confidently wrong results, and the user
    cannot tell what happened.
    """

    def __init__(self, products: list[dict], queries: list[dict] | None = None):
        vocab: dict[str, int] = {}
        for p in products:
            for tok in product_text(p).lower().split():
                vocab[tok] = vocab.get(tok, 0) + 1
        for q in (queries or []):
            for tok in q["text"].lower().split():
                vocab[tok] = vocab.get(tok, 0) + 1
        self.vocab = vocab
        self.by_len: dict[int, list[str]] = {}
        for w in vocab:
            self.by_len.setdefault(len(w), []).append(w)

    @staticmethod
    def _within_one(a: str, b: str) -> bool:
        if abs(len(a) - len(b)) > 1:
            return False
        if a == b:
            return True
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) == 1
        short, long_ = (a, b) if len(a) < len(b) else (b, a)
        i = j = diff = 0
        while i < len(short) and j < len(long_):
            if short[i] != long_[j]:
                diff += 1
                if diff > 1:
                    return False
                j += 1
            else:
                i += 1
                j += 1
        return True

    def correct(self, query: str) -> tuple[str, list[tuple[str, str]]]:
        out, changes = [], []
        for tok in query.lower().split():
            if tok in self.vocab:
                out.append(tok)
                continue
            cands = []
            for L in (len(tok) - 1, len(tok), len(tok) + 1):
                for w in self.by_len.get(L, ()):
                    if self._within_one(tok, w):
                        cands.append(w)
            # only correct when the answer is UNAMBIGUOUS
            if len(set(cands)) == 1:
                out.append(cands[0])
                changes.append((tok, cands[0]))
            else:
                out.append(tok)
        return " ".join(out), changes


# --------------------------------------------------------------------------
# interleaving
# --------------------------------------------------------------------------
def team_draft_interleave(ranking_a: list[str], ranking_b: list[str],
                          k: int = 10, rng=None):
    """Team-draft interleaving: the online comparison offline NDCG cannot give you.

    Two rankers take turns drafting their next unpicked document into a single
    result list, coin-flipping for who picks first at each round. The user sees
    ONE list and cannot tell which system produced any given result, so their
    clicks are an unbiased preference signal between the two.

    Why this matters and why the spec asks about it: offline NDCG says which
    ranker better matches the JUDGMENTS. Interleaving says which ranker users
    actually prefer -- and those disagree whenever the judgments are stale, the
    labellers were not the users, or the metric is insensitive to what changed.
    Interleaving is also far more sensitive than an A/B split, because every
    session compares both systems instead of being assigned to one.
    """
    rng = rng or np.random.default_rng(0)
    out, source = [], []
    ia = ib = 0
    while len(out) < k and (ia < len(ranking_a) or ib < len(ranking_b)):
        a_first = rng.random() < 0.5
        for team in ((0, 1) if a_first else (1, 0)):
            if len(out) >= k:
                break
            if team == 0:
                while ia < len(ranking_a) and ranking_a[ia] in out:
                    ia += 1
                if ia < len(ranking_a):
                    out.append(ranking_a[ia])
                    source.append("A")
                    ia += 1
            else:
                while ib < len(ranking_b) and ranking_b[ib] in out:
                    ib += 1
                if ib < len(ranking_b):
                    out.append(ranking_b[ib])
                    source.append("B")
                    ib += 1
    return out, source


def simulate_clicks(interleaved: list[str], labels: dict, rng,
                    position_bias: float = 0.75, noise: float = 0.10):
    """Simulate a user clicking, with POSITION BIAS -- the thing that makes naive
    click counting useless and interleaving necessary.

    P(click) falls geometrically with rank regardless of relevance, so a system
    whose results merely appear higher gets more clicks. Interleaving cancels
    this because both systems' documents are distributed across the same
    positions by the draft.
    """
    clicks = []
    for rank, pid in enumerate(interleaved):
        rel = labels.get(pid, 0) / 3.0
        p = (position_bias ** rank) * (noise + (1 - noise) * rel)
        if rng.random() < p:
            clicks.append(rank)
    return clicks


def interleaving_verdict(per_query_wins: list[int]) -> dict:
    """Sign test over per-query preferences.

    A per-query WIN is what interleaving produces; aggregating with a sign test
    rather than a t-test is right because the unit is a query and the outcome is
    ordinal, not a mean of anything.
    """
    a = sum(1 for w in per_query_wins if w > 0)
    b = sum(1 for w in per_query_wins if w < 0)
    ties = sum(1 for w in per_query_wins if w == 0)
    n = a + b
    if n == 0:
        return dict(a_wins=0, b_wins=0, ties=ties, preference=0.5, p_value=1.0)
    from scipy import stats
    p = float(stats.binomtest(a, n, 0.5).pvalue)
    return dict(a_wins=a, b_wins=b, ties=ties, preference=a / n, p_value=p)
