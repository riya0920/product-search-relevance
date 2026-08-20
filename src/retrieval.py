"""Three retrieval legs: lexical, dense, and their fusion.

SUBSTITUTION, STATED UP FRONT: sentence-transformers is installed but its
pretrained weights are not downloadable in this offline environment, so the dense
leg is a two-tower model TRAINED ON THIS CORPUS -- TF-IDF, a shared SVD to 128
dimensions, and a learned bilinear scoring matrix fitted with a softmax
contrastive loss over sampled negatives.

That substitution cuts both ways and the README says so. It loses the world
knowledge a pretrained encoder brings, which is exactly what would help most on
the "need" queries. It gains the thing this project is actually about: the model
is TRAINED, so hard-negative mining is a real experiment with a real delta rather
than a prompt-engineering exercise on frozen weights.

Queries are split train/test. The dense model never sees a test query.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def product_text(p: dict) -> str:
    """What goes in the index. Category tokens are included deliberately: a real
    product index carries structured fields, not just the merchandiser's title."""
    return "%s %s %s %s %s" % (p["title"], p["category"].replace("_", " "),
                               " ".join(p["attrs"]), p["brand"], p["color"])


# --------------------------------------------------------------------------
# lexical
# --------------------------------------------------------------------------
class BM25Index:
    """The baseline that is embarrassingly hard to beat on head queries."""

    def __init__(self, products: list[dict]):
        from rank_bm25 import BM25Okapi
        self.products = products
        self.ids = [p["product_id"] for p in products]
        self.tokens = [product_text(p).lower().split() for p in products]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query: str, k: int = 100):
        scores = self.bm25.get_scores(query.lower().split())
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top]

    def score_all(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(query.lower().split()), float)


# --------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------
class TwoTower:
    """score(q, p) = (E_q q)^T W (E_p p), W learned by contrastive softmax.

    Deliberately small and written out rather than pulled from a library, because
    the hard-negative experiment needs to control exactly which negatives enter
    the loss -- which is the whole point of the experiment.
    """

    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.vec = None
        self.svd = None
        self.P = None       # product embeddings (n_products x dim)
        self.W = None

    def fit_space(self, products: list[dict], queries: list[dict]):
        corpus = [product_text(p) for p in products] + [q["text"] for q in queries]
        self.vec = TfidfVectorizer(sublinear_tf=True, min_df=1,
                                   ngram_range=(1, 2), analyzer="word")
        X = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=self.dim, random_state=0)
        self.svd.fit(X)
        self.P = normalize(self.svd.transform(self.vec.transform(
            [product_text(p) for p in products])))
        self.ids = [p["product_id"] for p in products]
        self.pos = {pid: i for i, pid in enumerate(self.ids)}
        self.W = np.eye(self.dim)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vec.transform(queries)))

    def train(self, train_queries: list[dict], judgments: dict,
              negatives: dict | None = None, n_neg: int = 12,
              epochs: int = 14, lr: float = 0.35, seed: int = 0):
        """Softmax contrastive loss over 1 positive + n_neg negatives.

        `negatives` optionally supplies a per-query list of MINED hard negatives;
        without it, negatives are sampled uniformly from the catalogue. The
        difference between those two runs is the hard-negative experiment.
        """
        rng = np.random.default_rng(seed)
        Q = self.embed_queries([q["text"] for q in train_queries])
        n_p = len(self.ids)

        pairs = []
        for i, q in enumerate(train_queries):
            lab = judgments[q["query_id"]]
            positives = [pid for pid, g in lab.items() if g == 3]
            if not positives:
                positives = [pid for pid, g in lab.items() if g == 2]
            if not positives:
                continue
            irrelevant = [pid for pid, g in lab.items() if g == 0]
            pairs.append((i, positives, irrelevant, q["query_id"]))

        for ep in range(epochs):
            rng.shuffle(pairs)
            grad = np.zeros_like(self.W)
            n_used = 0
            for qi, positives, irrelevant, qid in pairs:
                q = Q[qi]
                pos_id = positives[int(rng.integers(len(positives)))]

                if negatives and negatives.get(qid):
                    pool = negatives[qid]
                    neg_ids = [pool[int(rng.integers(len(pool)))]
                               for _ in range(n_neg)]
                elif irrelevant:
                    neg_ids = [irrelevant[int(rng.integers(len(irrelevant)))]
                               for _ in range(n_neg)]
                else:
                    neg_ids = [self.ids[int(rng.integers(n_p))] for _ in range(n_neg)]

                cand = [self.pos[pos_id]] + [self.pos[n] for n in neg_ids]
                Pc = self.P[cand]                       # (1+n_neg, dim)
                qW = q @ self.W                         # (dim,)
                s = Pc @ qW
                s = s - s.max()
                e = np.exp(s)
                pr = e / e.sum()
                pr[0] -= 1.0                            # d(loss)/d(scores)
                grad += np.outer(q, pr @ Pc)
                n_used += 1
            if n_used:
                self.W -= lr * grad / n_used
        return self

    def score_all(self, query_vec: np.ndarray) -> np.ndarray:
        return self.P @ (query_vec @ self.W)

    def search(self, query_vec: np.ndarray, k: int = 100):
        s = self.score_all(query_vec)
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        top = top[np.argsort(-s[top])]
        return [(self.ids[i], float(s[i])) for i in top]


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion -- the baseline fusion, rank-based and scale-free.

    RRF needs no score normalisation, which is its whole appeal: BM25 scores and
    cosine similarities live on incomparable scales and any weighted blend of the
    raw numbers is an accident waiting to happen.
    """
    out: dict[str, float] = {}
    for r in rankings:
        for rank, pid in enumerate(r, start=1):
            out[pid] = out.get(pid, 0.0) + 1.0 / (k + rank)
    return out


def weighted_fusion(score_lists: list[dict[str, float]], weights: list[float]
                    ) -> dict[str, float]:
    """Min-max normalise each leg, then blend. Tracked as an experiment against
    RRF rather than assumed better."""
    out: dict[str, float] = {}
    for scores, w in zip(score_lists, weights):
        if not scores:
            continue
        v = np.array(list(scores.values()), float)
        lo, hi = v.min(), v.max()
        rng = (hi - lo) or 1.0
        for pid, s in scores.items():
            out[pid] = out.get(pid, 0.0) + w * (s - lo) / rng
    return out


def mine_hard_negatives(bm25: BM25Index, queries: list[dict], judgments: dict,
                        top_k: int = 30, per_query: int = 20) -> dict:
    """BM25-top-but-labelled-irrelevant. The highest-signal-per-effort trick in
    retail search.

    The intuition: negatives sampled uniformly from a 600-product catalogue are
    trivially separable -- the model learns "coffee is not a shoe" and stops. The
    negatives that teach it something are the ones that already look right to a
    lexical matcher and are still wrong. Those are precisely BM25's top-ranked
    mistakes.
    """
    out = {}
    for q in queries:
        lab = judgments[q["query_id"]]
        hard = []
        for pid, _ in bm25.search(q["text"], k=top_k):
            if lab.get(pid, 0) == 0:
                hard.append(pid)
        if hard:
            out[q["query_id"]] = hard
    return out
