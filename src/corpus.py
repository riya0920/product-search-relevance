"""An ESCI-shaped product corpus with GRADED relevance judgments.

Amazon's ESCI dataset is not downloadable in this offline environment. What makes
ESCI worth using is not its size -- it is that every (query, product) pair carries
a graded label:

    E  Exact       this product is what was asked for
    S  Substitute  serves the same need, is not the same thing
    C  Complement  bought alongside, does not satisfy the query
    I  Irrelevant

This module generates a corpus with that structure and, because it also writes
the generative process down, the labels are GROUND TRUTH rather than a
crowdsourced approximation. That is a real advantage for measuring a ranker and a
real limitation for claiming anything about human relevance -- both are stated in
the README.

The query mix is deliberately Zipf-distributed, because head/torso/tail is the
segmentation that decides whether semantic retrieval is worth deploying, and a
uniform query sample would hide it entirely.
"""
from __future__ import annotations

import json
import os

import numpy as np

RNG = np.random.default_rng(31337)

# category -> (product nouns, defining attributes, complementary categories)
TAXONOMY = {
    "running_shoes": (["running shoe", "trainer", "runner"],
                      ["cushioned", "lightweight", "stability", "trail", "road"],
                      ["athletic_socks", "insoles"]),
    "work_boots": (["work boot", "safety boot", "steel toe boot"],
                   ["waterproof", "steel toe", "insulated", "slip resistant"],
                   ["insoles", "athletic_socks"]),
    "athletic_socks": (["athletic sock", "crew sock", "running sock"],
                       ["merino", "cushioned", "no show", "compression"],
                       ["running_shoes", "work_boots"]),
    "insoles": (["insole", "shoe insert", "orthotic"],
                ["arch support", "gel", "memory foam", "heat moldable"],
                ["running_shoes", "work_boots"]),
    "coffee_maker": (["coffee maker", "drip brewer", "coffee machine"],
                     ["programmable", "thermal carafe", "single serve", "burr"],
                     ["coffee_beans", "coffee_filters"]),
    "coffee_beans": (["coffee beans", "ground coffee", "whole bean coffee"],
                     ["dark roast", "medium roast", "single origin", "decaf"],
                     ["coffee_maker", "coffee_filters"]),
    "coffee_filters": (["coffee filter", "paper filter", "reusable filter"],
                       ["unbleached", "cone", "basket", "permanent"],
                       ["coffee_maker", "coffee_beans"]),
    "office_chair": (["office chair", "desk chair", "task chair"],
                     ["ergonomic", "mesh back", "lumbar support", "standing"],
                     ["desk_mat", "monitor_stand"]),
    "desk_mat": (["desk mat", "desk pad", "mouse pad"],
                 ["leather", "felt", "extended", "waterproof"],
                 ["office_chair", "monitor_stand"]),
    "monitor_stand": (["monitor stand", "monitor riser", "laptop stand"],
                      ["adjustable", "bamboo", "dual monitor", "with drawer"],
                      ["office_chair", "desk_mat"]),
}

BRANDS = ["Aeris", "Northvale", "Kestrel", "Lumen", "Corvid", "Bastion",
          "Halcyon", "Pindrop", "Vantage", "Orrery"]
COLORS = ["black", "navy", "grey", "white", "olive", "burgundy"]

# The "need" queries -- exploratory, no exact product name. These are where a
# lexical index has nothing to match on and semantics should earn its keep.
NEED_QUERIES = {
    "shoes for standing all day": ["work_boots", "insoles", "running_shoes"],
    "something for sore feet at work": ["insoles", "work_boots"],
    "gift for a coffee lover": ["coffee_maker", "coffee_beans"],
    "make my home office comfortable": ["office_chair", "desk_mat", "monitor_stand"],
    "help with lower back pain sitting": ["office_chair"],
    "keep feet dry on site": ["work_boots", "athletic_socks"],
    "fresh coffee every morning": ["coffee_maker", "coffee_beans"],
    "reduce wrist strain typing": ["desk_mat", "monitor_stand"],
    "warm feet in winter": ["athletic_socks", "work_boots"],
    "run a marathon comfortably": ["running_shoes", "athletic_socks"],
}


def build_products(n_per_cat: int = 60) -> list[dict]:
    prods = []
    pid = 0
    for cat, (nouns, attrs, _) in TAXONOMY.items():
        for _ in range(n_per_cat):
            noun = str(RNG.choice(nouns))
            brand = str(RNG.choice(BRANDS))
            color = str(RNG.choice(COLORS))
            k = int(RNG.integers(1, 3))
            a = list(RNG.choice(attrs, size=k, replace=False))
            title = "%s %s %s %s" % (brand, " ".join(a), noun, color)
            prods.append(dict(
                product_id="P%05d" % pid, category=cat, brand=brand,
                noun=noun, attrs=a, color=color, title=title,
                price=round(float(RNG.uniform(8, 220)), 2),
                # popularity is Zipf-ish: a few products carry most of the demand
                popularity=float(RNG.zipf(1.6)) if RNG.random() < 0.9 else 1.0,
                rating=round(float(np.clip(RNG.normal(4.1, 0.5), 1, 5)), 1)))
            pid += 1
    # normalise popularity into [0,1]
    pop = np.array([p["popularity"] for p in prods], float)
    pop = np.log1p(pop)
    pop = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
    for p, v in zip(prods, pop):
        p["popularity"] = float(v)
    return prods


def build_queries(products: list[dict], n_queries: int = 600) -> list[dict]:
    """Three query archetypes, then a Zipf frequency assignment over all of them.

    - navigational : brand + product noun, sometimes a near-exact title
    - attribute    : attribute + noun ("waterproof work boot")
    - need         : natural-language intent with no product noun at all
    """
    queries = []
    qid = 0
    cats = list(TAXONOMY)

    for _ in range(int(n_queries * 0.45)):
        p = products[int(RNG.integers(len(products)))]
        text = ("%s %s" % (p["brand"], p["noun"]) if RNG.random() < 0.6
                else "%s %s %s" % (p["brand"], p["attrs"][0], p["noun"]))
        queries.append(dict(query_id="Q%04d" % qid, text=text, kind="navigational",
                            target_category=p["category"], target_brand=p["brand"],
                            target_attrs=[p["attrs"][0]]))
        qid += 1

    for _ in range(int(n_queries * 0.35)):
        cat = str(RNG.choice(cats))
        nouns, attrs, _ = TAXONOMY[cat]
        a = str(RNG.choice(attrs))
        text = "%s %s" % (a, str(RNG.choice(nouns)))
        queries.append(dict(query_id="Q%04d" % qid, text=text, kind="attribute",
                            target_category=cat, target_brand=None,
                            target_attrs=[a]))
        qid += 1

    need_texts = list(NEED_QUERIES)
    while len(queries) < n_queries:
        t = str(RNG.choice(need_texts))
        queries.append(dict(query_id="Q%04d" % qid, text=t, kind="need",
                            target_category=NEED_QUERIES[t][0], target_brand=None,
                            target_attrs=[]))
        qid += 1

    # Zipf frequencies -> head / torso / tail buckets. Real query logs are
    # brutally head-heavy and evaluating on a uniform sample hides that.
    ranks = np.arange(1, len(queries) + 1)
    RNG.shuffle(ranks)
    freqs = 1.0 / ranks ** 0.9
    freqs = freqs / freqs.sum()

    # Buckets are defined by frequency RANK, not by cumulative volume. Cutting at
    # 30% of volume is the textbook definition and it put 7 queries in the head
    # bucket -- statistically useless for an NDCG comparison, which is the whole
    # reason the segmentation exists. Rank cuts keep each bucket measurable; the
    # volume share each one carries is reported alongside so the head's real
    # weight is not hidden by the change.
    order = np.argsort(-freqs)
    n = len(queries)
    seg = np.empty(n, dtype=object)
    seg[order[:int(0.10 * n)]] = "head"
    seg[order[int(0.10 * n):int(0.40 * n)]] = "torso"
    seg[order[int(0.40 * n):]] = "tail"
    for q, f, s in zip(queries, freqs, seg):
        q["frequency"] = float(f)
        q["segment"] = str(s)
    return queries


def judge(query: dict, product: dict) -> int:
    """Graded label as an integer gain: E=3, S=2, C=1, I=0.

    The rules ARE the ground truth. They encode the retail distinction the whole
    project turns on: a Substitute serves the need, a Complement does not --
    ranking a complement above a substitute for a purchase-intent query is a
    ranking bug even though both are 'related'.
    """
    cat = product["category"]
    if query["kind"] == "need":
        wanted = NEED_QUERIES[query["text"]]
        if cat == wanted[0]:
            return 3
        if cat in wanted[1:]:
            return 2
        _, _, comps = TAXONOMY[wanted[0]]
        return 1 if cat in comps else 0

    if cat != query["target_category"]:
        _, _, comps = TAXONOMY[query["target_category"]]
        return 1 if cat in comps else 0

    # same category: brand and attribute decide exact vs substitute
    brand_ok = (query["target_brand"] is None
                or product["brand"] == query["target_brand"])
    attr_ok = (not query["target_attrs"]
               or any(a in product["attrs"] for a in query["target_attrs"]))
    if brand_ok and attr_ok:
        return 3
    if brand_ok or attr_ok:
        return 2
    return 2   # same category, nothing else matches: still a substitute


def build(out_dir: str, n_per_cat: int = 60, n_queries: int = 600) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    products = build_products(n_per_cat)
    queries = build_queries(products, n_queries)

    judgments = {}
    for q in queries:
        labels = {p["product_id"]: judge(q, p) for p in products}
        # keep the graded ones plus a sample of the irrelevant mass, which is how
        # a real judgment set is built (you cannot label a whole catalogue)
        keep = {pid: g for pid, g in labels.items() if g > 0}
        zeros = [pid for pid, g in labels.items() if g == 0]
        for pid in RNG.choice(zeros, size=min(60, len(zeros)), replace=False):
            keep[pid] = 0
        judgments[q["query_id"]] = keep

    with open(os.path.join(out_dir, "products.json"), "w") as f:
        json.dump(products, f)
    with open(os.path.join(out_dir, "queries.json"), "w") as f:
        json.dump(queries, f)
    with open(os.path.join(out_dir, "judgments.json"), "w") as f:
        json.dump(judgments, f)

    grades = [g for m in judgments.values() for g in m.values()]
    stats = dict(
        n_products=len(products), n_queries=len(queries),
        n_judgments=len(grades),
        grade_counts={str(k): int(np.sum(np.array(grades) == k)) for k in (0, 1, 2, 3)},
        segments={s: int(sum(1 for q in queries if q["segment"] == s))
                  for s in ("head", "torso", "tail")},
        segment_volume_share={
            s: round(float(sum(q["frequency"] for q in queries
                               if q["segment"] == s)), 4)
            for s in ("head", "torso", "tail")},
        kinds={k: int(sum(1 for q in queries if q["kind"] == k))
               for k in ("navigational", "attribute", "need")})
    with open(os.path.join(out_dir, "corpus_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(json.dumps(build(os.path.join(here, "data")), indent=2))
