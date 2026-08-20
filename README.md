# ML-2 — Product Search & Relevance Engine

**Roughly 50% of the spec.** A labelled eval with segmented NDCG against a BM25
baseline, three retrieval legs and a re-ranker - plus the five things the first
pass named as missing and has now built: a query-type router, governed
merchandiser boosting, spelling correction, interleaving, and the query-debug
view. Still no storefront and no HTTP API; what remains is named at the bottom.

```bash
python src/corpus.py        # ~2s    build the ESCI-shaped corpus
python run_search.py        # ~60s   the evaluation
python -m pytest tests -q   # 39 tests
```

## The corpus

Amazon ESCI is not downloadable offline, so `src/corpus.py` generates a corpus
with ESCI's *structure* — every (query, product) pair carries a graded label:

| | | |
|---|---|---|
| **E** Exact = 3 | **S** Substitute = 2 | **C** Complement = 1, **I** Irrelevant = 0 |

600 products across 10 categories, 600 queries, 144,960 judgments. Query mix is
Zipf-distributed across three archetypes: **navigational** (brand + product
noun), **attribute** ("waterproof work boot"), and **need** ("shoes for standing
all day" — no product noun at all).

Buckets are cut by frequency **rank**, not cumulative volume. The textbook 30%-of-
volume cut put 7 queries in the head bucket — useless for an NDCG comparison,
which is the whole reason the segmentation exists. Rank cuts keep each bucket
measurable and the volume share is reported alongside:

| segment | queries | share of volume |
|---|---|---|
| head | 60 | 59.2% |
| torso | 180 | 23.4% |
| tail | 360 | 17.4% |

**Split is by query TEXT, not by query instance.** The first version split by
instance, which put identical "need" phrasings in both halves and reported
NDCG@10 = **1.0000** on that segment. That is what memorisation looks like when
you don't check. A test asserts the grouped split shares no strings.

## Where BM25 still wins

NDCG@10 on held-out queries:

| | BM25 | dense | hybrid RRF | hybrid weighted |
|---|---|---|---|---|
| head | 0.787 | 0.781 | 0.807 | **0.818** |
| torso | 0.772 | 0.776 | 0.787 | **0.806** |
| tail | 0.773 | 0.782 | 0.800 | **0.805** |
| **navigational** | **0.894** | 0.858 | 0.880 | 0.891 |
| **attribute** | **0.978** | 0.977 | 0.978 | 0.977 |
| **need** | 0.280 | 0.372 | 0.399 | **0.423** |
| ALL | 0.774 | 0.780 | 0.797 | **0.807** |

BM25 **wins outright** on navigational and attribute queries. It should: the user
typed a brand and a product noun, the tokens are literally in the title, and
there is nothing for semantics to add. Need queries are the mirror image — BM25
scores 0.280 because no product noun appears at all, and the dense leg carries
the segment (+0.142 for the weighted hybrid).

**The deployment consequence:** head queries are mostly navigational *and* carry
59% of volume. A hybrid tuned on aggregate NDCG gets tuned by the tail — most of
the *queries*, little of the *traffic*. The mature answer is fusion weights that
vary by query type — which the second pass builds, below.

## The hard-negative experiment came out negative

The spec calls hard-negative mining the highest-signal-per-effort trick in retail
search. Measured here, it made the model **worse**:

| | random negs | hard negs | delta |
|---|---|---|---|
| ALL | 0.780 | 0.774 | **−0.0064** |
| need | 0.372 | 0.347 | −0.0249 |

Reporting that is the point of running the experiment rather than citing it. Why
it fails *here* — a property of the corpus, not of the technique:

1. **Too scarce.** Mining found hard negatives for only 60 of 271 training
   queries. In this taxonomy every complement category grades as 1, so BM25's
   top-30 contains very few genuine grade-0 items.
2. **With few negatives, hard ones overfit.** Replacing a broad sample of easy
   negatives with a narrow sample of hard ones shrinks the loss's effective
   diversity — worst on need queries, which most need a broad sense of "far away".
3. **600 products is not where this lives.** The technique pays when uniform
   negatives are trivially separable, which is a property of million-item
   catalogues.

The honest conclusion isn't "hard negatives don't work" — it's that this corpus
can't test the claim, and the production fix (**mixing** hard and random rather
than substituting) is the experiment I'd run next and didn't build.

## The sharpest finding: NDCG and substitute-ordering disagree

For a purchase-intent query a **Substitute must outrank a Complement** — someone
searching for running shoes wants a different running shoe, not socks. Rank the
systems both ways and you get different orders:

| system | NDCG@10 | sub > comp |
|---|---|---|
| reranked | **0.811** | 0.632 |
| hybrid weighted | 0.807 | 0.581 |
| hybrid RRF | 0.797 | 0.469 |
| dense (random negs) | 0.780 | 0.471 |
| **bm25** | 0.774 | **0.677** |
| dense (hard negs) | 0.774 | 0.388 |

BM25 — the baseline everything beats on NDCG — is **best** at substitute
ordering. The mechanism is in the discount: NDCG@10 with 2^g−1 gains is dominated
by whether Exact matches (gain 7) reach the first slots. Whether a Substitute
(gain 3) outranks a Complement (gain 1) five slots down barely moves the number.
So a ranker optimises NDCG happily while filling its lower slots with
accessories.

This is the concrete form of *"aggregates hide everything in search"*. The fix is
a listwise objective that sees grade ordering, or an explicit intent-aware
ranking constraint. Neither is built — what *is* built is the measurement saying
you need one.

## Re-ranker, latency, zero results

The re-ranker adds **+0.0147 NDCG@10** overall (+0.035 on navigational) for
**16.0 ms** at p95 → **+0.0050 NDCG per millisecond**. Search is a
latency-budgeted business, so the reranker's value is a ratio, not a delta —
those milliseconds compete with a bigger candidate set or personalisation.

Permutation importance ranks `dense` > `bm25` > `brand_in_query` >
`title_token_overlap`. **Popularity is near the bottom**, which is worth stating
because rich-get-richer is the failure mode screeners probe for — it would still
bite in production, where popularity is measured from logs the previous ranker
generated and is therefore endogenous.

Zero-result rescue: 150 queries with an unmatchable token appended go from
**100% → 0%** zero-result rate under relax-to-OR then semantic fallback, and the
rescued results score **0.819** NDCG@10. Showing *something* is not automatically
better than showing nothing, so the quality of the rescue is reported rather than
assumed.

## Substitutions, named

| spec asks for | used instead | what it costs |
|---|---|---|
| Amazon ESCI | generated ESCI-shaped corpus | labels are ground truth (better for measuring a ranker) but not human judgments (worse for claiming anything about real relevance) |
| sentence-transformers bi-encoder | two-tower TF-IDF → SVD(128) → learned bilinear W, contrastive loss | no pretrained world knowledge — exactly what would help most on need queries. Gains a genuinely *trained* model, so the hard-negative experiment is real |
| LightGBM LambdaMART | `HistGradientBoostingRegressor` on the graded label | pointwise objective standing in for listwise. Pointwise fits the label, listwise fits the *order*, and NDCG only cares about order — so the reranker gain is a floor |
| Elasticsearch/OpenSearch | `rank_bm25`, linear scan | no real index; latency numbers are relative costs between legs, not service latencies |

## Second pass: five gaps the first pass named

### The query-type router

Section 1 ended by saying a single fusion weight gets tuned by the tail - most of
the *queries*, little of the *traffic* - and that the mature answer is weights
that vary by query type. Built now, fitted per type on the training half:

| query type | fitted lexical weight | train NDCG | n |
|---|---|---|---|
| navigational | **1.00** | 0.8837 | 128 |
| attribute | **1.00** | 0.9469 | 129 |
| need | **0.00** | 1.0000 | 14 |

**Two things to be suspicious of before reading any gain.** The weights are
*extreme* - 1.00 and 0.00, not a blend - so the best "hybrid" here is not a
hybrid at all, it is a **router between two pure legs**. That is a property of
this corpus, where the legs are good at disjoint things; on a real catalogue they
overlap far more and interior weights usually win. And the `need` weight is
fitted on **14 queries with a train NDCG of 1.0000**, which is overfitting, not
learning - the test-set gain is the only evidence it generalises.

Router accuracy vs the generator's true query kind: **93.6%** on 329 test queries.

| segment | fixed 0.6/0.4 | routed | delta |
|---|---|---|---|
| navigational | 0.8905 | 0.8937 | +0.0032 |
| attribute | 0.9769 | 0.9775 | +0.0006 |
| **need** | 0.4227 | **0.4424** | **+0.0197** |
| ALL | 0.8065 | 0.8128 | +0.0063 |

Read the segment rows, not the aggregate - the router exists to help the segments
a compromise weight hurts, and the aggregate is exactly the number that hides
whether it did.

The router is **rules, not a model**, on purpose: it runs on every query inside
the latency budget, an engineer has to be able to explain why a query routed the
way it did, and its features are the ones a human would use.

*A misroute the confusion matrix caught:* "gift for a coffee lover" contains the
product noun `coffee`, so the original rule sent it to `attribute`. 21 need
queries landed there. The rule now treats *function word + length* as the need
signature even when a noun is present.

### Merchandiser boosting - governed, logged, priced

The spec's question: a merchandiser demands their brand ranks top-3 for a generic
query. Refusing and obeying both fail.

| ranking | NDCG@10 | brand in top 3 |
|---|---|---|
| organic | **0.4031** | 17% |
| with boost | 0.3662 | **100%** |

**The boost costs 0.0369 NDCG@10** on the 29 queries it touches. That number is
what turns "search is broken" into "this boost costs 0.037 NDCG, is it worth it".
Every firing is audited - rule id, owner, which documents moved and from where -
and the rule is *scoped* to one brand and one query pattern rather than a global
score multiplier, because a global boost changes every query a little and no
query in a way anyone can point at.

### Interleaving - and offline gains that don't survive it

The spec asks how you'd know offline NDCG gains are real before an A/B.
Team-draft interleaving, with position-biased simulated clicks and a sign test
over per-query preferences:

| comparison | offline NDCG delta | preference for A | p-value |
|---|---|---|---|
| routed hybrid vs bm25 | +0.0157 | 0.554 | **0.139** |
| routed hybrid vs dense only | +0.0162 | 0.507 | **0.893** |

**Both offline deltas are positive and neither preference is significant.** That
is the answer to the spec's question, and it is not the comfortable one: an
offline NDCG gain of this size does not necessarily show up as a detectable user
preference, and shipping on the offline number alone is a choice rather than a
default.

**Honest caveat, and it's large:** these clicks are *simulated from the relevance
labels*. That makes this a demonstration that the machinery works - drafting,
attribution, sign test - not evidence about user preference, because the
simulated user is defined to like what the labels like. On real traffic the whole
value of interleaving is that clicks and labels *disagree*, and nothing here can
show that.

### Spelling correction

150 queries with one character deleted from a token. The corrector is
deliberately **conservative** - it fires only when the token is out-of-vocabulary
*and* exactly one in-vocabulary word sits at edit distance one. Aggressive
correction is worse than none: a "corrected" query that changes intent produces
confidently wrong results and the user cannot tell what happened. Typos with no
unambiguous fix fall through to the semantic leg, which is what the rescue ladder
is for.

### The query-debug view

The spec calls this the internal tool every search team builds; the first pass
shipped a permutation-importance table, which answers a different question - what
the model uses *on average*. Debugging a complaint needs to know why **this**
document beat **that** one for **this** query. The view prints, per result: every
leg's raw score, the graded label, which boost rules fired and what they moved,
and per-feature contributions by **occlusion against the candidate-set median**.

That is *not* SHAP - it's blind to interactions and is labelled an approximation
everywhere it appears, because calling it SHAP would be a nicer word and a false
one. The baseline is this query's candidates rather than a global median, because
the question is why this document won *among these*.

## The other ~50% - what is still NOT here

- **No API and no storefront.** The debug view is a text renderer, not a UI.
- **No real ANN index** (no FAISS/HNSW), so no index size or build time, and the
  latency story would change entirely at catalogue scale.
- **Interleaving clicks are simulated from labels** (see above), so it
  demonstrates machinery rather than measuring preference.
- **The router is fitted on 14 need queries** and its weights are corner
  solutions; both are stated in the report but neither is fixed.
- **Boosting has no budget or fatigue control** - the rule fires on every
  matching query forever, with no cap on how much of the top-k any one brand can
  occupy across a session.
- **The speller handles edit distance 1 only**, no transpositions, no phonetic
  matching, no learned correction from query logs.
- **Popularity is generated, not observed**, so the feedback-loop story is
  reasoned about rather than demonstrated.
- **Single-locale, single-language, title-only.** No reviews, no images, no
  structured attribute matching beyond token overlap.
