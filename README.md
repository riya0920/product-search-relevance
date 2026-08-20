# ML-2 — Product Search & Relevance Engine

**This is not deployable.** It is the first ~20% of the spec: a labelled eval
with segmented NDCG against a BM25 baseline, three retrieval legs, a re-ranker,
and the honesty tables. No storefront, no API. Missing 80% at the bottom.

```bash
python src/corpus.py        # ~2s   build the ESCI-shaped corpus
python run_search.py        # ~45s  the evaluation
python -m pytest tests -q   # 18 tests
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
vary by query type, which is a router this build does not have.

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

## The other 80% — what is NOT here

- **No API, no storefront, no query-debug UI.** The spec asks for the internal
  tool every search team builds; feature contributions exist as a permutation-
  importance table, not a per-result view.
- **No query-type router**, which the head/tail analysis says is the actual
  deployment answer.
- **No interleaving.** The spec asks how you'd know offline gains are real before
  an A/B — interleaving is the answer and it is named, not implemented.
- **No spelling correction**, which is the other half of a real rescue ladder.
- **No merchandiser boosting** — the governed, logged, NDCG-cost-quantified
  business rule the spec asks about.
- **No real ANN index** (no FAISS/HNSW), so no index size or build time, and the
  latency story would change entirely at catalogue scale.
- **Popularity is generated, not observed**, so the feedback-loop story is
  reasoned about rather than demonstrated.
- **Single-locale, single-language, title-only.** No reviews, no images, no
  structured attribute matching beyond token overlap.
