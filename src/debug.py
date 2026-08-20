"""The query-debug view -- the internal tool every search team lives in.

The spec calls this out as a differentiator and the first pass shipped a
permutation-importance table instead, which answers a different question. Feature
importance tells you what the model uses ON AVERAGE. A search engineer debugging
a complaint needs to know why THIS document beat THAT one for THIS query, and no
aggregate can answer that.

What this prints, per result:
  - the raw score from every retrieval leg
  - the per-feature contribution to the reranker's score
  - the graded label, when the query is in the judgment set
  - which governed business rules fired and what they moved

The per-feature contribution uses OCCLUSION: re-score the document with one
feature replaced by its median across the candidate set, and attribute the
difference. That is not SHAP -- it is blind to interactions and it is labelled
as an approximation everywhere it appears, because calling it SHAP would be a
nicer word and a false one.
"""
from __future__ import annotations

import numpy as np

from .rank import FEATURES


def occlusion_contributions(model, X: np.ndarray, row: int) -> dict:
    """Per-feature contribution for one candidate, by occlusion against the
    candidate set's median.

    The baseline is the median of THIS query's candidates rather than a global
    median, because the question is "why did this document win among these
    candidates", not "why is this document good in the abstract".
    """
    base = model.predict(X[row:row + 1])[0]
    med = np.median(X, axis=0)
    out = {}
    for j, name in enumerate(FEATURES):
        probe = X[row:row + 1].copy()
        probe[0, j] = med[j]
        out[name] = float(base - model.predict(probe)[0])
    return out


def explain_query(query: dict, ranked: list[str], X: np.ndarray,
                  cand_ids: list[str], model, prod_by_id: dict,
                  bm25_scores: dict, dense_scores: dict,
                  labels: dict | None = None, top_n: int = 5,
                  boost_audit: list | None = None,
                  spelling: tuple | None = None,
                  query_type: str | None = None,
                  fusion_weight: float | None = None) -> str:
    """One query, fully explained. This is the surface a search engineer uses."""
    idx_of = {pid: i for i, pid in enumerate(cand_ids)}
    lines = []
    lines.append("=" * 74)
    lines.append("QUERY: %r" % query["text"])
    meta = ["kind=%s" % query.get("kind", "?"),
            "segment=%s" % query.get("segment", "?")]
    if query_type:
        meta.append("routed=%s" % query_type)
    if fusion_weight is not None:
        meta.append("lexical_weight=%.2f" % fusion_weight)
    lines.append("  " + "  ".join(meta))
    if spelling and spelling[1]:
        lines.append("  SPELLING: %s" % ", ".join("%s -> %s" % c for c in spelling[1]))
    if boost_audit:
        for b in boost_audit:
            if b.get("fired"):
                lines.append("  BOOST RULE %s (owner=%s) promoted %s from positions %s"
                             % (b["rule_id"], b["owner"], b["promoted"],
                                b["from_positions"]))
            elif b:
                lines.append("  BOOST RULE %s did not fire: %s"
                             % (b["rule_id"], b.get("reason", "-")))
    lines.append("")

    for rank, pid in enumerate(ranked[:top_n], start=1):
        p = prod_by_id[pid]
        grade = labels.get(pid) if labels else None
        gname = {3: "EXACT", 2: "SUBSTITUTE", 1: "COMPLEMENT", 0: "IRRELEVANT"}
        lines.append("%2d. %-46s %s" % (rank, p["title"][:46],
                                        gname.get(grade, "unjudged")
                                        if grade is not None else ""))
        lines.append("    category=%-16s brand=%-10s price=%7.2f pop=%.2f"
                     % (p["category"], p["brand"], p["price"], p["popularity"]))
        lines.append("    bm25=%8.3f   dense=%8.4f"
                     % (bm25_scores.get(pid, 0.0), dense_scores.get(pid, 0.0)))
        if pid in idx_of and model is not None:
            contrib = occlusion_contributions(model, X, idx_of[pid])
            top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:4]
            lines.append("    why (occlusion vs candidate median, NOT SHAP): %s"
                         % "  ".join("%s %+.3f" % (k, v) for k, v in top))
        lines.append("")
    return "\n".join(lines)


def zero_result_diagnosis(query: str, n_lexical: int, n_semantic: int,
                          spelling: tuple) -> str:
    """Why a query returned nothing, and what the rescue ladder did about it."""
    steps = []
    steps.append("  strict conjunctive match : %d results" % n_lexical)
    if spelling and spelling[1]:
        steps.append("  spelling correction      : %s"
                     % ", ".join("%s -> %s" % c for c in spelling[1]))
    else:
        steps.append("  spelling correction      : no unambiguous correction")
    steps.append("  semantic fallback        : %d results" % n_semantic)
    return "\n".join(steps)
