"""Cross-case evidence comparison: deterministic stats that expose agreement / conflict / gaps.

The LLM later *explains* these findings; the numbers are computed here so they are testable.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from math import sqrt

from pydantic import BaseModel, Field

from support_agent.schemas import EvidenceItem

_TOK = re.compile(r"[a-z0-9]{3,}")
_STOP = {
    "the",
    "and",
    "for",
    "you",
    "your",
    "that",
    "this",
    "with",
    "have",
    "has",
    "are",
    "was",
    "will",
    "can",
    "our",
    "please",
    "thank",
    "thanks",
    "regarding",
    "from",
    "not",
    "but",
    "they",
    "name",
    "dear",
    "customer",
    "support",
    "team",
    "assist",
    "assistance",
    "help",
    "issue",
    "would",
    "should",
    "could",
    "also",
    "any",
    "all",
    "let",
    "know",
    "further",
    "need",
    "kindly",
    "sincerely",
    "regards",
    "best",
    "following",
    "such",
    "may",
    "been",
    "which",
    "its",
    "there",
}


class EvidenceAssessment(BaseModel):
    n_evidence: int = 0
    agreement: str = "low"  # high | mixed | low
    answer_agreement: float = 0.0
    top_similarity: float = 0.0
    queue_distribution: dict[str, float] = Field(default_factory=dict)
    type_distribution: dict[str, float] = Field(default_factory=dict)
    dominant_queue: str | None = None
    dominant_type: str | None = None
    conflicts: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    supporting_ids: list[str] = Field(default_factory=list)


def _vec(text: str) -> Counter[str]:
    return Counter(t for t in _TOK.findall(text.lower()) if t not in _STOP)


def _cos(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a.keys() & b.keys())
    na, nb = sqrt(sum(v * v for v in a.values())), sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _dist(values: list[str | None]) -> dict[str, float]:
    vals = [v for v in values if v]
    if not vals:
        return {}
    c = Counter(vals)
    return {k: round(n / len(vals), 2) for k, n in c.most_common()}


def compare_evidence(items: list[EvidenceItem], min_similarity: float = 0.45) -> EvidenceAssessment:
    if not items:
        return EvidenceAssessment(coverage_gaps=["no historical case retrieved"])

    top = items[: min(6, len(items))]
    qd, td = _dist([i.queue for i in top]), _dist([i.type for i in top])
    dom_q = next(iter(qd), None)
    dom_t = next(iter(td), None)
    vecs = [_vec(i.answer_snippet) for i in top]
    pairs = [_cos(a, b) for a, b in combinations(vecs, 2)]
    ans_agree = round(sum(pairs) / len(pairs), 3) if pairs else 0.0
    top_sim = max(i.similarity for i in items)

    conflicts: list[str] = []
    gaps: list[str] = []
    if len(qd) > 1 and qd[dom_q] < 0.6:  # type: ignore[index]
        conflicts.append(f"queue split across similar cases: {qd}")
    if len(td) > 1 and td[dom_t] < 0.6:  # type: ignore[index]
        conflicts.append(f"ticket type split across similar cases: {td}")
    # find cases whose answer is dissimilar from the group's consensus
    if len(top) >= 3:
        for idx, v in enumerate(vecs):
            others = [_cos(v, w) for j, w in enumerate(vecs) if j != idx]
            if others and sum(others) / len(others) < 0.08:
                conflicts.append(
                    f"{top[idx].ticket_id} proposes a resolution unlike the other cases"
                )
    if top_sim < min_similarity:
        gaps.append(f"best match similarity {top_sim:.2f} is below {min_similarity}")
    if len(items) < 3:
        gaps.append(f"only {len(items)} relevant case(s) found")
    if ans_agree < 0.12 and len(top) >= 3:
        gaps.append("resolutions across cases share little common content")

    if not conflicts and not gaps and ans_agree >= 0.2 and top_sim >= 0.6:
        agreement = "high"
    elif top_sim < min_similarity or len(conflicts) >= 3 or len(items) < 2:
        agreement = "low"
    elif ans_agree >= 0.2 and top_sim >= 0.6 and len(conflicts) <= 2:
        # resolutions agree even though labels (queue/type) are noisy -> usable, but flag it
        agreement = "mixed"
    elif len(conflicts) >= 2:
        agreement = "low"
    else:
        agreement = "mixed"

    return EvidenceAssessment(
        n_evidence=len(items),
        agreement=agreement,
        answer_agreement=ans_agree,
        top_similarity=round(top_sim, 3),
        queue_distribution=qd,
        type_distribution=td,
        dominant_queue=dom_q,
        dominant_type=dom_t,
        conflicts=conflicts,
        coverage_gaps=gaps,
        supporting_ids=[i.ticket_id for i in top],
    )
