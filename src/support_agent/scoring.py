"""Deterministic confidence score (small local LLMs are poorly calibrated when self-reporting)."""

from __future__ import annotations

from typing import Any

W_AGREEMENT, W_RETRIEVAL, W_GROUNDING, W_CLASSIFICATION = 0.4, 0.2, 0.3, 0.1
_AGREEMENT = {"high": 1.0, "mixed": 0.6, "low": 0.25}


def compute_confidence(
    assessment: dict[str, Any],
    grounding_score: float,
    classification: dict[str, Any],
    evidence: list[dict],
    escalated: bool = False,
) -> float:
    agree = _AGREEMENT.get(assessment.get("agreement", "low"), 0.25)
    # blend qualitative level with measured answer overlap
    agree = 0.7 * agree + 0.3 * min(1.0, assessment.get("answer_agreement", 0.0) / 0.4)
    top = sorted((e.get("similarity", 0.0) for e in evidence), reverse=True)[:3]
    retrieval = min(1.0, (sum(top) / len(top)) / 0.8) if top else 0.0
    dist = assessment.get("queue_distribution", {})
    cls_certainty = dist.get(classification.get("queue"), 0.0) if dist else 0.0
    score = (
        W_AGREEMENT * agree
        + W_RETRIEVAL * retrieval
        + W_GROUNDING * max(0.0, min(1.0, grounding_score))
        + W_CLASSIFICATION * cls_certainty
    )
    if escalated:
        score = min(score, 0.4)
    return round(max(0.0, min(1.0, score)), 2)


def is_trusted(
    accepted: bool, escalated: bool, grounded: bool, confidence: float, threshold: float
) -> bool:
    return accepted and not escalated and grounded and confidence >= threshold
