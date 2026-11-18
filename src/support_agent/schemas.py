"""Shared Pydantic schemas (final output + intermediate structured LLM outputs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Classification(BaseModel):
    queue: str = Field(description="Support queue / department that should own the ticket")
    type: str = Field(description="Ticket type: Incident, Request, Problem or Change")
    priority: str = Field(description="Priority: low, medium, high or critical")
    tags: list[str] = Field(default_factory=list, description="Up to 6 short topical tags")


class EvidenceItem(BaseModel):
    ticket_id: str
    source: Literal["hist", "live"] = "hist"
    queue: str | None = None
    type: str | None = None
    priority: str | None = None
    similarity: float = 0.0
    summary: str = ""
    answer_snippet: str = ""


class Escalation(BaseModel):
    required: bool = False
    reason: str | None = None
    target_queue: str | None = None


class Verification(BaseModel):
    grounded: bool = True
    grounding_score: float = 1.0
    unsupported_claims: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Minimum output required by the assessment (section 5)."""

    case_id: str
    classification: Classification
    evidence: list[EvidenceItem]
    analysis: str
    resolution: str
    escalation: Escalation
    confidence: float
    verification: Verification | None = None


# ---- intermediate structured outputs -------------------------------------------------


class TriageOutput(BaseModel):
    classification: Classification
    skills: list[str] = Field(
        description="1-2 skill names from the catalogue most useful for this ticket"
    )
    missing_info: list[str] = Field(
        default_factory=list, description="Facts a support agent would need but are absent"
    )


class AssessOutput(BaseModel):
    route: Literal["resolve", "ask_user", "retrieve_more", "escalate"]
    hypothesis: str = Field(description="Current best hypothesis about root cause / fix")
    question: str | None = Field(default=None, description="ONE focused clarification question")
    followup_query: str | None = Field(
        default=None, description="Targeted search query for another retrieval round"
    )
    reason: str = ""


class DraftResolution(BaseModel):
    analysis: str = Field(description="Concise reasoning that cites evidence ticket IDs")
    resolution: str = Field(description="Customer-ready response / recommended action")
    cited_ticket_ids: list[str] = Field(default_factory=list)
    escalation_required: bool = False
    escalation_reason: str | None = None
    target_queue: str | None = None


class VerifyOutput(BaseModel):
    grounding_score: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)


class MemoryCandidate(BaseModel):
    kind: Literal["environment", "pattern", "preference", "caveat", "outcome"]
    text: str = Field(description="One short, stable, reusable fact about the CUSTOMER")


class MemoryExtraction(BaseModel):
    items: list[MemoryCandidate] = Field(default_factory=list)
