"""Core Pydantic schemas for the application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RouteType = Literal["clarify", "refuse_or_defer", "general_medical", "medication"]
RiskLevel = Literal["low", "medium", "high"]


class MemorySummary(BaseModel):
    """Rolling conversation memory."""

    patient_context: str | None = Field(default=None, description="Important context.")
    last_topics: list[str] = Field(default_factory=list, description="Recent topics.")
    safety_flags: list[str] = Field(default_factory=list, description="Safety watch-outs.")


class RouterDecision(BaseModel):
    """Decision made by the router agent."""

    route: RouteType
    risk_level: RiskLevel
    rationale: str = Field(min_length=3)
    clarification_question: str | None = None
    memory_used: bool = False


class RetrievedChunk(BaseModel):
    """Evidence chunk from retrieval tool."""

    chunk_id: str
    source: str
    title: str
    snippet: str


class DrugInfoResult(BaseModel):
    """Stub output for drug lookup."""

    drug_name: str
    uses: list[str] = Field(default_factory=list)
    common_side_effects: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    disclaimer: str


class Citation(BaseModel):
    """Citation attached to answer text."""

    citation_id: str
    source: str
    claim: str


class DraftMedicalAnswer(BaseModel):
    """Structured draft answer from specialist agent."""

    direct_answer: str = Field(min_length=10)
    evidence_summary: str = Field(min_length=5)
    citations: list[Citation] = Field(default_factory=list)
    safety_note: str = Field(min_length=5)
    confidence: float = Field(ge=0.0, le=1.0)
    follow_up_question: str | None = None


class EvidenceValidationResult(BaseModel):
    """Result of lightweight evidence check."""

    supported: bool
    missing_claims: list[str] = Field(default_factory=list)
    notes: str


class SafetyCheckResult(BaseModel):
    """Safety verifier output."""

    approved: bool
    revised_answer: DraftMedicalAnswer
    reviewer_notes: str
    escalation_needed: bool = False


class SpecialistResult(BaseModel):
    """Specialist output including answer and collected evidence."""

    answer: DraftMedicalAnswer
    evidence: list[RetrievedChunk] = Field(default_factory=list)


class RouterAgentOutput(BaseModel):
    """Final router output after delegation and safety checks."""

    route: RouteType
    risk_level: RiskLevel
    rationale: str = Field(min_length=3)
    answer: DraftMedicalAnswer


class PipelineResult(BaseModel):
    """Top-level orchestration output."""

    route: RouteType
    risk_level: RiskLevel
    answer: DraftMedicalAnswer
    memory: MemorySummary

