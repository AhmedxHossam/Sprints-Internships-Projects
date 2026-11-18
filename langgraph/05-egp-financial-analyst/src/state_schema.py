"""Validated wire models; graph checkpoints contain JSON primitives only."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ScratchpadEntry(WireModel):
    """Observable actions and outcomes, not private chain-of-thought."""
    step: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    tool_input: str
    observation: str


class ReviewVerdict(WireModel):
    is_approved: bool
    feedback_notes: str = Field(min_length=1, max_length=6000)


class Evidence(WireModel):
    source_id: str
    content: str
    source_date: str
    kind: Literal["cbe_training_document", "egx_historical_csv", "mock_fx"]


class FactClaim(WireModel):
    text: str = Field(min_length=1, max_length=2000)
    source_id: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)


class FinancialBrief(WireModel):
    title: str = Field(min_length=1, max_length=200)
    facts: list[FactClaim] = Field(min_length=1, max_length=12)
    analysis: list[str] = Field(max_length=6)
    caveats: list[str] = Field(min_length=1, max_length=10)

    def render(self) -> str:
        lines = [f"# {self.title}", "", "## Sourced facts"]
        lines.extend(f"- {fact.text} [{fact.source_id}]" for fact in self.facts)
        lines.extend(["", "## Analyst interpretation (not verified facts)"])
        lines.extend(f"- {item}" for item in self.analysis)
        lines.extend(["", "## Data limitations and safeguards"])
        lines.extend(f"- {item}" for item in self.caveats)
        return "\n".join(lines)


class QueryPlan(WireModel):
    queries: list[str] = Field(min_length=1, max_length=3)


class RetrievalAssessment(WireModel):
    sufficient: bool
    follow_up_query: str


class A2AMessage(WireModel):
    """Includes all required properties of the supplied A2A JSON schema."""
    sender: Literal["analyst", "reviewer"]
    recipient: Literal["analyst", "reviewer"]
    attempt: int = Field(ge=1)
    draft_report: str = Field(min_length=1)
    sources_cited: list[str] = Field(min_length=1)
    reviewer_feedback: str
    approval_status: bool
    scratchpad_memory: list[ScratchpadEntry]


class FXRequest(WireModel):
    base_currency: Literal["USD"] = "USD"


class FXQuote(WireModel):
    base: Literal["USD"]
    target: Literal["EGP"]
    rate: float = Field(gt=0, allow_inf_nan=False)
    inverse_rate: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["EGP per 1 USD"]
    source: Literal["TRAINING_MOCK_NOT_CBE"]
    last_updated: str
    is_mock: Literal[True]

    @field_validator("last_updated")
    @classmethod
    def timezone_required(cls, value: str) -> str:
        if datetime.fromisoformat(value).tzinfo is None:
            raise ValueError("FX timestamp must contain a timezone")
        return value


class HumanDecision(WireModel):
    action: Literal["publish", "reject"]
    reviewer: str = Field(min_length=1, max_length=200)
    draft_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("reviewer")
    @classmethod
    def named_reviewer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Human reviewer name is required")
        return value.strip()


def draft_digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class WorkflowState(TypedDict, total=False):
    request: str
    as_of: str
    iteration: int
    draft_report: str
    brief: dict
    sources_cited: list[str]
    evidence: list[dict]
    market_context: dict
    fx_quote: dict
    reviewer_feedback: str
    approval_status: bool
    review_history: list[dict]
    a2a_messages: list[dict]
    scratchpad_memory: list[dict]
    memory_step: int
    messages: list[dict]
    memory_summary: str
    retrieval_audits: list[dict]
    status: Literal["running", "pre_publish", "published", "rejected", "review_failed"]
    approved_draft_sha256: str
    human_decision: dict
    publication: dict
    demo_defect_injected: bool
