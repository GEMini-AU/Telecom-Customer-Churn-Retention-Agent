"""Structured contracts for the single-agent orchestration layer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from business_tools.schemas import OfferPlan, RiskLevel


class _AgentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerBasicInfo(_AgentSchema):
    customer_id: str
    tenure_months: int = Field(ge=0)
    contract: str
    monthly_charges: float = Field(ge=0)
    internet_service: str
    tech_support: str
    payment_method: str


class RiskAssessment(_AgentSchema):
    churn_probability: float = Field(ge=0, le=1)
    classification_threshold: float = Field(ge=0, le=1)
    prediction_label: Literal["预计不流失", "预计流失"]
    risk_level: RiskLevel
    risk_factors: list[str]


class PolicyCitation(_AgentSchema):
    chunk_id: str
    source_file: str
    document_title: str
    section_title: str
    excerpt: str


class RetentionPlan(_AgentSchema):
    customer_basic: CustomerBasicInfo
    risk_assessment: RiskAssessment
    policy_evidence: list[PolicyCitation] = Field(min_length=1)
    recommended_offer: OfferPlan
    communication_script: str = Field(min_length=10)
    next_actions: list[str] = Field(min_length=1, max_length=5)
    disclaimer: str = Field(min_length=1)


class AgentError(_AgentSchema):
    code: str
    message: str


class ToolCallRecord(_AgentSchema):
    call_index: int = Field(ge=1)
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    error_code: str | None = None


class AgentRunResult(_AgentSchema):
    success: bool
    plan: RetentionPlan | None = None
    error: AgentError | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
