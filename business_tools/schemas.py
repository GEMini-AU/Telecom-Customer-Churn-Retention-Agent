"""Pydantic input and output contracts for standalone business tools."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .errors import ToolError


YesNo = Literal["Yes", "No"]
InternetService = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
RiskLevel = Literal["低风险", "中风险", "高风险"]


class _Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ChurnFeatures(_Schema):
    """The exact 19 feature values expected by the saved Pipeline."""

    senior_citizen: Literal[0, 1] = Field(alias="SeniorCitizen")
    tenure: int = Field(ge=0, alias="tenure")
    monthly_charges: float = Field(ge=0, alias="MonthlyCharges")
    total_charges: float = Field(ge=0, alias="TotalCharges")
    gender: Literal["Female", "Male"] = Field(alias="gender")
    partner: YesNo = Field(alias="Partner")
    dependents: YesNo = Field(alias="Dependents")
    phone_service: YesNo = Field(alias="PhoneService")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        alias="MultipleLines"
    )
    internet_service: InternetService = Field(alias="InternetService")
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        alias="OnlineSecurity"
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        alias="OnlineBackup"
    )
    device_protection: Literal[
        "Yes", "No", "No internet service"
    ] = Field(alias="DeviceProtection")
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        alias="TechSupport"
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        alias="StreamingTV"
    )
    streaming_movies: Literal[
        "Yes", "No", "No internet service"
    ] = Field(alias="StreamingMovies")
    contract: ContractType = Field(alias="Contract")
    paperless_billing: YesNo = Field(alias="PaperlessBilling")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(alias="PaymentMethod")


class CustomerLookupInput(_Schema):
    customer_id: str = Field(min_length=1, max_length=64)


class CustomerProfile(_Schema):
    customer_id: str
    features: ChurnFeatures


class CustomerLookupOutput(_Schema):
    success: bool
    customer: CustomerProfile | None = None
    error: ToolError | None = None


class ChurnPredictionInput(_Schema):
    features: ChurnFeatures


class ChurnPrediction(_Schema):
    churn_probability: float = Field(ge=0, le=1)
    classification_threshold: float = Field(ge=0, le=1)
    predicted_class: Literal[0, 1]
    prediction_label: Literal["预计不流失", "预计流失"]
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)


class ChurnPredictionOutput(_Schema):
    success: bool
    prediction: ChurnPrediction | None = None
    error: ToolError | None = None


class KnowledgeSearchInput(_Schema):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.10, ge=0, le=1)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question 不能为空。")
        return value.strip()


class KnowledgeEvidence(_Schema):
    chunk_id: str
    document_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_file: str
    document_title: str
    section_title: str
    text: str
    score: float = Field(ge=0, le=1)


class KnowledgeSearchOutput(_Schema):
    success: bool
    has_sufficient_evidence: bool
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)
    error: ToolError | None = None


class OfferCalculationInput(_Schema):
    risk_level: RiskLevel
    contract: ContractType
    monthly_charges: Decimal = Field(ge=0, max_digits=10, decimal_places=2)


class OfferPlan(_Schema):
    offer_code: str
    offer_name: str
    discount_rate: Decimal = Field(ge=0, le=1)
    duration_months: int = Field(ge=0)
    monthly_discount: Decimal = Field(ge=0)
    total_discount: Decimal = Field(ge=0)
    currency_note: str
    stackable: bool
    requires_manual_approval: bool
    rationale: str
    disclaimer: str


class OfferCalculationOutput(_Schema):
    success: bool
    offer: OfferPlan
