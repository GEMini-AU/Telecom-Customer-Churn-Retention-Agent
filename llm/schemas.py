"""Validated inputs and outputs for retention-advice generation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["低风险", "中风险", "高风险"]


class CustomerRiskContext(BaseModel):
    """Authoritative customer facts produced before the LLM is called."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, description="演示或业务系统中的客户标识")
    churn_probability: float = Field(ge=0, le=1, description="流失模型输出的概率")
    risk_level: RiskLevel = Field(description="由项目风险阈值计算出的风险等级")
    risk_reasons: list[str] = Field(
        min_length=1,
        description="已经由模型结果或业务规则确认的风险信息",
    )
    contract: str = Field(min_length=1, description="客户合同类型")
    tenure_months: int = Field(ge=0, description="客户使用时长，单位为月")
    monthly_charges: float = Field(ge=0, description="客户月费")
    tech_support: str = Field(min_length=1, description="技术支持服务状态")
    payment_method: str = Field(min_length=1, description="付款方式")


class RetentionAdvice(BaseModel):
    """Structured, validated retention advice returned to application code."""

    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel = Field(description="客户风险等级")
    risk_reasons: list[str] = Field(min_length=1, description="客户风险原因")
    recommended_actions: list[str] = Field(
        min_length=1,
        max_length=3,
        description="一到三条建议行动，不包含虚构优惠",
    )
    communication_script: str = Field(
        min_length=1,
        description="客服可参考的中文沟通话术",
    )
