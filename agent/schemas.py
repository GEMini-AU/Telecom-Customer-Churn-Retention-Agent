# ============================================================================
# 文件职责：定义 Agent 方案、工具调用记录和错误结果的数据结构。
# 主要调用方：single_agent.py 生成结果；agent_app.py 与审计模块读取结果。
# 输入/输出：Pydantic 校验输入；输出固定字段的 RetentionPlan、AgentRunResult 等对象。
# 不负责：不调用模型或工具，也不包含业务计算规则。
# ============================================================================
"""单 Agent 编排层的结构化数据契约。

调用链：四个工具的结果经 ``SingleRetentionAgent`` 校验后，组成 ``RetentionPlan``；
``agent_app.py`` 再读取该方案展示页面、创建审计记录和处理人工确认。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from business_tools.schemas import OfferPlan, RiskLevel


class _AgentSchema(BaseModel):
    """所有 Agent 结构的共同 Pydantic 配置基类，不直接实例化。"""

    # 所有 Agent 结果共用：拒绝未定义字段，并统一清理字符串首尾空白。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerBasicInfo(_AgentSchema):
    """最终方案中允许展示的七项客户基本信息，均来自客户查询工具。"""
    customer_id: str
    tenure_months: int = Field(ge=0)
    contract: str
    monthly_charges: float = Field(ge=0)
    internet_service: str
    tech_support: str
    payment_method: str


class RiskAssessment(_AgentSchema):
    """预测工具输出的概率、阈值、标签、等级和风险因素。"""
    churn_probability: float = Field(ge=0, le=1)
    classification_threshold: float = Field(ge=0, le=1)
    prediction_label: Literal["预计不流失", "预计流失"]
    risk_level: RiskLevel
    risk_factors: list[str]


class PolicyCitation(_AgentSchema):
    """最终方案引用的一条 RAG 证据；正文和来源必须逐字来自检索结果。"""
    chunk_id: str
    source_file: str
    document_title: str
    section_title: str
    excerpt: str


class RetentionPlan(_AgentSchema):
    """Agent 最终可交给页面与审计层的完整挽留方案。"""

    # Agent 最终方案只能由已执行工具返回的事实、政策和优惠组成。
    customer_basic: CustomerBasicInfo
    risk_assessment: RiskAssessment
    policy_evidence: list[PolicyCitation] = Field(min_length=1)
    recommended_offer: OfferPlan
    communication_script: str = Field(min_length=10)
    next_actions: list[str] = Field(min_length=1, max_length=5)
    disclaimer: str = Field(min_length=1)


class AgentError(_AgentSchema):
    """Agent 中断时的标准错误：``code`` 供程序分支，``message`` 供页面展示。"""
    code: str
    message: str


class ToolCallRecord(_AgentSchema):
    """一次模型请求工具后的可追踪执行记录。"""

    # 一次工具调用的可审计记录；页面用它展示 Agent 实际做过的步骤。
    call_index: int = Field(ge=1)
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    error_code: str | None = None


class AgentRunResult(_AgentSchema):
    """``run`` 的统一返回：成功时有方案，失败时有错误和已执行工具记录。"""

    # 统一承载成功方案或可展示的失败原因，避免页面直接处理异常对象。
    success: bool
    plan: RetentionPlan | None = None
    error: AgentError | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
