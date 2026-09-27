# ============================================================================
# 文件职责：定义四个业务工具共享的 Pydantic 输入输出结构和字段取值限制。
# 主要调用方：所有 business_tools 实现、agent/registry.py、测试和命令行脚本。
# 输入/输出：输入原始 Python 数据；输出经过类型、范围、别名校验的模型对象。
# 不负责：不读取 CSV、不加载 joblib，也不执行任何业务计算。
# ============================================================================
"""四个独立业务工具的输入输出契约。

调用链中的数据先经过这些 Pydantic 模型校验，再交给 CSV 查询、Pipeline、RAG 或
优惠规则。字段别名刻意对齐训练 CSV 列名，避免模型输入列静默错位。
"""

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


# 以下 Literal 是可复用的取值集合；运行时 Pydantic 会拒绝集合外的字符串。
YesNo = Literal["Yes", "No"]
InternetService = Literal["DSL", "Fiber optic", "No"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
RiskLevel = Literal["低风险", "中风险", "高风险"]


class _Schema(BaseModel):
    """四个工具请求/响应模型共用的字段验证和字符串清理配置。"""

    # 禁止多余字段、允许使用 CSV 列名 alias，并统一清理字符串首尾空白。
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class ChurnFeatures(_Schema):
    """保存的 Pipeline 期望的 19 个特征，别名对应 CSV 的原始列名。"""

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
    """客户查询请求：仅接受非空、长度受限的客户标识。"""

    # 客户查询工具的最小输入：只接受一位客户的标识。
    customer_id: str = Field(min_length=1, max_length=64)


class CustomerProfile(_Schema):
    """客户查询成功后的最小资料：标识加模型特征，不包含原始完整行。"""
    customer_id: str
    features: ChurnFeatures


class CustomerLookupOutput(_Schema):
    """客户查询的二选一结果：成功时有 ``customer``，失败时有 ``error``。"""
    success: bool
    customer: CustomerProfile | None = None
    error: ToolError | None = None


class ChurnPredictionInput(_Schema):
    """流失预测请求：必须携带客户查询阶段校验过的 19 个特征。"""

    # 预测工具不接收随意拼装的字典，只接收已校验的 ChurnFeatures。
    features: ChurnFeatures


class ChurnPrediction(_Schema):
    """Pipeline 原始概率经项目阈值转换后的业务预测结果。"""
    churn_probability: float = Field(ge=0, le=1)
    classification_threshold: float = Field(ge=0, le=1)
    predicted_class: Literal[0, 1]
    prediction_label: Literal["预计不流失", "预计流失"]
    risk_level: RiskLevel
    risk_factors: list[str] = Field(default_factory=list)


class ChurnPredictionOutput(_Schema):
    """流失预测的二选一结果：预测成功或模型/输入错误。"""
    success: bool
    prediction: ChurnPrediction | None = None
    error: ToolError | None = None


class KnowledgeSearchInput(_Schema):
    """RAG 检索请求：问题、返回数量及最低相似度门槛。"""

    # min_score 是证据门槛；低于门槛时应返回“依据不足”，而不是生成结论。
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.10, ge=0, le=1)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        """拒绝只由空白字符组成的问题，并返回去首尾空格后的文本。"""
        if not value.strip():
            raise ValueError("question 不能为空。")
        return value.strip()


class KnowledgeEvidence(_Schema):
    """一条可追溯 RAG 证据：含来源文件、标题、原文片段与相似度。"""
    chunk_id: str
    document_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_file: str
    document_title: str
    section_title: str
    text: str
    score: float = Field(ge=0, le=1)


class KnowledgeSearchOutput(_Schema):
    """知识检索结果；``has_sufficient_evidence`` 明确表示能否据此回答。"""
    success: bool
    has_sufficient_evidence: bool
    evidence: list[KnowledgeEvidence] = Field(default_factory=list)
    error: ToolError | None = None


class OfferCalculationInput(_Schema):
    """优惠计算请求：只包含规则允许使用的三项业务输入。"""

    # 优惠计算只接收工具确认过的风险、合同和月消费，不接收大模型生成的金额。
    risk_level: RiskLevel
    contract: ContractType
    monthly_charges: Decimal = Field(ge=0, max_digits=10, decimal_places=2)


class OfferPlan(_Schema):
    """普通 Python 规则生成的单一演示优惠方案及其人工确认限制。"""
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
    """优惠计算成功结果；Pydantic 输入校验失败会在调用工具前被拦截。"""
    success: bool
    offer: OfferPlan
