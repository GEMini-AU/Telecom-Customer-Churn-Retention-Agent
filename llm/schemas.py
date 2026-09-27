# ============================================================================
# 文件职责：约束大模型调用前的客户事实和调用后的挽留建议 JSON 结构。
# 主要调用方：deepseek_client.py 构造提示与校验模型输出。
# 输入/输出：输入 Python 字典/模型文本；输出 CustomerRiskContext 或 RetentionAdvice。
# 不负责：不读取环境变量、不发送网络请求，也不计算流失概率。
# ============================================================================
"""大模型挽留建议的输入与输出结构。

调用链：业务工具/页面先构造 ``CustomerRiskContext``，再交给
``DeepSeekRetentionClient``；模型文本必须能解析成 ``RetentionAdvice``。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# 复用项目统一的三档风险文字，禁止模型输出自定义风险标签。
RiskLevel = Literal["低风险", "中风险", "高风险"]


class CustomerRiskContext(BaseModel):
    """模型调用前已确认的客户事实；不是由模型推测得到的内容。"""

    # 禁止模型调用端混入未定义字段，确保上下文只含已确认的业务事实。
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
    """模型可补充的建议与话术；结构错误会在客户端立即被拒绝。"""

    # 这是大模型输出的边界：风险等级和原因仍须符合预先定义的类型约束。
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
