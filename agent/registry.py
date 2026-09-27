# ============================================================================
# 文件职责：注册并执行四个业务工具，保存本轮 Agent 已确认的权威事实。
# 主要调用方：agent/single_agent.py 的工具调用循环。
# 输入/输出：输入工具名、JSON 参数、AgentToolState；输出 ToolExecutionResult。
# 不负责：不决定工具调用顺序，不生成最终自然语言方案，也不直接操作 Streamlit。
# ============================================================================
"""四个业务工具的白名单注册表。

调用链：``SingleRetentionAgent.run`` 把模型工具请求交给本模块；本模块校验参数、
执行对应工具、保存权威结果到状态，并把结果转换成 Agent 可回传给模型的 JSON。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from business_tools import (
    ChurnPredictionInput,
    ChurnPredictionTool,
    CustomerLookupInput,
    CustomerLookupTool,
    KnowledgeRetrievalTool,
    KnowledgeSearchInput,
    OfferCalculationInput,
    OfferCalculationTool,
)
from business_tools.errors import ToolError
from business_tools.schemas import (
    ChurnPrediction,
    CustomerProfile,
    KnowledgeEvidence,
    OfferPlan,
)


# 这四个名称既是模型可见的工具名，也是执行层白名单的唯一入口。
GET_CUSTOMER_PROFILE = "get_customer_profile"
PREDICT_CHURN_RISK = "predict_churn_risk"
SEARCH_RETENTION_POLICY = "search_retention_policy"
CALCULATE_RETENTION_OFFER = "calculate_retention_offer"


class ToolExecutionResult(BaseModel):
    """注册表一次执行的统一结果：成功数据或 ``ToolError``。"""

    # 禁止不同适配函数私自塞入额外字段，保证 Agent 的处理逻辑稳定。
    model_config = ConfigDict(extra="forbid")

    success: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None


@dataclass
class AgentToolState:
    """一轮 Agent 任务的内存事实缓存，按客户 ID 或证据 ID 索引。"""

    # 只保存已经由工具确认的事实，供后续工具和最终方案校验复用。
    customers: dict[str, CustomerProfile] = field(default_factory=dict)
    predictions: dict[str, ChurnPrediction] = field(default_factory=dict)
    evidence: dict[str, KnowledgeEvidence] = field(default_factory=dict)
    offers: dict[str, OfferPlan] = field(default_factory=dict)


class BusinessToolRegistry:
    """只暴露四个已测试工具，并集中保存本轮 Agent 的权威业务事实。"""

    def __init__(
        self,
        project_root: Path,
        customer_tool: CustomerLookupTool | None = None,
        prediction_tool: ChurnPredictionTool | None = None,
        retrieval_tool: KnowledgeRetrievalTool | None = None,
        offer_tool: OfferCalculationTool | None = None,
    ) -> None:
        """按项目根目录创建真实工具；可注入替身用于离线测试。"""
        # 默认工具均使用项目内真实的 CSV、模型文件和本地 RAG 索引；测试可注入替身。
        self.project_root = project_root.resolve()
        self.customer_tool = customer_tool or CustomerLookupTool(
            self.project_root / "telco_customer_churn.csv"
        )
        self.prediction_tool = prediction_tool or ChurnPredictionTool(
            self.project_root / "churn_pipeline.joblib"
        )
        self.retrieval_tool = retrieval_tool or KnowledgeRetrievalTool(
            self.project_root
        )
        self.offer_tool = offer_tool or OfferCalculationTool()

    @property
    def allowed_tool_names(self) -> set[str]:
        """返回执行层允许的工具名称集合，供 Agent 做双重白名单校验。"""
        return {
            GET_CUSTOMER_PROFILE,
            PREDICT_CHURN_RISK,
            SEARCH_RETENTION_POLICY,
            CALCULATE_RETENTION_OFFER,
        }

    def create_state(self) -> AgentToolState:
        """为一轮新任务创建空状态，避免不同客户任务之间串数据。"""
        return AgentToolState()

    def tool_definitions(self) -> list[dict[str, Any]]:
        """返回 OpenAI 工具调用格式的定义；只描述能力，不执行能力。"""
        # 向模型公开工具说明和 JSON 参数结构，但不在这里执行工具。
        customer_schema = CustomerLookupInput.model_json_schema()
        search_schema = KnowledgeSearchInput.model_json_schema()
        return [
            _tool_definition(
                GET_CUSTOMER_PROFILE,
                "根据 customerID 查询客户，只返回预测和运营所需字段。",
                customer_schema,
            ),
            _tool_definition(
                PREDICT_CHURN_RISK,
                "使用已查询客户和现有模型计算流失概率与风险等级。调用前必须先查询客户。",
                customer_schema,
            ),
            _tool_definition(
                SEARCH_RETENTION_POLICY,
                "检索项目演示政策，返回带来源的知识片段。",
                search_schema,
            ),
            _tool_definition(
                CALCULATE_RETENTION_OFFER,
                "根据已查询客户和预测结果计算确定性的演示优惠。调用前必须完成客户查询和预测。",
                customer_schema,
            ),
        ]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        state: AgentToolState,
    ) -> ToolExecutionResult:
        """执行一个已注册工具，并把成功结果写入 ``state`` 供后续步骤依赖。"""
        # 二次白名单校验：即使模型返回异常工具名，也不会越过执行边界。
        if tool_name not in self.allowed_tool_names:
            return _error(
                "UNREGISTERED_TOOL",
                f"工具 {tool_name} 未注册，拒绝执行。",
            )
        try:
            if tool_name == GET_CUSTOMER_PROFILE:
                return self._get_customer(arguments, state)
            if tool_name == PREDICT_CHURN_RISK:
                return self._predict(arguments, state)
            if tool_name == SEARCH_RETENTION_POLICY:
                return self._search(arguments, state)
            return self._calculate_offer(arguments, state)
        except ValidationError as error:
            return _error("TOOL_INPUT_INVALID", str(error))
        except Exception as error:
            return _error(
                "TOOL_EXECUTION_ERROR",
                f"{tool_name} 执行异常：{type(error).__name__}：{error}",
            )

    def _get_customer(
        self,
        arguments: dict[str, Any],
        state: AgentToolState,
    ) -> ToolExecutionResult:
        """查询客户并缓存模型特征；预测和优惠只能使用这份缓存。"""
        # 客户查询成功后先写入状态，预测和优惠计算只能读取这里的权威客户特征。
        request = CustomerLookupInput.model_validate(arguments)
        result = self.customer_tool.run(request)
        if not result.success or result.customer is None:
            return ToolExecutionResult(success=False, error=result.error)
        state.customers[request.customer_id] = result.customer
        return _success(result.customer)

    def _predict(
        self,
        arguments: dict[str, Any],
        state: AgentToolState,
    ) -> ToolExecutionResult:
        """从已查询客户提取特征，调用现有 Pipeline 生成流失预测。"""
        # 预测依赖客户查询结果，禁止模型自行拼接特征绕过已有 Pipeline。
        request = CustomerLookupInput.model_validate(arguments)
        customer = state.customers.get(request.customer_id)
        if customer is None:
            return _error(
                "TOOL_DEPENDENCY_MISSING",
                "预测前必须先调用 get_customer_profile 查询同一客户。",
            )
        result = self.prediction_tool.run(
            ChurnPredictionInput(features=customer.features)
        )
        if not result.success or result.prediction is None:
            return ToolExecutionResult(success=False, error=result.error)
        state.predictions[request.customer_id] = result.prediction
        return _success(result.prediction)

    def _search(
        self,
        arguments: dict[str, Any],
        state: AgentToolState,
    ) -> ToolExecutionResult:
        """检索演示政策并按 ``chunk_id`` 保存可引用的原始证据。"""
        # 检索结果按 chunk_id 保存，后续最终方案只能引用本次实际返回的片段。
        request = KnowledgeSearchInput.model_validate(arguments)
        result = self.retrieval_tool.run(request)
        if not result.success:
            return ToolExecutionResult(success=False, error=result.error)
        for item in result.evidence:
            state.evidence[item.chunk_id] = item
        return ToolExecutionResult(
            success=True,
            data={
                "has_sufficient_evidence": result.has_sufficient_evidence,
                "evidence": [
                    item.model_dump(mode="json") for item in result.evidence
                ],
            },
        )

    def _calculate_offer(
        self,
        arguments: dict[str, Any],
        state: AgentToolState,
    ) -> ToolExecutionResult:
        """用同一客户的已确认风险、合同和月费执行确定性优惠计算。"""
        # 优惠金额由 Python 规则计算，输入必须来自同一客户的查询与预测结果。
        request = CustomerLookupInput.model_validate(arguments)
        customer = state.customers.get(request.customer_id)
        prediction = state.predictions.get(request.customer_id)
        if customer is None or prediction is None:
            return _error(
                "TOOL_DEPENDENCY_MISSING",
                "优惠计算前必须完成同一客户的查询和流失预测。",
            )
        result = self.offer_tool.run(
            OfferCalculationInput(
                risk_level=prediction.risk_level,
                contract=customer.features.contract,
                monthly_charges=customer.features.monthly_charges,
            )
        )
        state.offers[request.customer_id] = result.offer
        return _success(result.offer)


def _tool_definition(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """把名称、说明和 Pydantic JSON Schema 组装成模型可识别的工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _success(payload: BaseModel) -> ToolExecutionResult:
    """将任意 Pydantic 成功对象序列化为 JSON 兼容字典。"""
    return ToolExecutionResult(
        success=True,
        data=payload.model_dump(mode="json"),
    )


def _error(code: str, message: str) -> ToolExecutionResult:
    """创建统一的失败结果，调用方无需自行拼装 ``ToolError``。"""
    return ToolExecutionResult(
        success=False,
        error=ToolError(code=code, message=message),
    )
