"""Allow-listed adapters around the four tested business tools."""

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


GET_CUSTOMER_PROFILE = "get_customer_profile"
PREDICT_CHURN_RISK = "predict_churn_risk"
SEARCH_RETENTION_POLICY = "search_retention_policy"
CALCULATE_RETENTION_OFFER = "calculate_retention_offer"


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None


@dataclass
class AgentToolState:
    customers: dict[str, CustomerProfile] = field(default_factory=dict)
    predictions: dict[str, ChurnPrediction] = field(default_factory=dict)
    evidence: dict[str, KnowledgeEvidence] = field(default_factory=dict)
    offers: dict[str, OfferPlan] = field(default_factory=dict)


class BusinessToolRegistry:
    """Expose exactly four registered tools and keep authoritative state."""

    def __init__(
        self,
        project_root: Path,
        customer_tool: CustomerLookupTool | None = None,
        prediction_tool: ChurnPredictionTool | None = None,
        retrieval_tool: KnowledgeRetrievalTool | None = None,
        offer_tool: OfferCalculationTool | None = None,
    ) -> None:
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
        return {
            GET_CUSTOMER_PROFILE,
            PREDICT_CHURN_RISK,
            SEARCH_RETENTION_POLICY,
            CALCULATE_RETENTION_OFFER,
        }

    def create_state(self) -> AgentToolState:
        return AgentToolState()

    def tool_definitions(self) -> list[dict[str, Any]]:
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
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _success(payload: BaseModel) -> ToolExecutionResult:
    return ToolExecutionResult(
        success=True,
        data=payload.model_dump(mode="json"),
    )


def _error(code: str, message: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        success=False,
        error=ToolError(code=code, message=message),
    )
