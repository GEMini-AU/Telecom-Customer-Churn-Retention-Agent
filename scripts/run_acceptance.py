"""Run the final offline acceptance suite against real project components."""

from __future__ import annotations

import json
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.sqlite_audit import (
    create_pending_decision,
    decide_pending_decision,
    list_decision_records,
)
from agent.registry import PREDICT_CHURN_RISK
from agent.single_agent import SingleRetentionAgent
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
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import (
    LoopingToolCompletions,
    OfflineAgentCompletions,
    build_fake_client,
)


class _BrokenRAG:
    def retrieve(self, *_: object, **__: object) -> list[object]:
        raise RuntimeError("验收用模拟检索异常")


class AcceptanceContext:
    def __init__(self) -> None:
        self.lookup_tool = CustomerLookupTool(
            PROJECT_ROOT / "telco_customer_churn.csv"
        )
        self.prediction_tool = ChurnPredictionTool(
            PROJECT_ROOT / "churn_pipeline.joblib"
        )
        self.offer_tool = OfferCalculationTool()
        self._agent_result: Any | None = None

    def lookup(self, customer_id: str) -> Any:
        return self.lookup_tool.run(CustomerLookupInput(customer_id=customer_id))

    def predict(self, customer_id: str) -> Any:
        customer = self.lookup(customer_id)
        assert customer.success and customer.customer is not None
        return self.prediction_tool.run(
            ChurnPredictionInput(features=customer.customer.features)
        )

    def agent_result(self) -> Any:
        if self._agent_result is None:
            agent = SingleRetentionAgent(
                PROJECT_ROOT,
                settings=DeepSeekSettings(api_key="offline-acceptance"),
                openai_client=build_fake_client(OfflineAgentCompletions()),
            )
            self._agent_result = agent.run(
                "请为客户 7590-VHVEG 生成完整挽留方案。"
            )
        return self._agent_result


def _check_customer_found(context: AcceptanceContext) -> str:
    result = context.lookup("7590-VHVEG")
    assert result.success and result.customer is not None
    assert len(result.customer.features.__class__.model_fields) == 19
    assert "Churn" not in result.model_dump_json(by_alias=True)
    return "客户存在，返回19个模型字段且不含Churn"


def _check_customer_missing(context: AcceptanceContext) -> str:
    result = context.lookup("0000-AAAAA")
    assert not result.success and result.error is not None
    assert result.error.code == "CUSTOMER_NOT_FOUND"
    return result.error.code


def _check_risk(
    context: AcceptanceContext,
    customer_id: str,
    expected_level: str,
) -> str:
    result = context.predict(customer_id)
    assert result.success and result.prediction is not None
    assert result.prediction.risk_level == expected_level
    return (
        f"{expected_level}，probability="
        f"{result.prediction.churn_probability:.6f}"
    )


def _check_prediction_tool(context: AcceptanceContext) -> str:
    result = context.agent_result()
    assert result.success
    record = next(
        item for item in result.tool_calls if item.tool_name == PREDICT_CHURN_RISK
    )
    assert record.success
    return "predict_churn_risk 调用成功"


def _check_policy(_: AcceptanceContext) -> str:
    result = KnowledgeRetrievalTool(PROJECT_ROOT).run(
        KnowledgeSearchInput(question="高风险客户的演示挽留政策是什么？")
    )
    assert result.success and result.has_sufficient_evidence
    assert any(
        item.source_file.endswith("retention_policies.md")
        for item in result.evidence
    )
    assert all(item.text and item.document_title for item in result.evidence)
    return "命中 retention_policies.md 且来源字段完整"


def _check_no_evidence(_: AcceptanceContext) -> str:
    result = KnowledgeRetrievalTool(PROJECT_ROOT).run(
        KnowledgeSearchInput(
            question="火星量子通信卫星的轨道参数是多少？"
        )
    )
    assert result.success
    assert not result.has_sufficient_evidence
    assert result.evidence == []
    return "无依据，evidence=[]"


def _check_offer(
    context: AcceptanceContext,
    monthly_charges: Decimal,
    expected_code: str,
) -> str:
    result = context.offer_tool.run(
        OfferCalculationInput(
            risk_level="中风险",
            contract="Month-to-month",
            monthly_charges=monthly_charges,
        )
    )
    assert result.offer.offer_code == expected_code
    return result.offer.offer_code


def _check_tool_error(_: AcceptanceContext) -> str:
    result = KnowledgeRetrievalTool(
        PROJECT_ROOT,
        rag_service=_BrokenRAG(),
    ).run(KnowledgeSearchInput(question="高风险客户如何挽留？"))
    assert not result.success and result.error is not None
    assert result.error.code == "KNOWLEDGE_RETRIEVAL_ERROR"
    return result.error.code


def _check_audit(
    context: AcceptanceContext,
    decision: Literal["confirmed", "rejected"],
    expected_status: str,
) -> str:
    result = context.agent_result()
    assert result.success and result.plan is not None
    with tempfile.TemporaryDirectory(prefix="final-acceptance-audit-") as temp:
        path = Path(temp) / "decisions.sqlite3"
        pending = create_pending_decision(path, result.plan)
        assert pending.status == "pending_confirmation"
        record = decide_pending_decision(
            path,
            pending.audit_id,
            result.plan,
            decision,
        )
        stored = list_decision_records(path)
    assert len(stored) == 1
    assert stored[0].status == expected_status
    assert stored[0].customer_id == "7590-VHVEG"
    assert stored[0].decided_at_utc
    assert stored[0].plan_summary.policy_sources
    assert record.status == expected_status
    return expected_status


def _check_agent_chain(context: AcceptanceContext) -> str:
    result = context.agent_result()
    assert result.success and result.plan is not None
    actual = [item.tool_name for item in result.tool_calls]
    expected = [
        "get_customer_profile",
        "predict_churn_risk",
        "search_retention_policy",
        "calculate_retention_offer",
    ]
    assert actual == expected
    assert all(item.success for item in result.tool_calls)
    return " -> ".join(actual)


def _check_loop_limit(_: AcceptanceContext) -> str:
    agent = SingleRetentionAgent(
        PROJECT_ROOT,
        settings=DeepSeekSettings(api_key="offline-acceptance"),
        openai_client=build_fake_client(
            LoopingToolCompletions("7590-VHVEG")
        ),
        max_tool_calls=2,
    )
    result = agent.run("请分析客户 7590-VHVEG。")
    assert not result.success and result.error is not None
    assert result.error.code == "MAX_TOOL_CALLS_EXCEEDED"
    return result.error.code


def run_acceptance() -> list[dict[str, Any]]:
    manifest = json.loads(
        (
            PROJECT_ROOT / "evaluation" / "final_acceptance_cases.json"
        ).read_text(encoding="utf-8")
    )
    context = AcceptanceContext()
    handlers: dict[str, Callable[[], str]] = {
        "AC-001": lambda: _check_customer_found(context),
        "AC-002": lambda: _check_customer_missing(context),
        "AC-003": lambda: _check_risk(context, "7590-VHVEG", "高风险"),
        "AC-004": lambda: _check_risk(context, "8012-SOUDQ", "中风险"),
        "AC-005": lambda: _check_risk(context, "5575-GNVDE", "低风险"),
        "AC-006": lambda: _check_prediction_tool(context),
        "AC-007": lambda: _check_policy(context),
        "AC-008": lambda: _check_no_evidence(context),
        "AC-009": lambda: _check_offer(
            context, Decimal("74.99"), "MEDIUM_M2M_STANDARD"
        ),
        "AC-010": lambda: _check_offer(
            context, Decimal("75.00"), "MEDIUM_M2M_HIGH_USAGE"
        ),
        "AC-011": lambda: _check_tool_error(context),
        "AC-012": lambda: _check_audit(
            context, "confirmed", "confirmed_execution"
        ),
        "AC-013": lambda: _check_audit(context, "rejected", "rejected"),
        "AC-014": lambda: _check_agent_chain(context),
        "AC-015": lambda: _check_loop_limit(context),
    }
    assert len(manifest) >= 10
    assert {case["id"] for case in manifest} == set(handlers)

    results: list[dict[str, Any]] = []
    for case in manifest:
        case_id = case["id"]
        try:
            actual = handlers[case_id]()
        except Exception as error:
            results.append(
                {
                    "id": case_id,
                    "category": case["category"],
                    "passed": False,
                    "actual": f"{type(error).__name__}: {error}",
                }
            )
        else:
            results.append(
                {
                    "id": case_id,
                    "category": case["category"],
                    "passed": True,
                    "actual": actual,
                }
            )
    return results


def main() -> None:
    results = run_acceptance()
    for result in results:
        status = "通过" if result["passed"] else "失败"
        print(
            f"{status} {result['id']} {result['category']}：{result['actual']}"
        )
    passed = sum(item["passed"] for item in results)
    print(f"最终验收用例：{passed}/{len(results)} 通过。")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
