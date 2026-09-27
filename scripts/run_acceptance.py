# ============================================================================
# 文件职责：运行项目最终离线验收案例，输出每项的实际验证结果。
# 主要调用方：用户手工运行 ``python scripts/run_acceptance.py``，也被最终验收测试引用。
# 输入/输出：输入 evaluation/final_acceptance_cases.json；输出每个案例通过/失败和退出码。
# 不负责：不发起真实大模型请求，不修改生产审计库或正式向量库。
# ============================================================================
"""最终离线验收脚本。

运行：``python scripts/run_acceptance.py``。它用真实 CSV、模型、RAG 和优惠规则，
配合离线模型替身验证至少十五个端到端场景，不调用真实 DeepSeek。
"""

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
    """故意失败的 RAG 替身，用于验收工具异常会被结构化返回。"""
    def retrieve(self, *_: object, **__: object) -> list[object]:
        """模拟检索服务故障，供异常验收项调用。"""
        raise RuntimeError("验收用模拟检索异常")


class AcceptanceContext:
    """复用验收中真实工具实例和离线 Agent 结果，减少各用例重复初始化。"""
    def __init__(self) -> None:
        """绑定正式项目文件；只缓存 Agent 结果，查询和预测仍按用例执行。"""
        self.lookup_tool = CustomerLookupTool(
            PROJECT_ROOT / "telco_customer_churn.csv"
        )
        self.prediction_tool = ChurnPredictionTool(
            PROJECT_ROOT / "churn_pipeline.joblib"
        )
        self.offer_tool = OfferCalculationTool()
        self._agent_result: Any | None = None

    def lookup(self, customer_id: str) -> Any:
        """调用真实客户查询工具，返回其完整结构化结果。"""
        return self.lookup_tool.run(CustomerLookupInput(customer_id=customer_id))

    def predict(self, customer_id: str) -> Any:
        """先查询再调用真实 Pipeline，确保预测输入与实际调用链一致。"""
        customer = self.lookup(customer_id)
        assert customer.success and customer.customer is not None
        return self.prediction_tool.run(
            ChurnPredictionInput(features=customer.customer.features)
        )

    def agent_result(self) -> Any:
        """首次构造离线 Agent 运行完整链路，后续验收用例复用同一结果。"""
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
    """验收存在客户、19 个模型特征以及标签未泄露到输入中。"""
    result = context.lookup("7590-VHVEG")
    assert result.success and result.customer is not None
    assert len(result.customer.features.__class__.model_fields) == 19
    assert "Churn" not in result.model_dump_json(by_alias=True)
    return "客户存在，返回19个模型字段且不含Churn"


def _check_customer_missing(context: AcceptanceContext) -> str:
    """验收不存在客户时返回明确 ``CUSTOMER_NOT_FOUND`` 错误码。"""
    result = context.lookup("0000-AAAAA")
    assert not result.success and result.error is not None
    assert result.error.code == "CUSTOMER_NOT_FOUND"
    return result.error.code


def _check_risk(
    context: AcceptanceContext,
    customer_id: str,
    expected_level: str,
) -> str:
    """验收指定客户由真实预测工具得到预期的风险等级。"""
    result = context.predict(customer_id)
    assert result.success and result.prediction is not None
    assert result.prediction.risk_level == expected_level
    return (
        f"{expected_level}，probability="
        f"{result.prediction.churn_probability:.6f}"
    )


def _check_prediction_tool(context: AcceptanceContext) -> str:
    """验收完整 Agent 调用记录中包含成功的流失预测工具。"""
    result = context.agent_result()
    assert result.success
    record = next(
        item for item in result.tool_calls if item.tool_name == PREDICT_CHURN_RISK
    )
    assert record.success
    return "predict_churn_risk 调用成功"


def _check_policy(_: AcceptanceContext) -> str:
    """验收 RAG 命中挽留政策文件，且结果带标题、来源和正文。"""
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
    """验收无关问题不会伪造证据，而是返回空 evidence。"""
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
    """验收月费刚好跨过 75 边界时选择不同的中风险优惠规则。"""
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
    """验收 RAG 工具异常会变成 ``KNOWLEDGE_RETRIEVAL_ERROR``。"""
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
    """在临时 SQLite 中验收人工确认或拒绝会持久化为预期状态。"""
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
    """验收完整方案严格按查询、预测、检索、优惠顺序调用四个工具。"""
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
    """验收模型反复请求工具时会触发最大调用次数安全上限。"""
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
    """读取验收清单，将案例 ID 映射到真实检查函数并收集通过/失败结果。"""
    manifest = json.loads(
        (
            PROJECT_ROOT / "evaluation" / "final_acceptance_cases.json"
        ).read_text(encoding="utf-8")
    )
    context = AcceptanceContext()
    # JSON 清单中的 ID 必须和此处处理函数一一对应，防止漏跑用例。
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
    """打印每个验收项及汇总；只要有一项失败就返回退出码 1。"""
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
