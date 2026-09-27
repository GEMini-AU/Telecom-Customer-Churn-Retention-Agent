# ============================================================================
# 文件职责：使用离线模型替身运行完整 Agent 任务，验证工具依赖、风险和数据来源。
# 主要调用方：用户手工运行 ``python scripts/test_agent.py``。
# 输入/输出：输入 evaluation/agent_tasks.json；输出每个案例的通过提示。
# 不负责：不使用真实 API Key、不访问网络，也不写入人工审计库。
# ============================================================================
"""离线运行训练样例与独立测试集样例的完整 Agent 任务。

运行：``python scripts/test_agent.py``。它使用测试替身模拟模型工具选择，仍调用真实
客户查询、预测、RAG 和优惠工具，因此不需要网络或真实 API Key。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.single_agent import SingleRetentionAgent
from agent.registry import (
    CALCULATE_RETENTION_OFFER,
    GET_CUSTOMER_PROFILE,
    PREDICT_CHURN_RISK,
    SEARCH_RETENTION_POLICY,
)
from business_tools.data_provenance import customer_source
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


def tools_follow_dependencies(actual: list[str], required: list[str]) -> bool:
    """只检查工具集合及真实依赖；政策检索与预测没有固定先后。"""
    if not set(required).issubset(actual):
        return False
    if not {
        GET_CUSTOMER_PROFILE, PREDICT_CHURN_RISK,
        SEARCH_RETENTION_POLICY, CALCULATE_RETENTION_OFFER,
    }.issubset(actual):
        return False
    return (
        actual.index(GET_CUSTOMER_PROFILE) < actual.index(PREDICT_CHURN_RISK)
        < actual.index(CALCULATE_RETENTION_OFFER)
    )


def main() -> None:
    """读取评估清单，逐个构造离线 Agent，并断言工具依赖和风险等级。"""
    cases = json.loads(
        (PROJECT_ROOT / "evaluation" / "agent_tasks.json").read_text(
            encoding="utf-8"
        )
    )
    # 每条用例单独创建 Agent，避免前一条的对话/状态影响后一条。
    for case in cases:
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-agent-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run(case["task"])
        assert result.success, result.error
        assert result.plan is not None
        # 离线替身恰好固定顺序，但验收标准必须允许真实模型先检索再预测。
        actual_tools = [item.tool_name for item in result.tool_calls]
        assert all(item.success for item in result.tool_calls), case["name"]
        assert tools_follow_dependencies(actual_tools, case["expected_tools"]), (
            case["name"], actual_tools
        )
        assert result.plan.risk_assessment.risk_level == case[
            "expected_risk_level"
        ]
        # 来源断言防止把训练集客户误写成独立测试集客户。
        source = customer_source(
            result.plan.customer_basic.customer_id,
            PROJECT_ROOT / "telco_customer_churn.csv",
        )
        if "expected_source" in case:
            assert source == case["expected_source"], (case["name"], source)
        print(f"通过：{case['name']} ｜ {source}")
    print(f"单 Agent 完整任务：{len(cases)}/{len(cases)} 通过。")


if __name__ == "__main__":
    main()
