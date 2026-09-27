# ============================================================================
# 文件职责：验证单 Agent 完整工具链及客户缺失、幻觉、未注册工具和循环等安全边界。
# 主要调用方：``python -m unittest`` 自动发现或手工指定本测试模块。
# 输入/输出：输入离线模型替身、真实工具和评估任务；输出 unittest 通过/失败结果。
# 不负责：不调用真实 DeepSeek，也不依赖 Streamlit 页面交互。
# ============================================================================
"""单 Agent 端到端与安全边界的离线测试。

离线替身模拟模型选择工具，真实注册表和四个业务工具仍会执行，因此可验证实际编排顺序。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm.deepseek_client import DeepSeekSettings

from agent.single_agent import SingleRetentionAgent
from tests.offline_agent_model import (
    LoopingToolCompletions,
    OfflineAgentCompletions,
    UnknownToolCompletions,
    build_fake_client,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_SETTINGS = DeepSeekSettings(api_key="offline-agent-test")


class SingleRetentionAgentTests(unittest.TestCase):
    """验证完整链路以及客户缺失、未知工具、循环和幻觉概率等停止边界。"""
    def test_five_complete_tasks_select_correct_tools(self) -> None:
        """读取五条任务清单，验证均按预期工具顺序生成方案。"""
        cases = json.loads(
            (PROJECT_ROOT / "evaluation" / "agent_tasks.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertGreaterEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case["name"]):
                agent = SingleRetentionAgent(
                    PROJECT_ROOT,
                    settings=OFFLINE_SETTINGS,
                    openai_client=build_fake_client(
                        OfflineAgentCompletions()
                    ),
                )
                result = agent.run(case["task"])
                self.assertTrue(result.success, result.error)
                self.assertIsNotNone(result.plan)
                assert result.plan is not None
                actual_tools = [item.tool_name for item in result.tool_calls]
                self.assertEqual(actual_tools, case["expected_tools"])
                self.assertEqual(
                    result.plan.risk_assessment.risk_level,
                    case["expected_risk_level"],
                )
                self.assertTrue(result.plan.policy_evidence)
                self.assertTrue(result.plan.recommended_offer.disclaimer)

    def test_customer_not_found_stops_after_first_tool(self) -> None:
        """客户查询失败后，Agent 应停止，不能继续预测或优惠计算。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=OFFLINE_SETTINGS,
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run("请为客户 0000-AAAAA 生成完整挽留方案。")
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "TOOL_EXECUTION_FAILED")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertFalse(result.tool_calls[0].success)

    def test_unregistered_tool_is_rejected(self) -> None:
        """模型请求白名单外工具时，Agent 必须拒绝且不执行。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=OFFLINE_SETTINGS,
            openai_client=build_fake_client(UnknownToolCompletions()),
        )
        result = agent.run("请分析客户 7590-VHVEG。")
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "UNREGISTERED_TOOL")

    def test_max_tool_calls_stops_loop(self) -> None:
        """重复工具请求达到上限时，Agent 返回明确安全错误。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=OFFLINE_SETTINGS,
            openai_client=build_fake_client(
                LoopingToolCompletions("7590-VHVEG")
            ),
            max_tool_calls=2,
        )
        result = agent.run("请分析客户 7590-VHVEG。")
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "MAX_TOOL_CALLS_EXCEEDED")
        self.assertEqual(len(result.tool_calls), 2)

    def test_hallucinated_probability_is_rejected(self) -> None:
        """最终 JSON 中概率被篡改时，逐字段验证应拒绝方案。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=OFFLINE_SETTINGS,
            openai_client=build_fake_client(
                OfflineAgentCompletions(tamper_probability=True)
            ),
        )
        result = agent.run("为客户 7590-VHVEG 生成完整挽留方案。")
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "FINAL_OUTPUT_INVALID")

    def test_missing_customer_id_returns_clear_error(self) -> None:
        """自然语言任务缺 customerID 时，应在调用模型前返回错误。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=OFFLINE_SETTINGS,
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run("请帮我生成一份客户挽留方案。")
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "CUSTOMER_ID_REQUIRED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
