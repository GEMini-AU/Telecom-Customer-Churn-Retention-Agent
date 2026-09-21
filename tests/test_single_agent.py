"""Offline end-to-end and safety tests for the single Agent."""

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
    def test_five_complete_tasks_select_correct_tools(self) -> None:
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
