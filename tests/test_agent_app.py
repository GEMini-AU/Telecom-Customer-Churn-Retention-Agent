"""Streamlit smoke tests for the independent Agent page."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent.audit import calculate_plan_fingerprint
from agent.single_agent import SingleRetentionAgent
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "agent_app.py"


class AgentAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-app-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        cls.result = agent.run("请为客户 7590-VHVEG 生成完整挽留方案。")
        assert cls.result.plan is not None

    def test_page_renders_plan_and_decision_buttons(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
        app.session_state["agent_result"] = self.result
        app.session_state["decision_state"] = {
            "plan_fingerprint": calculate_plan_fingerprint(self.result.plan),
            "status": "pending_confirmation",
            "audit_id": None,
        }
        app.run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertIn("确认执行", [item.label for item in app.button])
        self.assertIn("拒绝方案", [item.label for item in app.button])
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["客户ID"], "7590-VHVEG")
        self.assertEqual(metrics["风险等级"], "高风险")
        self.assertTrue(any("工具调用记录" == item.label for item in app.expander))

    def test_missing_key_is_shown_without_page_exception(self) -> None:
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            app = AppTest.from_file(str(APP_PATH)).run(timeout=20)
            app.button[0].click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Agent 配置失败" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
