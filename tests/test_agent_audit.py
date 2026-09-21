"""Tests for append-only human decision audit records."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.audit import append_decision_record, calculate_plan_fingerprint
from agent.single_agent import SingleRetentionAgent
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AgentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-audit-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run("请为客户 7590-VHVEG 生成完整挽留方案。")
        assert result.plan is not None
        cls.plan = result.plan

    def test_confirmed_decision_writes_required_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-audit-test-") as temp:
            path = Path(temp) / "audit" / "decisions.jsonl"
            record = append_decision_record(path, self.plan, "confirmed")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["customer_id"], "7590-VHVEG")
            self.assertEqual(payload["decision"], "confirmed")
            self.assertEqual(
                payload["result_status"],
                "confirmed_execution",
            )
            self.assertEqual(payload["audit_id"], record.audit_id)
            self.assertTrue(payload["timestamp_utc"])
            self.assertTrue(payload["plan_summary"]["policy_sources"])

    def test_rejected_decision_appends_second_record(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-audit-test-") as temp:
            path = Path(temp) / "decisions.jsonl"
            first = append_decision_record(path, self.plan, "confirmed")
            second = append_decision_record(path, self.plan, "rejected")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotEqual(first.audit_id, second.audit_id)
            self.assertEqual(json.loads(lines[1])["result_status"], "rejected")

    def test_fingerprint_is_stable_for_same_plan(self) -> None:
        first = calculate_plan_fingerprint(self.plan)
        second = calculate_plan_fingerprint(self.plan)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_write_error_is_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-audit-test-") as temp:
            directory_as_file = Path(temp) / "decisions.jsonl"
            directory_as_file.mkdir()
            with self.assertRaises(OSError):
                append_decision_record(
                    directory_as_file,
                    self.plan,
                    "confirmed",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
