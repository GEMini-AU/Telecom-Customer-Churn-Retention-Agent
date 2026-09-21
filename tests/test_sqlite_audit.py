"""Transactional and persistent human-decision audit tests."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agent.single_agent import SingleRetentionAgent
from agent.sqlite_audit import (
    DecisionConflictError,
    create_pending_decision,
    decide_pending_decision,
    export_decisions_jsonl,
    list_decision_records,
)
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SqliteAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-sqlite-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run("请为客户 7590-VHVEG 生成完整挽留方案。")
        assert result.success and result.plan is not None
        cls.plan = result.plan

    def test_pending_then_confirmed_is_persistent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            path = Path(temp) / "audit" / "decisions.sqlite3"
            pending = create_pending_decision(path, self.plan)
            self.assertEqual(pending.status, "pending_confirmation")
            self.assertIsNone(pending.decision)
            self.assertIsNone(pending.decided_at_utc)
            confirmed = decide_pending_decision(
                path, pending.audit_id, self.plan, "confirmed"
            )
            self.assertEqual(confirmed.status, "confirmed_execution")
            self.assertEqual(confirmed.decision, "confirmed")
            self.assertIsNotNone(confirmed.decided_at_utc)
            after_reopen = list_decision_records(path)
            self.assertEqual(len(after_reopen), 1)
            self.assertEqual(after_reopen[0].audit_id, pending.audit_id)
            self.assertEqual(after_reopen[0].plan_summary.offer_code,
                             self.plan.recommended_offer.offer_code)

    def test_duplicate_confirm_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            path = Path(temp) / "decisions.sqlite3"
            pending = create_pending_decision(path, self.plan)
            again = create_pending_decision(path, self.plan)
            self.assertEqual(pending.audit_id, again.audit_id)
            first = decide_pending_decision(
                path, pending.audit_id, self.plan, "confirmed"
            )
            second = decide_pending_decision(
                path, pending.audit_id, self.plan, "confirmed"
            )
            self.assertEqual(first.audit_id, second.audit_id)
            self.assertEqual(first.decided_at_utc, second.decided_at_utc)
            self.assertEqual(len(list_decision_records(path)), 1)

    def test_rejected_cannot_be_confirmed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            path = Path(temp) / "decisions.sqlite3"
            pending = create_pending_decision(path, self.plan)
            rejected = decide_pending_decision(
                path, pending.audit_id, self.plan, "rejected"
            )
            self.assertEqual(rejected.status, "rejected")
            with self.assertRaises(DecisionConflictError):
                decide_pending_decision(
                    path, pending.audit_id, self.plan, "confirmed"
                )
            self.assertEqual(list_decision_records(path)[0].status, "rejected")

    def test_changed_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            path = Path(temp) / "decisions.sqlite3"
            pending = create_pending_decision(path, self.plan)
            changed_plan = self.plan.model_copy(
                update={"communication_script": "这是一段发生变更的客服沟通话术。"}
            )
            with self.assertRaises(DecisionConflictError):
                decide_pending_decision(
                    path, pending.audit_id, changed_plan, "confirmed"
                )
            self.assertEqual(
                list_decision_records(path)[0].status,
                "pending_confirmation",
            )

    def test_database_write_error_does_not_confirm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            invalid_path = Path(temp) / "directory.sqlite3"
            invalid_path.mkdir()
            with self.assertRaises(sqlite3.OperationalError):
                create_pending_decision(invalid_path, self.plan)

    def test_export_finalized_records_without_overwriting_legacy_log(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            root = Path(temp)
            path = root / "decisions.sqlite3"
            legacy_path = root / "old_decisions.jsonl"
            legacy_path.write_text("legacy record\n", encoding="utf-8")
            first = create_pending_decision(path, self.plan)
            decide_pending_decision(path, first.audit_id, self.plan, "confirmed")
            changed_plan = self.plan.model_copy(
                update={"communication_script": "另一份演示方案的沟通话术。"}
            )
            second = create_pending_decision(path, changed_plan)
            decide_pending_decision(
                path, second.audit_id, changed_plan, "rejected"
            )
            export_path = root / "export.jsonl"
            self.assertEqual(export_decisions_jsonl(path, export_path), 2)
            records = [
                json.loads(line)
                for line in export_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {record["result_status"] for record in records},
                {"confirmed_execution", "rejected"},
            )
            self.assertEqual(legacy_path.read_text(encoding="utf-8"),
                             "legacy record\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
