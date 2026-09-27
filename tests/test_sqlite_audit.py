# ============================================================================
# 文件职责：验证 SQLite 人工确认记录的事务、幂等、拒绝锁定、方案变化和导出边界。
# 主要调用方：``python -m unittest`` 自动发现或手工指定本测试模块。
# 输入/输出：输入离线完整方案和临时数据库；输出 unittest 通过/失败结果。
# 不负责：不改写项目正式审计库，不触发页面按钮或真实运营商动作。
# ============================================================================
"""SQLite 人工确认审计的事务与持久化离线测试。"""

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
    """验证待确认状态、幂等点击、拒绝锁定、方案变更和导出边界。"""
    @classmethod
    def setUpClass(cls) -> None:
        """生成一份离线完整方案，供各临时 SQLite 数据库测试使用。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-sqlite-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run("请为客户 7590-VHVEG 生成完整挽留方案。")
        assert result.success and result.plan is not None
        cls.plan = result.plan

    def test_pending_then_confirmed_is_persistent(self) -> None:
        """验证 pending 记录确认后重开数据库仍能读取确认状态。"""
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
        """同一方案重复创建或重复确认均不创建第二条或改写时间。"""
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
        """方案已拒绝后不能再改为确认执行。"""
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
        """页面方案指纹变化后，旧待确认审计记录不能被用于确认。"""
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
        """数据库路径不可写时抛异常，不能把方案误标为已确认。"""
        with tempfile.TemporaryDirectory(prefix="sqlite-audit-test-") as temp:
            invalid_path = Path(temp) / "directory.sqlite3"
            invalid_path.mkdir()
            with self.assertRaises(sqlite3.OperationalError):
                create_pending_decision(invalid_path, self.plan)

    def test_export_finalized_records_without_overwriting_legacy_log(self) -> None:
        """导出仅包含最终状态，并验证无关旧 JSONL 文件不被覆盖。"""
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
