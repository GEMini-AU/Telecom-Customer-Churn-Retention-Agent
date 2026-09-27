# ============================================================================
# 文件职责：验证旧版 JSONL 审计记录能追加、保留字段并在写入失败时抛错。
# 主要调用方：``python -m unittest`` 自动发现或手工指定本测试模块。
# 输入/输出：输入临时目录和离线方案；输出 unittest 通过/失败结果。
# 不负责：不测试当前页面使用的 SQLite 审计状态机。
# ============================================================================
"""旧版 JSONL 人工决定审计的离线测试。"""

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
    """验证 JSONL 追加、方案指纹稳定性和写入失败不会被吞掉。"""
    @classmethod
    def setUpClass(cls) -> None:
        """使用离线 Agent 生成可供审计的真实结构化方案。"""
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-audit-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run("请为客户 7590-VHVEG 生成完整挽留方案。")
        assert result.plan is not None
        cls.plan = result.plan

    def test_confirmed_decision_writes_required_fields(self) -> None:
        """确认后验证 JSONL 包含客户、状态、时间、来源摘要和唯一 ID。"""
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
        """验证确认和拒绝各追加一行，而不是覆盖已有审计记录。"""
        with tempfile.TemporaryDirectory(prefix="agent-audit-test-") as temp:
            path = Path(temp) / "decisions.jsonl"
            first = append_decision_record(path, self.plan, "confirmed")
            second = append_decision_record(path, self.plan, "rejected")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotEqual(first.audit_id, second.audit_id)
            self.assertEqual(json.loads(lines[1])["result_status"], "rejected")

    def test_fingerprint_is_stable_for_same_plan(self) -> None:
        """同一方案多次计算应得到相同的 64 位 SHA-256 指纹。"""
        first = calculate_plan_fingerprint(self.plan)
        second = calculate_plan_fingerprint(self.plan)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_write_error_is_not_silently_ignored(self) -> None:
        """把目录误当文件时应抛系统写入异常，不能伪装成审计成功。"""
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
