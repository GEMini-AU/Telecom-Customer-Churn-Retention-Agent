# ============================================================================
# 文件职责：回归测试新增客户、训练/测试来源标记和统一页面入口。
# 主要调用方：``python -m unittest discover -s tests``。
# 输入/输出：输入临时 SQLite 和真实公开 CSV；输出离线断言结果。
# 不负责：不调用在线 DeepSeek，也不向项目正式客户数据库写测试记录。
# ============================================================================
"""四项展示改进中的核心边界测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from streamlit.testing.v1 import AppTest

from agent.registry import BusinessToolRegistry
from agent.single_agent import SingleRetentionAgent
from business_tools.customer_lookup import CustomerLookupTool
from business_tools.customer_store import CustomCustomerStore
from business_tools.data_provenance import customer_source, split_customer_ids
from scripts.test_agent import tools_follow_dependencies
from business_tools.schemas import CustomerLookupInput, CustomerProfile
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "telco_customer_churn.csv"


class PortfolioUpgradeTests(unittest.TestCase):
    """验证新增客户不污染原 CSV，且测试集标记和双页面入口可信。"""

    def test_custom_customer_persists_and_is_queryable(self) -> None:
        """新客户经独立 SQLite 保存后，新的查询工具实例也能读取。"""
        with tempfile.TemporaryDirectory() as directory:
            store = CustomCustomerStore(Path(directory) / "customers.sqlite3")
            original = CustomerLookupTool(DATA).run(
                CustomerLookupInput(customer_id="7590-VHVEG")
            )
            assert original.customer is not None
            profile = CustomerProfile(
                customer_id="9000-NEWAA", features=original.customer.features
            )
            store.add(profile)
            fresh_lookup = CustomerLookupTool(DATA, custom_store=store)
            result = fresh_lookup.run(CustomerLookupInput(customer_id="9000-NEWAA"))
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.customer, profile)
            self.assertEqual(store.list_ids(), ["9000-NEWAA"])
            # 同一个 ID 不应被静默覆盖，避免审计引用到不同客户内容。
            with self.assertRaises(sqlite3.IntegrityError):
                store.add(profile)

            # 用新客户跑完整的四工具 Agent；离线替身只代替模型选工具与成文。
            registry = BusinessToolRegistry(
                ROOT, customer_tool=CustomerLookupTool(DATA, custom_store=store)
            )
            agent = SingleRetentionAgent(
                ROOT, registry=registry,
                settings=DeepSeekSettings(api_key="offline-custom-test"),
                openai_client=build_fake_client(OfflineAgentCompletions()),
            )
            plan_result = agent.run("请为客户 9000-NEWAA 生成完整挽留方案。")
            self.assertTrue(plan_result.success, plan_result.error)
            self.assertEqual(
                [call.tool_name for call in plan_result.tool_calls],
                ["get_customer_profile", "predict_churn_risk", "search_retention_policy", "calculate_retention_offer"],
            )

    def test_provenance_separates_train_test_and_manual(self) -> None:
        """复现 Notebook 划分，并确认新增客户不冒充独立测试集。"""
        train, test = split_customer_ids(DATA)
        self.assertFalse(train & test)
        self.assertEqual(len(train) + len(test), 7043)
        self.assertIn("7590-VHVEG", train)
        self.assertIn("2754-SDJRD", test)
        self.assertEqual(customer_source("9000-NEWAA", DATA), "自定义客户或未知 ID")

    def test_unified_page_launches_without_exception(self) -> None:
        """统一入口的默认页和 Agent 页都能加载，旧脚本也仍独立存在。"""
        app = AppTest.from_file(str(ROOT / "main_app.py")).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        app.switch_page("agent_app.py").run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Agent" in item.value for item in app.title))
        self.assertTrue((ROOT / "app.py").exists())
        self.assertTrue((ROOT / "agent_app.py").exists())

    def test_provenance_cache_invalidates_when_csv_changes(self) -> None:
        """同一文件重复查询只读一次，文件内容变化后必须重新读取。"""
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "small.csv"
            rows = [
                {"customerID": f"ID-{index}", "Churn": "Yes" if index % 2 else "No"}
                for index in range(10)
            ]
            pd.DataFrame(rows).to_csv(data_path, index=False)
            with patch("business_tools.data_provenance.pd.read_csv", wraps=pd.read_csv) as read:
                split_customer_ids(data_path)
                split_customer_ids(data_path)
                self.assertEqual(read.call_count, 1)
                rows.append({"customerID": "ID-10", "Churn": "No"})
                pd.DataFrame(rows).to_csv(data_path, index=False)
                split_customer_ids(data_path)
                self.assertEqual(read.call_count, 2)

    def test_agent_tool_order_checks_only_true_dependencies(self) -> None:
        """政策先检索再预测应合法；预测先于客户或优惠先于预测应拒绝。"""
        required = [
            "get_customer_profile", "predict_churn_risk",
            "search_retention_policy", "calculate_retention_offer",
        ]
        self.assertTrue(tools_follow_dependencies(required, required))
        self.assertTrue(tools_follow_dependencies([
            "get_customer_profile", "search_retention_policy",
            "predict_churn_risk", "calculate_retention_offer",
        ], required))
        self.assertFalse(tools_follow_dependencies([
            "predict_churn_risk", "get_customer_profile",
            "search_retention_policy", "calculate_retention_offer",
        ], required))

    def test_custom_customer_form_renders_all_model_fields(self) -> None:
        """页面切到新客户模式时能渲染输入表单，不需要真实密钥。"""
        app = AppTest.from_file(str(ROOT / "agent_app.py")).run(timeout=20)
        app.radio[0].set_value("添加自定义客户").run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(button.label == "保存演示客户" for button in app.button))
        # 19 个模型字段分别由 15 个选择项和 4 个数值项收集。
        self.assertGreaterEqual(len(app.selectbox), 15)
        self.assertGreaterEqual(len(app.number_input), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
