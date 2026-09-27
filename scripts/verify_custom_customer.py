# ============================================================================
# 文件职责：用纯虚拟新客户验证“保存→四工具Agent→拒绝审计”的完整链路。
# 主要调用方：用户手工运行 --offline 或 --live；在线模式调用真实 DeepSeek。
# 输入/输出：输入命令行模式和进程环境变量；输出不含密钥的验收摘要。
# 不负责：不改训练 CSV/模型/知识库，不执行真实优惠或客户触达。
# ============================================================================
"""虚拟新客户端到端验收。默认离线；--live 才会访问模型服务并可能计费。"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.registry import BusinessToolRegistry
from agent.single_agent import SingleRetentionAgent
from agent.sqlite_audit import (
    create_pending_decision,
    decide_pending_decision,
    list_decision_records,
)
from business_tools.customer_lookup import CustomerLookupTool
from business_tools.customer_store import CustomCustomerStore
from business_tools.schemas import ChurnFeatures, CustomerProfile
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


VIRTUAL_ID = "9000-NEWAA"
EXPECTED_TOOLS = {
    "get_customer_profile", "predict_churn_risk",
    "search_retention_policy", "calculate_retention_offer",
}


def virtual_profile() -> CustomerProfile:
    """构造不含姓名和联系方式、也不复制原 CSV 行的虚拟客户。"""
    features = ChurnFeatures(
        SeniorCitizen=0, tenure=4, MonthlyCharges=85.0, TotalCharges=340.0,
        gender="Female", Partner="No", Dependents="No", PhoneService="Yes",
        MultipleLines="No", InternetService="Fiber optic", OnlineSecurity="No",
        OnlineBackup="No", DeviceProtection="No", TechSupport="No",
        StreamingTV="Yes", StreamingMovies="No", Contract="Month-to-month",
        PaperlessBilling="Yes", PaymentMethod="Electronic check",
    )
    return CustomerProfile(customer_id=VIRTUAL_ID, features=features)


def main() -> None:
    """分别验证客户、四工具结果、政策来源和拒绝审计的关键不变量。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="真实在线调用，可能产生费用")
    args = parser.parse_args()

    # 在线模式先检查配置，避免创建临时数据后才发现没有密钥；不打印密钥内容。
    settings = DeepSeekSettings.from_env() if args.live else DeepSeekSettings(
        api_key="offline-custom-customer"
    )
    with tempfile.TemporaryDirectory(prefix="churn-new-customer-") as folder:
        temp_root = Path(folder)
        store = CustomCustomerStore(temp_root / "customers.sqlite3")
        store.add(virtual_profile())
        registry = BusinessToolRegistry(
            ROOT,
            customer_tool=CustomerLookupTool(
                ROOT / "telco_customer_churn.csv", custom_store=store
            ),
        )
        kwargs = {} if args.live else {
            "openai_client": build_fake_client(OfflineAgentCompletions())
        }
        agent = SingleRetentionAgent(ROOT, registry=registry, settings=settings, **kwargs)
        result = agent.run(f"请为客户 {VIRTUAL_ID} 查询资料，预测流失，检索演示政策，计算优惠并生成完整方案。")
        if not result.success or result.plan is None:
            code = result.error.code if result.error else "UNKNOWN"
            raise SystemExit(f"验收失败：{code}；已调用 {[x.tool_name for x in result.tool_calls]}")
        actual_tools = [item.tool_name for item in result.tool_calls]
        # RAG 与预测彼此独立，可先后互换；只强制真正的业务依赖顺序。
        if (
            len(actual_tools) != 4
            or set(actual_tools) != EXPECTED_TOOLS
            or actual_tools.index("get_customer_profile") >= actual_tools.index("predict_churn_risk")
            or actual_tools.index("predict_churn_risk") >= actual_tools.index("calculate_retention_offer")
        ):
            raise SystemExit(f"工具集合或依赖顺序不符：{actual_tools}")
        plan = result.plan
        if not plan.policy_evidence:
            raise SystemExit("验收失败：方案没有可追溯的政策依据。")
        # 人工拒绝写入临时审计库；这不是自动执行真实优惠。
        audit_path = temp_root / "audit.sqlite3"
        pending = create_pending_decision(audit_path, plan)
        if pending.status != "pending_confirmation":
            raise SystemExit("验收失败：方案未经人工决定就不处于待确认状态。")
        rejected = decide_pending_decision(
            audit_path, pending.audit_id, plan, decision="rejected"
        )
        if rejected.status != "rejected" or rejected.decided_at_utc is None:
            raise SystemExit("验收失败：拒绝结果未正确写入 SQLite。")
        # 重新打开数据库读取，不只检查更新函数的内存返回值。
        persisted = list_decision_records(audit_path, limit=1)
        if not persisted or persisted[0].status != "rejected":
            raise SystemExit("验收失败：重读数据库后未看到拒绝状态。")
        print(f"模式：{'在线' if args.live else '离线'}；客户：{VIRTUAL_ID}")
        print(f"工具：{' -> '.join(actual_tools)}")
        print(f"风险：{plan.risk_assessment.risk_level}；流失概率：{plan.risk_assessment.churn_probability:.4f}")
        print(f"政策来源：{[item.source_file for item in plan.policy_evidence]}")
        print(f"优惠代码：{plan.recommended_offer.offer_code}；合计：{plan.recommended_offer.total_discount}")
        print(f"人工结果：{rejected.status}；审计时间已写入：{rejected.decided_at_utc is not None}")


if __name__ == "__main__":
    main()
