"""Independent Streamlit page for the single Agent and human approval."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import streamlit as st

from agent.audit import calculate_plan_fingerprint
from agent.schemas import AgentRunResult, RetentionPlan, ToolCallRecord
from agent.single_agent import SingleRetentionAgent
from agent.sqlite_audit import (
    create_pending_decision,
    decide_pending_decision,
    list_decision_records,
)
from llm.deepseek_client import DeepSeekConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "telco_customer_churn.csv"
AUDIT_DB_PATH = PROJECT_ROOT / "storage" / "audit" / "agent_decisions.sqlite3"


st.set_page_config(
    page_title="客户流失预测与智能运营 Agent",
    page_icon="🧭",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_customer_ids(data_path: Path) -> list[str]:
    data = pd.read_csv(data_path, usecols=["customerID"])
    return sorted(data["customerID"].astype(str).str.strip().unique().tolist())


@st.cache_resource(show_spinner=False)
def build_agent(max_tool_calls: int) -> SingleRetentionAgent:
    return SingleRetentionAgent(
        PROJECT_ROOT,
        max_tool_calls=max_tool_calls,
    )


def initialize_state() -> None:
    st.session_state.setdefault("agent_result", None)
    st.session_state.setdefault("decision_state", None)


def render_page() -> None:
    initialize_state()
    st.title("客户流失预测与智能运营 Agent")
    st.caption(
        "单 Agent 演示页面。客户、模型、政策和优惠均来自已注册工具；"
        "方案必须经过人工确认，不会自动下发到真实运营商系统。"
    )

    with st.sidebar:
        st.subheader("运行设置")
        max_tool_calls = st.number_input(
            "最大工具调用次数",
            min_value=4,
            max_value=20,
            value=8,
            step=1,
            help="达到上限后 Agent 会停止，防止无限循环。",
        )
        st.info("API Key 仅从环境变量 DEEPSEEK_API_KEY 读取。")
        st.caption(f"本地SQLite审计：{AUDIT_DB_PATH}")

    customer_id = render_customer_selector()
    natural_task = st.text_area(
        "自然语言任务",
        placeholder=(
            "例如：分析该客户的流失风险，检索适用的演示政策，"
            "计算允许的演示优惠并生成完整挽留方案。"
        ),
        height=120,
    )

    if st.button("生成挽留方案", type="primary", use_container_width=True):
        run_agent(customer_id, natural_task, int(max_tool_calls))

    result = st.session_state.agent_result
    if isinstance(result, AgentRunResult):
        render_result(result)
    render_audit_history()


def render_customer_selector() -> str:
    mode = st.radio(
        "客户选择方式",
        ["从项目数据选择", "手动输入"],
        horizontal=True,
    )
    if mode == "手动输入":
        return st.text_input(
            "客户ID",
            placeholder="例如：7590-VHVEG",
        ).strip().upper()

    try:
        customer_ids = load_customer_ids(DATA_PATH)
    except Exception as error:
        st.warning(
            f"客户列表读取失败，请改用手动输入：{type(error).__name__}：{error}"
        )
        return st.text_input("客户ID").strip().upper()
    return st.selectbox(
        "客户ID",
        customer_ids,
        index=customer_ids.index("7590-VHVEG")
        if "7590-VHVEG" in customer_ids
        else 0,
    )


def run_agent(
    customer_id: str,
    natural_task: str,
    max_tool_calls: int,
) -> None:
    if not customer_id:
        st.error("请先输入或选择客户ID。")
        return
    task_body = natural_task.strip() or "生成完整的客户流失分析与挽留方案。"
    scoped_task = f"请只处理客户 {customer_id}。用户任务：{task_body}"

    try:
        agent = build_agent(max_tool_calls)
        with st.spinner("Agent 正在查询客户、预测风险并检索演示政策……"):
            result = agent.run(scoped_task)
    except DeepSeekConfigurationError as error:
        st.error(f"Agent 配置失败：{error}")
        return
    except Exception as error:
        st.error(f"Agent 页面调用失败：{type(error).__name__}：{error}")
        return

    st.session_state.agent_result = result
    if result.success and result.plan is not None:
        try:
            record = create_pending_decision(AUDIT_DB_PATH, result.plan)
        except Exception as error:
            st.session_state.decision_state = None
            st.error(
                "方案已生成，但审计记录创建失败，不能确认执行："
                f"{type(error).__name__}：{error}"
            )
            return
        st.session_state.decision_state = {
            "plan_fingerprint": record.plan_fingerprint,
            "status": record.status,
            "audit_id": record.audit_id,
        }
    else:
        st.session_state.decision_state = None


def render_result(result: AgentRunResult) -> None:
    if not result.success or result.plan is None:
        message = result.error.message if result.error else "Agent 未生成有效方案。"
        st.error(message)
        render_tool_records(result.tool_calls)
        return

    plan = result.plan
    render_decision_status(plan)
    render_plan(plan)
    render_tool_records(result.tool_calls)
    render_decision_buttons(plan)


def render_decision_status(plan: RetentionPlan) -> None:
    decision = st.session_state.decision_state
    expected_fingerprint = calculate_plan_fingerprint(plan)
    if not isinstance(decision, dict) or decision.get(
        "plan_fingerprint"
    ) != expected_fingerprint:
        st.warning("当前方案尚未建立人工确认状态，请重新生成方案。")
        return

    status = decision.get("status")
    if status == "pending_confirmation":
        st.warning("当前状态：待人工确认。方案尚未标记为已执行。")
    elif status == "confirmed_execution":
        st.success(
            "当前状态：已确认执行（项目演示记录，不会向真实运营商系统下发）。"
        )
    elif status == "rejected":
        st.error("当前状态：方案已被人工拒绝，不得执行。")


def render_plan(plan: RetentionPlan) -> None:
    st.subheader("客户基本情况与风险判断")
    first, second, third = st.columns(3)
    first.metric("客户ID", plan.customer_basic.customer_id)
    second.metric(
        "流失概率",
        f"{plan.risk_assessment.churn_probability:.2%}",
    )
    third.metric("风险等级", plan.risk_assessment.risk_level)
    st.write(
        {
            "合同类型": plan.customer_basic.contract,
            "使用时长（月）": plan.customer_basic.tenure_months,
            "月消费": plan.customer_basic.monthly_charges,
            "互联网服务": plan.customer_basic.internet_service,
            "技术支持": plan.customer_basic.tech_support,
            "付款方式": plan.customer_basic.payment_method,
        }
    )

    st.subheader("主要风险因素")
    if plan.risk_assessment.risk_factors:
        for item in plan.risk_assessment.risk_factors:
            st.write(f"- {item}")
    else:
        st.info("未命中当前项目定义的五条演示风险信息规则。")

    st.subheader("RAG政策依据")
    for index, citation in enumerate(plan.policy_evidence, start=1):
        title = (
            f"来源 {index}：{citation.document_title} / "
            f"{citation.section_title}"
        )
        with st.expander(title, expanded=index == 1):
            st.caption(
                f"文件：{citation.source_file} ｜ chunk_id：{citation.chunk_id}"
            )
            st.write(citation.excerpt)

    st.subheader("推荐优惠（项目演示）")
    offer = plan.recommended_offer
    offer_left, offer_middle, offer_right = st.columns(3)
    offer_left.metric("方案", offer.offer_name)
    offer_middle.metric("每月优惠", str(offer.monthly_discount))
    offer_right.metric("合计优惠", str(offer.total_discount))
    st.write(
        f"优惠比例：{offer.discount_rate}；期限：{offer.duration_months}个月；"
        f"不可叠加：{'是' if not offer.stackable else '否'}；"
        f"需要人工审批：{'是' if offer.requires_manual_approval else '否'}。"
    )
    st.caption(offer.currency_note)
    st.warning(offer.disclaimer)

    st.subheader("客服沟通话术")
    st.info(plan.communication_script)

    st.subheader("下一步行动")
    for item in plan.next_actions:
        st.write(f"- {item}")
    st.caption(plan.disclaimer)


def render_tool_records(records: list[ToolCallRecord]) -> None:
    with st.expander("工具调用记录", expanded=False):
        if not records:
            st.info("本次没有执行工具。")
            return
        rows: list[dict[str, Any]] = []
        for item in records:
            rows.append(
                {
                    "序号": item.call_index,
                    "工具": item.tool_name,
                    "参数": item.arguments,
                    "成功": item.success,
                    "错误码": item.error_code or "",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_audit_history() -> None:
    with st.expander("本地审计历史", expanded=False):
        try:
            records = list_decision_records(AUDIT_DB_PATH, limit=10)
        except Exception as error:
            st.warning(f"审计历史读取失败：{type(error).__name__}：{error}")
            return
        if not records:
            st.info("暂无本地审计记录。")
            return
        rows = [
            {
                "创建时间(UTC)": item.created_at_utc.isoformat(),
                "决定时间(UTC)": (
                    item.decided_at_utc.isoformat() if item.decided_at_utc else ""
                ),
                "客户ID": item.customer_id,
                "状态": item.status,
                "优惠规则": item.plan_summary.offer_code,
                "审计ID": item.audit_id,
            }
            for item in records
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_decision_buttons(plan: RetentionPlan) -> None:
    st.subheader("人工确认")
    decision = st.session_state.decision_state
    pending = isinstance(decision, dict) and decision.get(
        "status"
    ) == "pending_confirmation"
    confirm_column, reject_column = st.columns(2)
    confirm_clicked = confirm_column.button(
        "确认执行",
        type="primary",
        disabled=not pending,
        use_container_width=True,
    )
    reject_clicked = reject_column.button(
        "拒绝方案",
        disabled=not pending,
        use_container_width=True,
    )

    if confirm_clicked:
        save_decision(plan, "confirmed")
    elif reject_clicked:
        save_decision(plan, "rejected")


def save_decision(
    plan: RetentionPlan,
    decision: Literal["confirmed", "rejected"],
) -> None:
    state = st.session_state.decision_state
    if not isinstance(state, dict) or not state.get("audit_id"):
        st.error("缺少待确认的审计记录，请重新生成方案。")
        return
    try:
        record = decide_pending_decision(
            AUDIT_DB_PATH,
            state["audit_id"],
            plan,
            decision=decision,
        )
    except Exception as error:
        st.error(f"审计日志写入失败，状态未改变：{type(error).__name__}：{error}")
        return

    st.session_state.decision_state = {
        "plan_fingerprint": record.plan_fingerprint,
        "status": record.status,
        "audit_id": record.audit_id,
    }
    st.rerun()


def main() -> None:
    try:
        render_page()
    except Exception as error:
        st.error(
            "页面发生未预期错误，但应用仍可继续运行："
            f"{type(error).__name__}：{error}"
        )


if __name__ == "__main__":
    main()
