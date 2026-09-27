# ============================================================================
# 文件职责：新版 Streamlit 页面入口，负责收集用户输入、展示 Agent 方案并触发人工确认。
# 主要调用方：命令 ``streamlit run agent_app.py``。
# 输入/输出：输入客户 ID、自然语言任务；输出页面内容和 SQLite 审计状态。
# 不负责：不直接查询模型、检索政策或计算优惠，实际业务由 SingleRetentionAgent 调用工具完成。
# ============================================================================
"""独立的 Streamlit 单 Agent 页面。

运行入口：``streamlit run agent_app.py``。页面选择客户、收集自然语言任务、调用
``SingleRetentionAgent``，展示工具证据和方案；只有人工按钮才会改变 SQLite 审计状态。
"""

from __future__ import annotations

from pathlib import Path
import re
import sqlite3
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
from business_tools.customer_store import CustomCustomerStore
from business_tools.data_provenance import customer_source
from business_tools.schemas import ChurnFeatures, CustomerProfile


# 所有运行时路径都从当前文件推导，避免依赖开发电脑上的绝对路径。
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "telco_customer_churn.csv"
AUDIT_DB_PATH = PROJECT_ROOT / "storage" / "audit" / "agent_decisions.sqlite3"
CUSTOMER_DB_PATH = PROJECT_ROOT / "storage" / "customers" / "demo_customers.sqlite3"


st.set_page_config(
    # 必须在其他 Streamlit 输出之前调用，用于设置浏览器页签与页面布局。
    page_title="客户流失预测与智能运营 Agent",
    page_icon="🧭",
    layout="wide",
)


# 客户下拉列表只依赖 CSV；缓存可避免 Streamlit 每次重跑都重复读文件。
@st.cache_data(show_spinner=False)
def load_customer_ids(data_path: Path) -> list[str]:
    """只读取 CSV 的 ``customerID`` 列，生成下拉选择项并由 Streamlit 缓存。"""
    data = pd.read_csv(data_path, usecols=["customerID"])
    return sorted(data["customerID"].astype(str).str.strip().unique().tolist())


# Agent 内部持有模型客户端和工具实例，属于“资源”而非普通数据。
@st.cache_resource(show_spinner=False)
def build_agent(max_tool_calls: int) -> SingleRetentionAgent:
    """按工具调用上限创建并缓存 Agent 资源，避免每次组件交互重复初始化。"""
    return SingleRetentionAgent(
        PROJECT_ROOT,
        max_tool_calls=max_tool_calls,
    )


def initialize_state() -> None:
    """初始化当前浏览器会话的 Agent 结果和人工决定状态。"""
    # Streamlit 每次交互都会重跑脚本；结果和人工确认状态需保存在 session_state。
    st.session_state.setdefault("agent_result", None)
    st.session_state.setdefault("decision_state", None)


def render_page() -> None:
    """组织完整页面：参数、客户、任务输入、结果及审计历史。"""
    initialize_state()
    st.title("客户流失预测与智能运营 Agent")
    st.caption(
        "单 Agent 演示页面。客户、模型、政策和优惠均来自已注册工具；"
        "方案必须经过人工确认，不会自动下发到真实运营商系统。"
    )

    # 侧栏只放运行参数，不在页面中暴露任何 API Key。
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
    if customer_id:
        try:
            source = customer_source(customer_id, DATA_PATH)
            st.caption(f"当前客户来源：{source}。训练集样例只用于流程演示，不能作为泛化能力证据。")
        except Exception as error:
            st.warning(f"客户来源暂无法核对：{type(error).__name__}：{error}")
    natural_task = st.text_area(
        "自然语言任务",
        placeholder=(
            "例如：分析该客户的流失风险，检索适用的演示政策，"
            "计算允许的演示优惠并生成完整挽留方案。"
        ),
        height=120,
    )

    # 只有用户主动点击时才启动 Agent 和可能产生费用的大模型调用。
    if st.button("生成挽留方案", type="primary", use_container_width=True):
        run_agent(customer_id, natural_task, int(max_tool_calls))

    result = st.session_state.agent_result
    # 切换客户或新建客户后不展示上一位客户的旧方案，防止人工误确认。
    if (
        isinstance(result, AgentRunResult)
        and result.plan is not None
        and result.plan.customer_basic.customer_id != customer_id
    ):
        st.session_state.agent_result = None
        st.session_state.decision_state = None
        result = None
    if isinstance(result, AgentRunResult):
        render_result(result)
    render_audit_history()


def render_customer_selector() -> str:
    """返回用户选择或输入的规范化客户 ID；CSV 失败时回退到手动输入。"""
    mode = st.radio(
        "客户选择方式",
        ["从项目数据选择", "手动输入", "添加自定义客户"],
        horizontal=True,
    )
    # 手工输入统一去空格并转大写，以匹配数据集中规范化后的 customerID。
    if mode == "手动输入":
        return st.text_input(
            "客户ID",
            placeholder="例如：7590-VHVEG",
        ).strip().upper()

    if mode == "添加自定义客户":
        return render_custom_customer_form()

    # CSV 读取失败时仍允许手工输入，让页面保持可用。
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


def render_custom_customer_form() -> str:
    """收集模型的 19 个必要字段，验证后单独落库，不修改训练 CSV。"""
    st.info("只填写演示客户的模型字段；不要录入真实姓名、电话或其他个人信息。")
    store = CustomCustomerStore(CUSTOMER_DB_PATH)
    # 保存过的客户可重新选择；数据库查询失败只影响本表单，不让整页崩溃。
    try:
        saved_ids = store.list_ids()
    except Exception as error:
        st.warning(f"自定义客户列表读取失败：{type(error).__name__}：{error}")
        saved_ids = []
    if saved_ids:
        selected = st.selectbox("已保存的演示客户", ["新建客户", *saved_ids])
        if selected != "新建客户":
            st.session_state["custom_customer_id"] = selected
    # 提交之前 Streamlit 不会执行保存；保存之后该 ID 可以走原来的四工具流程。
    with st.form("custom_customer_form"):
        customer_id = st.text_input("新客户ID", placeholder="例如：9000-NEWAA")
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("性别", ["Female", "Male"])
            senior = st.selectbox("是否老年客户", [0, 1])
            partner = st.selectbox("是否有伴侣", ["No", "Yes"])
            dependents = st.selectbox("是否有家属", ["No", "Yes"])
            tenure = st.number_input("在网月数", min_value=0, max_value=120, value=12)
            phone = st.selectbox("电话服务", ["Yes", "No"])
            lines = st.selectbox("多线服务", ["No", "Yes", "No phone service"])
            internet = st.selectbox("网络服务", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("合同类型", ["Month-to-month", "One year", "Two year"])
        with right:
            security = st.selectbox("在线安全", ["No", "Yes", "No internet service"])
            backup = st.selectbox("在线备份", ["No", "Yes", "No internet service"])
            protection = st.selectbox("设备保护", ["No", "Yes", "No internet service"])
            support = st.selectbox("技术支持", ["No", "Yes", "No internet service"])
            tv = st.selectbox("流媒体电视", ["No", "Yes", "No internet service"])
            movies = st.selectbox("流媒体电影", ["No", "Yes", "No internet service"])
            paperless = st.selectbox("电子账单", ["Yes", "No"])
            payment = st.selectbox("付款方式", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("月消费", min_value=0.0, max_value=10000.0, value=65.0, step=0.01)
            total = st.number_input("累计消费", min_value=0.0, max_value=1000000.0, value=780.0, step=0.01)
        submitted = st.form_submit_button("保存演示客户")

    if submitted:
        normalized_id = customer_id.strip().upper()
        if not re.fullmatch(r"[0-9]{4}-[A-Z]{5}", normalized_id):
            st.error("客户ID须为四位数字、连字符、五位大写字母，例如 9000-NEWAA。")
        else:
            try:
                # 关联字段必须互相一致，避免把不存在的服务组合送入模型。
                if (phone == "No") != (lines == "No phone service"):
                    raise ValueError("电话服务与多线服务选项不一致。")
                internet_fields = [security, backup, protection, support, tv, movies]
                if any((value == "No internet service") != (internet == "No") for value in internet_fields):
                    raise ValueError("网络服务与其附属服务选项不一致。")
                if normalized_id in load_customer_ids(DATA_PATH):
                    raise ValueError("该 ID 已存在于公开 CSV，不能覆盖原始数据。")
                features = ChurnFeatures(
                    SeniorCitizen=senior, tenure=tenure, MonthlyCharges=monthly,
                    TotalCharges=total, gender=gender, Partner=partner,
                    Dependents=dependents, PhoneService=phone, MultipleLines=lines,
                    InternetService=internet, OnlineSecurity=security,
                    OnlineBackup=backup, DeviceProtection=protection,
                    TechSupport=support, StreamingTV=tv, StreamingMovies=movies,
                    Contract=contract, PaperlessBilling=paperless,
                    PaymentMethod=payment,
                )
                store.add(CustomerProfile(customer_id=normalized_id, features=features))
                st.session_state["custom_customer_id"] = normalized_id
                st.session_state.agent_result = None
                st.session_state.decision_state = None
                st.success("演示客户已保存。现在可点击“生成挽留方案”。")
            except (ValueError, sqlite3.IntegrityError) as error:
                st.error(f"无法保存客户：{error}")
            except Exception as error:
                st.error(f"客户保存失败：{type(error).__name__}：{error}")
    selected_id = st.session_state.get("custom_customer_id", "")
    if selected_id:
        st.caption(f"当前已保存的自定义客户：{selected_id}")
    return selected_id


def run_agent(
    customer_id: str,
    natural_task: str,
    max_tool_calls: int,
) -> None:
    """把页面输入限定为一位客户的任务，运行 Agent 并为成功方案创建待确认审计。"""
    if not customer_id:
        st.error("请先输入或选择客户ID。")
        return
    # 将页面选择的客户 ID 写入任务，限制 Agent 只处理这一位客户。
    # 用户没填描述时提供默认业务目标，客户 ID 仍由页面选择器强制限定。
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

    # 先保存本次结果；只有成功方案才能进入待人工确认状态。
    st.session_state.agent_result = result
    if result.success and result.plan is not None:
        # 审计记录先以“待确认”状态写入，确认按钮不会直接执行外部业务动作。
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
    """根据 Agent 成功或失败结果选择展示方案，或展示错误及工具记录。"""
    if not result.success or result.plan is None:
        message = result.error.message if result.error else "Agent 未生成有效方案。"
        st.error(message)
        render_tool_records(result.tool_calls)
        return

    # 成功时按“状态 -> 方案 -> 工具轨迹 -> 人工确认”顺序展示，便于人工复核。
    plan = result.plan
    render_decision_status(plan)
    render_plan(plan)
    render_tool_records(result.tool_calls)
    render_decision_buttons(plan)


def render_decision_status(plan: RetentionPlan) -> None:
    """按当前方案指纹显示待确认、已确认执行或已拒绝状态。"""
    decision = st.session_state.decision_state
    # 指纹绑定当前方案内容，防止把旧审计状态误用于新生成的方案。
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
    """将已验证方案分区展示：客户、风险、RAG 依据、优惠、话术和行动。"""
    st.subheader("客户基本情况与风险判断")
    # 三列仅是页面布局，不改变预测工具中的概率、风险等级或优惠数据。
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

    # 政策依据逐条保留文件、章节和 chunk_id，保证页面可追溯到知识库原文。
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
    """在可展开区域展示实际工具调用轨迹，而非展示模型内部推理。"""
    with st.expander("工具调用记录", expanded=False):
        if not records:
            st.info("本次没有执行工具。")
            return
        # 只展示工具名、参数、成功状态和错误码，不展示模型内部推理文本。
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
    """读取最近 SQLite 审计摘要，读取失败只提示，不让页面崩溃。"""
    with st.expander("本地审计历史", expanded=False):
        try:
            records = list_decision_records(AUDIT_DB_PATH, limit=10)
        except Exception as error:
            st.warning(f"审计历史读取失败：{type(error).__name__}：{error}")
            return
        if not records:
            st.info("暂无本地审计记录。")
            return
        # 审计历史只展示方案摘要，不把完整客户特征或完整大模型响应写回页面。
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
    """仅当方案处于待确认状态时启用确认与拒绝按钮。"""
    st.subheader("人工确认")
    decision = st.session_state.decision_state
    # 只有数据库记录仍为待确认时，两个按钮才可点击。
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
    """把一次人工按钮点击交给 SQLite 原子落库，再刷新页面显示新状态。"""
    state = st.session_state.decision_state
    if not isinstance(state, dict) or not state.get("audit_id"):
        st.error("缺少待确认的审计记录，请重新生成方案。")
        return
    # SQLite 层会再次核对方案指纹和当前状态，避免重复点击或方案变更后误写入。
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
    """页面脚本入口；捕获未预期错误并在 UI 中展示。"""
    # 页面兜底保护：单个意外错误显示在页面上，不让整个 Streamlit 应用直接崩溃。
    try:
        render_page()
    except Exception as error:
        st.error(
            "页面发生未预期错误，但应用仍可继续运行："
            f"{type(error).__name__}：{error}"
        )


if __name__ == "__main__":
    main()
