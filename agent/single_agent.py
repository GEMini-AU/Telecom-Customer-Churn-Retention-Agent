"""A bounded single-agent loop using only the registered business tools."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from llm.deepseek_client import DeepSeekSettings

from .registry import (
    CALCULATE_RETENTION_OFFER,
    GET_CUSTOMER_PROFILE,
    PREDICT_CHURN_RISK,
    SEARCH_RETENTION_POLICY,
    AgentToolState,
    BusinessToolRegistry,
)
from .schemas import (
    AgentError,
    AgentRunResult,
    RetentionPlan,
    ToolCallRecord,
)


AGENT_DISCLAIMER = (
    "本方案由项目演示模型、演示政策和演示优惠规则生成，"
    "不是真实运营商政策或审批结果，执行前必须人工复核。"
)
CUSTOMER_ID_PATTERN = re.compile(r"(?<![A-Z0-9])[0-9]{4}-[A-Z]{5}(?![A-Z0-9])")
FORBIDDEN_SCRIPT_PATTERN = re.compile(r"[0-9０-９%％￥¥$]")
FORBIDDEN_SCRIPT_CLAIMS = ("模型认定", "一定会", "保证挽留", "永久有效")


class SingleRetentionAgent:
    """One DeepSeek agent with an allow-list and a hard tool-call budget."""

    def __init__(
        self,
        project_root: Path,
        registry: BusinessToolRegistry | None = None,
        settings: DeepSeekSettings | None = None,
        openai_client: Any | None = None,
        max_tool_calls: int = 8,
    ) -> None:
        if not 1 <= max_tool_calls <= 20:
            raise ValueError("max_tool_calls 必须位于 1 到 20 之间。")
        self.project_root = project_root.resolve()
        self.registry = registry or BusinessToolRegistry(self.project_root)
        self.settings = settings or DeepSeekSettings.from_env()
        self._client = openai_client or OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            max_retries=1,
        )
        self.max_tool_calls = max_tool_calls

    def run(self, task: str) -> AgentRunResult:
        clean_task = task.strip()
        if not clean_task:
            return _failure("TASK_EMPTY", "自然语言任务不能为空。")

        customer_ids = list(dict.fromkeys(CUSTOMER_ID_PATTERN.findall(clean_task)))
        if not customer_ids:
            return _failure(
                "CUSTOMER_ID_REQUIRED",
                "任务中缺少 customerID，请提供例如 7590-VHVEG 的客户编号。",
            )
        if len(customer_ids) > 1:
            return _failure(
                "MULTIPLE_CUSTOMERS_NOT_SUPPORTED",
                "第一版单 Agent 每次只处理一位客户，请只提供一个 customerID。",
            )
        requested_customer_id = customer_ids[0]

        state = self.registry.create_state()
        records: list[ToolCallRecord] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": clean_task},
        ]

        for _ in range(self.max_tool_calls + 2):
            try:
                response = self._client.chat.completions.create(
                    model=self.settings.model,
                    messages=messages,
                    tools=self.registry.tool_definitions(),
                    tool_choice="auto",
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
            except Exception as error:
                return _failure(
                    "MODEL_CALL_ERROR",
                    f"大模型调用失败：{type(error).__name__}：{error}",
                    records,
                )

            if not response.choices:
                return _failure(
                    "MODEL_RESPONSE_EMPTY",
                    "大模型没有返回候选结果。",
                    records,
                )
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if tool_calls:
                messages.append(_assistant_tool_message(message, tool_calls))
                for tool_call in tool_calls:
                    if len(records) >= self.max_tool_calls:
                        return _failure(
                            "MAX_TOOL_CALLS_EXCEEDED",
                            f"工具调用达到上限 {self.max_tool_calls} 次，已停止执行。",
                            records,
                        )
                    tool_name = tool_call.function.name
                    arguments, argument_error = _parse_arguments(
                        tool_call.function.arguments
                    )
                    if argument_error is not None:
                        records.append(
                            ToolCallRecord(
                                call_index=len(records) + 1,
                                tool_name=tool_name,
                                arguments={},
                                success=False,
                                error_code="TOOL_ARGUMENTS_INVALID",
                            )
                        )
                        return _failure(
                            "TOOL_ARGUMENTS_INVALID",
                            argument_error,
                            records,
                        )
                    if tool_name not in self.registry.allowed_tool_names:
                        records.append(
                            ToolCallRecord(
                                call_index=len(records) + 1,
                                tool_name=tool_name,
                                arguments=arguments,
                                success=False,
                                error_code="UNREGISTERED_TOOL",
                            )
                        )
                        return _failure(
                            "UNREGISTERED_TOOL",
                            f"模型请求了未注册工具 {tool_name}，已拒绝执行。",
                            records,
                        )
                    if tool_name != SEARCH_RETENTION_POLICY:
                        if arguments.get("customer_id") != requested_customer_id:
                            records.append(
                                ToolCallRecord(
                                    call_index=len(records) + 1,
                                    tool_name=tool_name,
                                    arguments=arguments,
                                    success=False,
                                    error_code="CUSTOMER_SCOPE_MISMATCH",
                                )
                            )
                            return _failure(
                                "CUSTOMER_SCOPE_MISMATCH",
                                "工具参数中的 customerID 与用户任务不一致。",
                                records,
                            )

                    execution = self.registry.execute(
                        tool_name,
                        arguments,
                        state,
                    )
                    error_code = (
                        execution.error.code if execution.error is not None else None
                    )
                    records.append(
                        ToolCallRecord(
                            call_index=len(records) + 1,
                            tool_name=tool_name,
                            arguments=arguments,
                            success=execution.success,
                            error_code=error_code,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": execution.model_dump_json(),
                        }
                    )
                    if not execution.success:
                        error_message = (
                            execution.error.message
                            if execution.error is not None
                            else "工具返回失败但没有错误详情。"
                        )
                        return _failure(
                            "TOOL_EXECUTION_FAILED",
                            f"{tool_name} 执行失败：{error_message}",
                            records,
                        )
                continue

            content = getattr(message, "content", None)
            if not content or not content.strip():
                return _failure(
                    "MODEL_RESPONSE_EMPTY",
                    "大模型既没有调用工具，也没有返回最终方案。",
                    records,
                )
            try:
                plan = RetentionPlan.model_validate_json(content)
                _verify_plan(plan, state, requested_customer_id)
            except (ValidationError, ValueError) as error:
                return _failure(
                    "FINAL_OUTPUT_INVALID",
                    f"最终方案未通过工具依据校验：{error}",
                    records,
                )
            return AgentRunResult(
                success=True,
                plan=plan,
                tool_calls=records,
            )

        return _failure(
            "AGENT_LOOP_LIMIT_EXCEEDED",
            "Agent 对话轮次超过安全上限，已停止执行。",
            records,
        )

    def _system_prompt(self) -> str:
        schema = json.dumps(
            RetentionPlan.model_json_schema(),
            ensure_ascii=False,
        )
        return (
            "你是客户流失预测与智能运营项目的单 Agent。"
            "你只能调用系统提供的四个工具，不得假设存在其他工具。"
            "对于完整挽留方案，必须依次完成客户查询、流失预测、政策检索和优惠计算。"
            "客户信息、概率、风险因素必须逐字复制对应工具结果；"
            "政策引用必须复制检索结果中的 chunk_id、来源、标题、章节和完整片段；"
            "优惠对象必须逐字段复制计算工具结果。"
            "不得自行计算或改写概率、金额、折扣、期限与政策来源。"
            "工具失败时不要继续，也不要生成替代事实。"
            "沟通话术和下一步行动不得出现任何数字、百分号、币种符号，也不得声称"
            "模型认定客户会流失、保证挽留成功或优惠永久有效。"
            "为避免误判，沟通话术和下一步行动中连否定形式也不要使用"
            "‘模型认定’、‘一定会’、‘保证挽留’、‘永久有效’这些原词；"
            "需要表达不确定性时，改写为‘效果以实际情况为准’、"
            "‘方案以审批结果为准’。"
            "最终只返回符合以下 JSON Schema 的 JSON，不要使用 Markdown："
            f"{schema}"
            f"disclaimer 必须严格等于：{AGENT_DISCLAIMER}"
        )


def _parse_arguments(raw_arguments: str) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as error:
        return {}, f"工具参数不是有效 JSON：{error.msg}。"
    if not isinstance(value, dict):
        return {}, "工具参数必须是 JSON 对象。"
    return value, None


def _assistant_tool_message(
    message: Any,
    tool_calls: list[Any],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": getattr(message, "content", None) or "",
        "tool_calls": [
            {
                "id": item.id,
                "type": "function",
                "function": {
                    "name": item.function.name,
                    "arguments": item.function.arguments,
                },
            }
            for item in tool_calls
        ],
    }


def _verify_plan(
    plan: RetentionPlan,
    state: AgentToolState,
    customer_id: str,
) -> None:
    customer = state.customers.get(customer_id)
    prediction = state.predictions.get(customer_id)
    offer = state.offers.get(customer_id)
    if customer is None:
        raise ValueError("缺少客户查询工具结果。")
    if prediction is None:
        raise ValueError("缺少流失预测工具结果。")
    if not state.evidence:
        raise ValueError("缺少有依据的政策检索结果。")
    if offer is None:
        raise ValueError("缺少优惠计算工具结果。")

    expected_customer = {
        "customer_id": customer.customer_id,
        "tenure_months": customer.features.tenure,
        "contract": customer.features.contract,
        "monthly_charges": customer.features.monthly_charges,
        "internet_service": customer.features.internet_service,
        "tech_support": customer.features.tech_support,
        "payment_method": customer.features.payment_method,
    }
    actual_customer = plan.customer_basic.model_dump()
    if actual_customer != expected_customer:
        raise ValueError("客户基本情况与客户查询工具结果不一致。")

    actual_risk = plan.risk_assessment
    if not math.isclose(
        actual_risk.churn_probability,
        prediction.churn_probability,
        abs_tol=1e-12,
    ):
        raise ValueError("流失概率与预测工具结果不一致。")
    if actual_risk.classification_threshold != prediction.classification_threshold:
        raise ValueError("分类阈值与预测工具结果不一致。")
    if actual_risk.prediction_label != prediction.prediction_label:
        raise ValueError("预测结果与预测工具结果不一致。")
    if actual_risk.risk_level != prediction.risk_level:
        raise ValueError("风险等级与预测工具结果不一致。")
    if actual_risk.risk_factors != prediction.risk_factors:
        raise ValueError("主要风险因素与预测工具结果不一致。")

    seen_chunk_ids: set[str] = set()
    for citation in plan.policy_evidence:
        evidence = state.evidence.get(citation.chunk_id)
        if evidence is None:
            raise ValueError(f"政策片段 {citation.chunk_id} 未由检索工具返回。")
        expected_citation = {
            "chunk_id": evidence.chunk_id,
            "source_file": evidence.source_file,
            "document_title": evidence.document_title,
            "section_title": evidence.section_title,
            "excerpt": evidence.text,
        }
        if citation.model_dump() != expected_citation:
            raise ValueError(f"政策片段 {citation.chunk_id} 的来源或正文被改写。")
        if citation.chunk_id in seen_chunk_ids:
            raise ValueError(f"政策片段 {citation.chunk_id} 被重复引用。")
        seen_chunk_ids.add(citation.chunk_id)

    if plan.recommended_offer != offer:
        raise ValueError("推荐优惠与优惠计算工具结果不一致。")
    if FORBIDDEN_SCRIPT_PATTERN.search(plan.communication_script):
        raise ValueError("客服沟通话术包含未经允许的数字或金额表达。")
    if any(
        claim in plan.communication_script for claim in FORBIDDEN_SCRIPT_CLAIMS
    ):
        raise ValueError("客服沟通话术包含禁止的确定性承诺。")
    for action in plan.next_actions:
        if FORBIDDEN_SCRIPT_PATTERN.search(action):
            raise ValueError("下一步行动包含未经工具确认的数字或金额表达。")
        if any(claim in action for claim in FORBIDDEN_SCRIPT_CLAIMS):
            raise ValueError("下一步行动包含禁止的确定性承诺。")
    if plan.disclaimer != AGENT_DISCLAIMER:
        raise ValueError("演示免责声明不符合固定文本。")


def _failure(
    code: str,
    message: str,
    records: list[ToolCallRecord] | None = None,
) -> AgentRunResult:
    return AgentRunResult(
        success=False,
        error=AgentError(code=code, message=message),
        tool_calls=records or [],
    )
