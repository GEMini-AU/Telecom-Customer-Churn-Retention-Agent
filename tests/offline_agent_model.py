# ============================================================================
# 文件职责：模拟 OpenAI SDK 的模型响应，使单 Agent 测试不依赖真实网络和 API Key。
# 主要调用方：单 Agent、页面、审计和最终验收测试，以及 scripts/test_agent.py。
# 输入/输出：输入模型对话 messages；输出模拟的工具调用或最终 JSON 方案响应。
# 不负责：不调用真实模型，不实现业务工具逻辑，也不代表生产模型行为。
# ============================================================================
"""确定性 Agent 离线测试替身。

这些类模仿 OpenAI SDK 的 ``chat.completions.create`` 返回结构：根据对话中的工具结果
依次请求四个工具，最后拼装 JSON 方案。它们用于测试编排逻辑，不是生产模型。
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

from agent.registry import (
    CALCULATE_RETENTION_OFFER,
    GET_CUSTOMER_PROFILE,
    PREDICT_CHURN_RISK,
    SEARCH_RETENTION_POLICY,
)
from agent.single_agent import AGENT_DISCLAIMER


CUSTOMER_ID_PATTERN = re.compile(r"[0-9]{4}-[A-Z]{5}")


class OfflineAgentCompletions:
    """根据已回传的工具结果选择下一工具，最后仅复制工具事实组成方案。"""

    def __init__(self, tamper_probability: bool = False) -> None:
        """``tamper_probability=True`` 时故意篡改概率，用于验证最终一致性校验。"""
        self.call_number = 0
        self.tamper_probability = tamper_probability

    def create(self, **kwargs: Any) -> SimpleNamespace:
        """模拟 SDK 请求入口；读取消息并返回一次工具调用或最终文本回复。"""
        self.call_number += 1
        messages = kwargs["messages"]
        user_text = next(
            item["content"] for item in messages if item["role"] == "user"
        )
        customer_id = CUSTOMER_ID_PATTERN.search(user_text).group(0)
        # Agent 回传的每条 tool 消息以工具名为键保存，供下一步决定依赖是否已满足。
        outputs = {
            item["name"]: json.loads(item["content"])
            for item in messages
            if item["role"] == "tool"
        }

        if GET_CUSTOMER_PROFILE not in outputs:
            return _tool_response(
                self.call_number,
                GET_CUSTOMER_PROFILE,
                {"customer_id": customer_id},
            )
        if PREDICT_CHURN_RISK not in outputs:
            return _tool_response(
                self.call_number,
                PREDICT_CHURN_RISK,
                {"customer_id": customer_id},
            )
        if SEARCH_RETENTION_POLICY not in outputs:
            prediction = outputs[PREDICT_CHURN_RISK]["data"]
            customer = outputs[GET_CUSTOMER_PROFILE]["data"]
            query = (
                f"{prediction['risk_level']}客户 "
                f"{customer['features']['contract']}合同 挽留政策 优惠限制"
            )
            return _tool_response(
                self.call_number,
                SEARCH_RETENTION_POLICY,
                {"question": query, "top_k": 5, "min_score": 0.10},
            )
        if CALCULATE_RETENTION_OFFER not in outputs:
            return _tool_response(
                self.call_number,
                CALCULATE_RETENTION_OFFER,
                {"customer_id": customer_id},
            )

        customer = outputs[GET_CUSTOMER_PROFILE]["data"]
        features = customer["features"]
        prediction = outputs[PREDICT_CHURN_RISK]["data"]
        evidence = outputs[SEARCH_RETENTION_POLICY]["data"]["evidence"]
        offer = outputs[CALCULATE_RETENTION_OFFER]["data"]
        probability = prediction["churn_probability"]
        if self.tamper_probability:
            probability = min(1.0, probability + 0.01)
        plan = {
            "customer_basic": {
                "customer_id": customer["customer_id"],
                "tenure_months": features["tenure"],
                "contract": features["contract"],
                "monthly_charges": features["monthly_charges"],
                "internet_service": features["internet_service"],
                "tech_support": features["tech_support"],
                "payment_method": features["payment_method"],
            },
            "risk_assessment": {
                "churn_probability": probability,
                "classification_threshold": prediction[
                    "classification_threshold"
                ],
                "prediction_label": prediction["prediction_label"],
                "risk_level": prediction["risk_level"],
                "risk_factors": prediction["risk_factors"],
            },
            "policy_evidence": [
                {
                    "chunk_id": item["chunk_id"],
                    "source_file": item["source_file"],
                    "document_title": item["document_title"],
                    "section_title": item["section_title"],
                    "excerpt": item["text"],
                }
                for item in evidence[:2]
            ],
            "recommended_offer": offer,
            "communication_script": (
                "您好，我们想了解您近期的服务体验，并协助检查适合您的服务方案。"
                "具体方案需要完成资格核验后确认。"
            ),
            "next_actions": [
                "人工复核客户当前套餐、历史优惠和联系意愿",
                "由客服进行一次有明确目的的服务回访",
                "记录客户反馈以及接受、修改或拒绝结果",
            ],
            "disclaimer": AGENT_DISCLAIMER,
        }
        return _text_response(json.dumps(plan, ensure_ascii=False))


class UnknownToolCompletions:
    """故意请求白名单外工具，用于验证 Agent 会拒绝执行。"""
    def create(self, **_: Any) -> SimpleNamespace:
        """忽略输入，始终返回一个白名单外工具请求。"""
        return _tool_response(1, "browse_customer_database", {})


class LoopingToolCompletions:
    """持续请求同一工具，用于验证最大工具调用次数限制。"""
    def __init__(self, customer_id: str) -> None:
        self.customer_id = customer_id
        self.call_number = 0

    def create(self, **_: Any) -> SimpleNamespace:
        """每次请求都返回同一客户查询调用，制造受控循环。"""
        self.call_number += 1
        return _tool_response(
            self.call_number,
            GET_CUSTOMER_PROFILE,
            {"customer_id": self.customer_id},
        )


def build_fake_client(completions: Any) -> SimpleNamespace:
    """将 completions 替身包装成与 OpenAI 客户端访问路径一致的对象。"""
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _tool_response(
    call_number: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> SimpleNamespace:
    """构造包含一个 function tool_call 的最小 SDK 风格响应。"""
    tool_call = SimpleNamespace(
        id=f"offline-call-{call_number}",
        function=SimpleNamespace(
            name=tool_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _text_response(content: str) -> SimpleNamespace:
    """构造没有工具调用、只包含最终文本内容的最小 SDK 风格响应。"""
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])
