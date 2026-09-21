"""Offline chat-completions doubles for deterministic Agent tests."""

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
    """Choose the required tools from messages and then emit grounded JSON."""

    def __init__(self, tamper_probability: bool = False) -> None:
        self.call_number = 0
        self.tamper_probability = tamper_probability

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.call_number += 1
        messages = kwargs["messages"]
        user_text = next(
            item["content"] for item in messages if item["role"] == "user"
        )
        customer_id = CUSTOMER_ID_PATTERN.search(user_text).group(0)
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
    def create(self, **_: Any) -> SimpleNamespace:
        return _tool_response(1, "browse_customer_database", {})


class LoopingToolCompletions:
    def __init__(self, customer_id: str) -> None:
        self.customer_id = customer_id
        self.call_number = 0

    def create(self, **_: Any) -> SimpleNamespace:
        self.call_number += 1
        return _tool_response(
            self.call_number,
            GET_CUSTOMER_PROFILE,
            {"customer_id": self.customer_id},
        )


def build_fake_client(completions: Any) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _tool_response(
    call_number: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> SimpleNamespace:
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
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])
