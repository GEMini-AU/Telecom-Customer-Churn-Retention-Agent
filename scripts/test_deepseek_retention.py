"""Minimal live/offline check for the DeepSeek retention client.

Live call: python scripts/test_deepseek_retention.py
Offline:   python scripts/test_deepseek_retention.py --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from llm.deepseek_client import (
    DeepSeekConfigurationError,
    DeepSeekResponseError,
    DeepSeekRetentionClient,
    DeepSeekSettings,
)
from llm.schemas import CustomerRiskContext, RetentionAdvice


def build_virtual_customer() -> CustomerRiskContext:
    return CustomerRiskContext(
        customer_id="DEMO-0001",
        churn_probability=0.72,
        risk_level="高风险",
        risk_reasons=[
            "合同类型为月付合同",
            "客户已使用互联网服务，但未开通技术支持服务",
            "付款方式为电子支票",
        ],
        contract="Month-to-month",
        tenure_months=6,
        monthly_charges=85.0,
        tech_support="No",
        payment_method="Electronic check",
    )


class OfflineCompletions:
    def create(self, **_: object) -> SimpleNamespace:
        content = RetentionAdvice(
            risk_level="高风险",
            risk_reasons=["这些值会被客户端的权威输入覆盖"],
            recommended_actions=[
                "优先安排人工回访，了解近期服务体验",
                "介绍技术支持服务，并确认客户是否需要协助",
            ],
            communication_script=(
                "您好，我们想了解您近期的服务体验，并协助检查是否有需要改进的地方。"
            ),
        ).model_dump_json()
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def build_offline_client() -> DeepSeekRetentionClient:
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=OfflineCompletions())
    )
    settings = DeepSeekSettings(api_key="offline-test-key")
    return DeepSeekRetentionClient(settings=settings, openai_client=fake_client)


def verify_required_fields(advice: RetentionAdvice) -> None:
    assert advice.risk_level in {"低风险", "中风险", "高风险"}
    assert advice.risk_reasons
    assert advice.recommended_actions
    assert advice.communication_script.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline",
        action="store_true",
        help="使用本地模拟响应检查结构，不访问 DeepSeek。",
    )
    args = parser.parse_args()

    context = build_virtual_customer()
    try:
        client = build_offline_client() if args.offline else DeepSeekRetentionClient()
        advice = client.generate_retention_advice(context)
    except (DeepSeekConfigurationError, DeepSeekResponseError) as error:
        raise SystemExit(f"测试未完成：{error}") from error
    verify_required_fields(advice)

    print("结构化挽留建议字段验证通过。")
    print(json.dumps(advice.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
