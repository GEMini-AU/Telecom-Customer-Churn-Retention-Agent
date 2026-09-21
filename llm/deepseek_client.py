"""DeepSeek client built on the installed OpenAI-compatible SDK."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .schemas import CustomerRiskContext, RetentionAdvice


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"


class DeepSeekConfigurationError(RuntimeError):
    """Raised when required DeepSeek environment configuration is missing."""


class DeepSeekResponseError(RuntimeError):
    """Raised when DeepSeek returns empty or schema-invalid content."""


@dataclass(frozen=True)
class DeepSeekSettings:
    """Environment-backed settings for the DeepSeek compatible endpoint."""

    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "DeepSeekSettings":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise DeepSeekConfigurationError(
                "未设置环境变量 DEEPSEEK_API_KEY，无法调用 DeepSeek。"
            )

        model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
        base_url = os.getenv(
            "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL
        ).strip()
        timeout_text = os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60").strip()
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as error:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_TIMEOUT_SECONDS 必须是有效数字。"
            ) from error

        if not model:
            raise DeepSeekConfigurationError("DEEPSEEK_MODEL 不能为空。")
        if not base_url:
            raise DeepSeekConfigurationError("DEEPSEEK_BASE_URL 不能为空。")
        if timeout_seconds <= 0:
            raise DeepSeekConfigurationError(
                "DEEPSEEK_TIMEOUT_SECONDS 必须大于 0。"
            )

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            timeout_seconds=timeout_seconds,
        )


class DeepSeekRetentionClient:
    """Generate validated retention advice without coupling to Streamlit."""

    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        openai_client: Any | None = None,
    ) -> None:
        self.settings = settings or DeepSeekSettings.from_env()
        self._client = openai_client or OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            max_retries=1,
        )

    def generate_retention_advice(
        self,
        context: CustomerRiskContext,
    ) -> RetentionAdvice:
        """Call DeepSeek and return schema-validated retention advice."""

        response = self._client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是谨慎的电信客户运营辅助分析助手。"
                        "请只依据用户提供的事实生成建议，并以 JSON 输出。"
                        "不得虚构优惠金额、企业政策或客户信息；不得保证客户一定留存；"
                        "如提到优惠，只能表述为需要按企业实际政策审批。"
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(context),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        if not response.choices:
            raise DeepSeekResponseError("DeepSeek 没有返回候选结果。")

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise DeepSeekResponseError("DeepSeek 返回了空内容。")

        try:
            advice = RetentionAdvice.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            raise DeepSeekResponseError(
                "DeepSeek 返回内容不符合 RetentionAdvice 结构。"
            ) from error

        return advice.model_copy(
            update={
                "risk_level": context.risk_level,
                "risk_reasons": context.risk_reasons,
            }
        )

    @staticmethod
    def _build_user_prompt(context: CustomerRiskContext) -> str:
        output_schema = json.dumps(
            RetentionAdvice.model_json_schema(),
            ensure_ascii=False,
            indent=2,
        )
        return (
            "请为以下客户生成结构化挽留建议。客户数据为项目演示数据。\n\n"
            f"客户风险信息：\n{context.model_dump_json(indent=2)}\n\n"
            "输出要求：\n"
            "1. risk_level 必须复制输入中的风险等级。\n"
            "2. risk_reasons 必须复制输入中的风险原因，不得添加因果断言。\n"
            "3. recommended_actions 提供 1 到 3 条可执行建议。\n"
            "4. communication_script 提供一段简洁、尊重客户的中文话术。\n"
            "5. 只返回符合下列 JSON Schema 的 JSON 对象，不要添加 Markdown。\n\n"
            f"JSON Schema：\n{output_schema}"
        )
