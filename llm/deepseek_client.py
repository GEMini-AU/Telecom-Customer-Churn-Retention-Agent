# ============================================================================
# 文件职责：从环境变量读取 DeepSeek 配置，并通过 OpenAI 兼容 SDK 获取结构化挽留建议。
# 主要调用方：旧页面/独立测试；Agent 自己只复用其中的 DeepSeekSettings。
# 输入/输出：输入环境变量和 CustomerRiskContext；输出 RetentionAdvice 或配置/响应异常。
# 不负责：不训练流失模型、不决定真实优惠、不检索电信知识库。
# ============================================================================
"""独立的 DeepSeek 大模型客户端。

调用者可以是旧页面或独立脚本；它负责“环境变量配置 -> OpenAI 兼容请求 ->
Pydantic 结构校验”。它不查询客户、不预测流失、不检索政策，也不计算优惠。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .schemas import CustomerRiskContext, RetentionAdvice


# 默认值只提供连接参数；真实 API Key 必须由环境变量提供，不能写入源码。
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"


class DeepSeekConfigurationError(RuntimeError):
    """环境变量缺失或格式非法时抛出；请求尚未发送。"""


class DeepSeekResponseError(RuntimeError):
    """远端返回空内容或不符合 ``RetentionAdvice`` 时抛出。"""


@dataclass(frozen=True)
class DeepSeekSettings:
    """从环境变量读取的不可变连接配置。

    ``frozen=True`` 表示实例创建后不能修改，避免同一进程中 API 地址或超时被意外改写。
    """

    api_key: str
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> "DeepSeekSettings":
        """读取并校验环境变量，成功返回配置对象，失败抛配置异常。"""
        # 在发起任何请求前完成读取和校验，让配置错误尽早、明确地暴露。
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
        # 环境变量都是字符串，超时需要显式转换为 float。
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

        # rstrip('/') 避免 base_url 与 SDK 拼接路径时出现双斜杠。
        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            timeout_seconds=timeout_seconds,
        )


class DeepSeekRetentionClient:
    """调用 DeepSeek 生成建议文本，但不依赖 Streamlit 页面。

    输入必须是已确认的 ``CustomerRiskContext``；输出必须通过 ``RetentionAdvice`` 校验。
    """

    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        openai_client: Any | None = None,
    ) -> None:
        """使用给定配置或环境变量创建 OpenAI 兼容客户端；测试可注入假客户端。"""
        # 可注入 OpenAI 兼容客户端，便于离线测试而无需真实密钥或网络。
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
        """调用模型并校验 JSON；风险等级与原因最终以输入事实强制覆盖。"""

        # 要求 JSON 输出；模型只补充建议和话术，风险等级与原因仍以后续强制覆盖为准。
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

        # SDK 返回候选列表；空列表代表没有可解析的模型回复。
        if not response.choices:
            raise DeepSeekResponseError("DeepSeek 没有返回候选结果。")

        # 只取首个候选的文本内容；后续不能直接信任，仍要经 Pydantic 校验。
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise DeepSeekResponseError("DeepSeek 返回了空内容。")

        try:
            advice = RetentionAdvice.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            raise DeepSeekResponseError(
                "DeepSeek 返回内容不符合 RetentionAdvice 结构。"
            ) from error

        # 防止模型改写模型/规则已确认的风险事实。
        return advice.model_copy(
            update={
                "risk_level": context.risk_level,
                "risk_reasons": context.risk_reasons,
            }
        )

    @staticmethod
    def _build_user_prompt(context: CustomerRiskContext) -> str:
        """把已确认客户事实和目标 JSON Schema 组成用户提示词。"""
        # 将输入事实与目标 JSON Schema 一起发送，减少自由文本导致的解析失败。
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
