# ============================================================================
# 文件职责：在检索证据已经存在后，生成摘录式或受证据约束的 DeepSeek 回答。
# 主要调用方：rag/service.py 的 answer_question 方法。
# 输入/输出：输入问题和 SearchResult 列表；输出文本回答或 RagGenerationError。
# 不负责：不进行向量检索，不允许无证据调用模型，也不承担 Agent 工具选择。
# ============================================================================
"""基于 RAG 证据的回答生成器。

调用链：``DemoTelecomRAG.answer_question`` 先检索，再调用本模块生成回答。
离线生成器仅整理片段；DeepSeek 生成器可润色，但必须引用本次传入的 chunk。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm.deepseek_client import DeepSeekSettings

from .models import SearchResult


class RagGenerationError(RuntimeError):
    """生成内容为空、格式错误或引用越界时抛出，服务层会回退到摘录式回答。"""


class AnswerGenerator(Protocol):
    """回答生成器的共同接口：输入问题和证据，返回纯文本答案。"""

    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        """Generate an answer using only the supplied evidence."""


class ExtractiveAnswerGenerator:
    """确定性的离线回答器：只拼接实际检索片段，不扩写政策。"""

    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        """忽略问题的措辞，将每条证据压缩为“标题/章节：正文”的可读列表。"""
        del question
        answer_parts: list[str] = []
        for item in evidence:
            compact_text = " ".join(item.text.split())
            answer_parts.append(
                f"- 《{item.document_title}》/{item.section_title}：{compact_text}"
            )
        return "\n".join(answer_parts)


class _DeepSeekAnswerPayload(BaseModel):
    """DeepSeek RAG 生成器期待的最小 JSON；内部使用，不直接暴露给页面。"""
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    used_chunk_ids: list[str] = Field(min_length=1)


class DeepSeekGroundedAnswerGenerator:
    """可选的 DeepSeek 证据回答器；它是 RAG 组件，不会选择或调用业务工具。"""

    def __init__(
        self,
        settings: DeepSeekSettings | None = None,
        openai_client: Any | None = None,
    ) -> None:
        """用环境配置创建客户端；可注入假客户端用于离线测试。"""
        # 与 Agent 相同，支持注入客户端做离线测试；正常情况下从环境变量读取配置。
        self.settings = settings or DeepSeekSettings.from_env()
        self._client = openai_client or OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
            timeout=self.settings.timeout_seconds,
            max_retries=1,
        )

    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        """仅根据 ``evidence`` 生成答案，并验证模型引用的 chunk 不越界。"""
        # 证据为空时严禁调用大模型，防止“看似自然”的无依据回答。
        if not evidence:
            raise RagGenerationError("没有检索证据，禁止调用大模型生成答案。")

        # 连同 chunk_id 与来源一起发送，模型返回后还要校验引用没有超出给定证据。
        # 发送文本和来源元数据，让模型无法声称引用不存在的资料。
        context = [
            {
                "chunk_id": item.chunk_id,
                "document_md5": item.document_md5,
                "title": item.document_title,
                "section": item.section_title,
                "source_file": item.source_file,
                "text": item.text,
            }
            for item in evidence
        ]
        response = self._client.chat.completions.create(
            model=self.settings.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是电信项目演示知识库问答助手。只能使用提供的检索片段回答，"
                        "不得补充片段中没有的金额、政策、时限或事实。"
                        "这些材料均为项目演示政策，不是真实运营商内部政策。"
                        "请返回 JSON。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{question}\n\n"
                        "检索片段：\n"
                        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
                        "请返回 answer 和 used_chunk_ids。"
                        "used_chunk_ids 只能包含实际使用的片段编号。"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        if not response.choices:
            raise RagGenerationError("DeepSeek 没有返回候选结果。")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RagGenerationError("DeepSeek 返回了空内容。")

        try:
            payload = _DeepSeekAnswerPayload.model_validate_json(content)
        except (ValidationError, ValueError) as error:
            raise RagGenerationError("DeepSeek 返回内容不符合 RAG 输出结构。") from error

        # 只接受模型引用本次提供的 chunk_id，阻止伪造来源。
        available_ids = {item.chunk_id for item in evidence}
        used_ids = set(payload.used_chunk_ids)
        if not used_ids.issubset(available_ids):
            raise RagGenerationError("DeepSeek 引用了未提供的知识片段。")
        return payload.answer.strip()
