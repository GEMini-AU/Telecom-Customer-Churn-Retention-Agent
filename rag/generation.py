"""Grounded answer generators for retrieved knowledge chunks."""

from __future__ import annotations

import json
from typing import Any, Protocol

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm.deepseek_client import DeepSeekSettings

from .models import SearchResult


class RagGenerationError(RuntimeError):
    """Raised when a generated answer is empty or not grounded in evidence."""


class AnswerGenerator(Protocol):
    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        """Generate an answer using only the supplied evidence."""


class ExtractiveAnswerGenerator:
    """Deterministic offline generator that quotes retrieved chunks."""

    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        del question
        answer_parts: list[str] = []
        for item in evidence:
            compact_text = " ".join(item.text.split())
            answer_parts.append(
                f"- 《{item.document_title}》/{item.section_title}：{compact_text}"
            )
        return "\n".join(answer_parts)


class _DeepSeekAnswerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    used_chunk_ids: list[str] = Field(min_length=1)


class DeepSeekGroundedAnswerGenerator:
    """Optional DeepSeek generator; it is a plain RAG component, not an Agent."""

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

    def generate(self, question: str, evidence: list[SearchResult]) -> str:
        if not evidence:
            raise RagGenerationError("没有检索证据，禁止调用大模型生成答案。")

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

        available_ids = {item.chunk_id for item in evidence}
        used_ids = set(payload.used_chunk_ids)
        if not used_ids.issubset(available_ids):
            raise RagGenerationError("DeepSeek 引用了未提供的知识片段。")
        return payload.answer.strip()
