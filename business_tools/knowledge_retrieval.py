# ============================================================================
# 文件职责：将本地 RAG 检索服务封装成可独立调用的知识检索工具。
# 主要调用方：agent/registry.py 的 search_retention_policy 工具适配器。
# 输入/输出：输入问题、top_k 和相似度门槛；输出带来源的 KnowledgeSearchOutput。
# 不负责：不调用大模型生成回答，不把客户 CSV 写进知识库。
# ============================================================================
"""知识检索工具。

调用链：Agent 工具注册表 -> 本工具 -> ``rag.service.DemoTelecomRAG.retrieve``。
输入是问题、返回条数和相似度门槛；输出只含带来源的证据片段，不生成答案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.service import DemoTelecomRAG

from .errors import ToolError
from .schemas import (
    KnowledgeEvidence,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)


class KnowledgeRetrievalTool:
    """将独立 RAG 服务适配成 Agent 可调用的、类型固定的检索工具。"""

    def __init__(
        self,
        project_root: Path,
        rag_service: Any | None = None,
    ) -> None:
        """绑定项目根目录；测试可传入 ``rag_service`` 替身以避免真实磁盘依赖。"""
        # RAG 服务可注入测试替身；默认使用项目内持久化的演示知识库。
        self.project_root = project_root.resolve()
        self._rag = rag_service or DemoTelecomRAG(self.project_root)

    def run(self, request: KnowledgeSearchInput) -> KnowledgeSearchOutput:
        """检索并保留 ``chunk_id``、来源、标题、片段和相似度。"""
        # 此工具只返回检索证据和来源，不在这里调用大模型生成答案。
        try:
            results = self._rag.retrieve(
                request.question,
                top_k=request.top_k,
                min_score=request.min_score,
            )
        except Exception as error:
            return KnowledgeSearchOutput(
                success=False,
                has_sufficient_evidence=False,
                error=ToolError(
                    code="KNOWLEDGE_RETRIEVAL_ERROR",
                    message=f"知识检索失败：{type(error).__name__}：{error}",
                ),
            )

        # 将 RAG 返回对象转换为工具统一的 Pydantic 输出，保留来源和 chunk_id。
        evidence = [
            KnowledgeEvidence.model_validate(item.model_dump())
            for item in results
        ]
        return KnowledgeSearchOutput(
            success=True,
            has_sufficient_evidence=bool(evidence),
            evidence=evidence,
        )
