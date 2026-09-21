"""Knowledge retrieval tool that delegates to the standalone RAG service."""

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
    """Return evidence fragments and metadata without calling an LLM."""

    def __init__(
        self,
        project_root: Path,
        rag_service: Any | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self._rag = rag_service or DemoTelecomRAG(self.project_root)

    def run(self, request: KnowledgeSearchInput) -> KnowledgeSearchOutput:
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

        evidence = [
            KnowledgeEvidence.model_validate(item.model_dump())
            for item in results
        ]
        return KnowledgeSearchOutput(
            success=True,
            has_sufficient_evidence=bool(evidence),
            evidence=evidence,
        )
