"""Independent retrieval and grounded-answer service."""

from __future__ import annotations

from pathlib import Path

from .errors import RagIndexError, RagRetrievalError
from .generation import AnswerGenerator, ExtractiveAnswerGenerator
from .knowledge_base import (
    calculate_knowledge_hash,
    load_markdown_documents,
    split_documents,
)
from .models import (
    DocumentLoadResult,
    IndexUpdateReport,
    RagAnswer,
    SearchResult,
)
from .vector_store import LocalVectorStore


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "知识库依据不足，无法基于当前项目演示政策回答该问题。"
    "请查询正式资料或交由人工确认，不能据此编造结论。"
)
RETRIEVAL_ERROR_MESSAGE = (
    "检索服务暂时不可用，无法确认知识库依据。"
    "请稍后重试或交由人工确认，不能在缺少依据时生成结论。"
)
DEMO_ANSWER_PREFIX = (
    "以下回答仅依据项目演示知识库，不是真实运营商内部政策：\n"
)


class DemoTelecomRAG:
    """Load, incrementally index, retrieve, and answer demo documents."""

    def __init__(
        self,
        project_root: Path,
        knowledge_dir: Path | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.knowledge_dir = (
            knowledge_dir or self.project_root / "knowledge" / "demo_telecom"
        ).resolve()
        self.index_path = (
            index_path
            or self.project_root / "storage" / "rag" / "demo_telecom.joblib"
        ).resolve()
        self._store: LocalVectorStore | None = None
        self.last_update_report: IndexUpdateReport | None = None

    def ensure_index(self, force_rebuild: bool = False) -> LocalVectorStore:
        """Reuse, incrementally update, or safely rebuild the local index."""

        try:
            knowledge_hash = calculate_knowledge_hash(self.knowledge_dir)
            load_result = load_markdown_documents(
                self.knowledge_dir,
                self.project_root,
            )
        except Exception as error:
            raise RagIndexError(
                f"无法读取知识库文档：{type(error).__name__}"
            ) from error

        if force_rebuild:
            return self._full_rebuild(
                load_result,
                knowledge_hash,
                reason="force_rebuild",
            )

        if not self.index_path.exists():
            return self._full_rebuild(
                load_result,
                knowledge_hash,
                reason="index_missing",
            )

        try:
            existing = LocalVectorStore.load(self.index_path)
        except RagIndexError as error:
            return self._full_rebuild(
                load_result,
                knowledge_hash,
                reason=f"index_load_failed:{type(error.__cause__ or error).__name__}",
            )

        if existing.knowledge_hash == knowledge_hash:
            self._store = existing
            self.last_update_report = IndexUpdateReport(
                mode="reused",
                skipped_duplicates=[
                    item.skipped_source for item in load_result.duplicates
                ],
                retained_chunk_count=len(existing.chunks),
                reason="knowledge_hash_unchanged",
            )
            return existing

        current_by_source = {
            document.source_file: document
            for document in load_result.documents
        }
        current_sources = set(current_by_source)
        previous_sources = set(existing.document_manifest)
        added_sources = current_sources - previous_sources
        deleted_sources = previous_sources - current_sources
        updated_sources = {
            source
            for source in current_sources & previous_sources
            if current_by_source[source].content_md5
            != existing.document_manifest[source].content_md5
        }
        replaced_sources = added_sources | updated_sources
        replacement_documents = [
            current_by_source[source] for source in sorted(replaced_sources)
        ]
        replacement_chunks = split_documents(replacement_documents)
        excluded_sources = replaced_sources | deleted_sources
        retained_chunk_count = sum(
            chunk.source_file not in excluded_sources
            for chunk in existing.chunks
        )

        try:
            store = LocalVectorStore.incremental_update(
                existing=existing,
                current_documents=load_result.documents,
                replacement_chunks=replacement_chunks,
                replaced_sources=replaced_sources,
                deleted_sources=deleted_sources,
                knowledge_hash=knowledge_hash,
                duplicate_documents=load_result.duplicates,
            )
            store.save(self.index_path)
        except Exception as error:
            return self._full_rebuild(
                load_result,
                knowledge_hash,
                reason=(
                    "incremental_update_failed:"
                    f"{type(error.__cause__ or error).__name__}"
                ),
            )

        self._store = store
        self.last_update_report = IndexUpdateReport(
            mode="incremental_update",
            added_sources=sorted(added_sources),
            updated_sources=sorted(updated_sources),
            deleted_sources=sorted(deleted_sources),
            skipped_duplicates=[
                item.skipped_source for item in load_result.duplicates
            ],
            retained_chunk_count=retained_chunk_count,
            new_chunk_count=len(replacement_chunks),
            reason="knowledge_hash_changed",
        )
        return store

    def retrieve(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.10,
    ) -> list[SearchResult]:
        try:
            store = self._store or self.ensure_index()
            return store.search(question, top_k=top_k, min_score=min_score)
        except RagRetrievalError:
            raise
        except Exception as error:
            raise RagRetrievalError(
                f"知识检索不可用：{type(error).__name__}"
            ) from error

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.10,
        generator: AnswerGenerator | None = None,
    ) -> RagAnswer:
        try:
            evidence = self.retrieve(
                question,
                top_k=top_k,
                min_score=min_score,
            )
        except RagRetrievalError as error:
            return RagAnswer(
                question=question,
                answer=RETRIEVAL_ERROR_MESSAGE,
                has_sufficient_evidence=False,
                evidence=[],
                status="retrieval_error",
                error_type=type(error.__cause__ or error).__name__,
            )

        if not evidence:
            return RagAnswer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_MESSAGE,
                has_sufficient_evidence=False,
                evidence=[],
                status="insufficient_evidence",
            )

        answer_generator = generator or ExtractiveAnswerGenerator()
        status = "answered"
        error_type: str | None = None
        try:
            generated_answer = answer_generator.generate(question, evidence)
            if not generated_answer.strip():
                raise ValueError("回答生成器返回空内容。")
        except Exception as error:
            generated_answer = ExtractiveAnswerGenerator().generate(
                question,
                evidence,
            )
            status = "generation_fallback"
            error_type = type(error).__name__

        return RagAnswer(
            question=question,
            answer=f"{DEMO_ANSWER_PREFIX}{generated_answer}",
            has_sufficient_evidence=True,
            evidence=evidence,
            status=status,
            error_type=error_type,
        )

    def _full_rebuild(
        self,
        load_result: DocumentLoadResult,
        knowledge_hash: str,
        reason: str,
    ) -> LocalVectorStore:
        chunks = split_documents(load_result.documents)
        store = LocalVectorStore.build(
            documents=load_result.documents,
            chunks=chunks,
            knowledge_hash=knowledge_hash,
            duplicate_documents=load_result.duplicates,
        )
        store.save(self.index_path)
        self._store = store
        self.last_update_report = IndexUpdateReport(
            mode="full_rebuild",
            added_sources=sorted(store.document_manifest),
            skipped_duplicates=[
                item.skipped_source for item in load_result.duplicates
            ],
            new_chunk_count=len(chunks),
            reason=reason,
        )
        return store
