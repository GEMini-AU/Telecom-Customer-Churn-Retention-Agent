"""Persistent local vector store with document-level incremental updates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .embedding import create_local_embedder
from .errors import RagIndexError, RagRetrievalError
from .models import (
    DocumentManifestEntry,
    DuplicateDocument,
    KnowledgeChunk,
    KnowledgeDocument,
    SearchResult,
)


INDEX_VERSION = 3


class LocalVectorStore:
    """Persist fixed-size embeddings, chunks, and a document manifest."""

    def __init__(
        self,
        vectorizer: HashingVectorizer,
        matrix: sparse.spmatrix,
        chunks: list[KnowledgeChunk],
        knowledge_hash: str,
        document_manifest: dict[str, DocumentManifestEntry],
        duplicate_documents: list[DuplicateDocument],
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix.tocsr()
        self.chunks = chunks
        self.knowledge_hash = knowledge_hash
        self.document_manifest = document_manifest
        self.duplicate_documents = duplicate_documents
        self._validate_internal_state()

    @classmethod
    def build(
        cls,
        documents: list[KnowledgeDocument],
        chunks: list[KnowledgeChunk],
        knowledge_hash: str,
        duplicate_documents: list[DuplicateDocument] | None = None,
    ) -> "LocalVectorStore":
        if not chunks:
            raise RagIndexError("没有可用于建立向量索引的文本片段。")

        vectorizer = create_local_embedder()
        matrix = vectorizer.transform([chunk.text for chunk in chunks])
        return cls(
            vectorizer=vectorizer,
            matrix=matrix,
            chunks=chunks,
            knowledge_hash=knowledge_hash,
            document_manifest=_build_manifest(documents, chunks),
            duplicate_documents=duplicate_documents or [],
        )

    @classmethod
    def incremental_update(
        cls,
        existing: "LocalVectorStore",
        current_documents: list[KnowledgeDocument],
        replacement_chunks: list[KnowledgeChunk],
        replaced_sources: set[str],
        deleted_sources: set[str],
        knowledge_hash: str,
        duplicate_documents: list[DuplicateDocument],
    ) -> "LocalVectorStore":
        excluded_sources = replaced_sources | deleted_sources
        retained_indices = [
            index
            for index, chunk in enumerate(existing.chunks)
            if chunk.source_file not in excluded_sources
        ]
        retained_chunks = [existing.chunks[index] for index in retained_indices]
        retained_matrix = existing.matrix[retained_indices]

        if replacement_chunks:
            replacement_matrix = existing.vectorizer.transform(
                [chunk.text for chunk in replacement_chunks]
            )
        else:
            replacement_matrix = sparse.csr_matrix(
                (0, existing.matrix.shape[1]),
                dtype=existing.matrix.dtype,
            )

        chunks = retained_chunks + replacement_chunks
        if not chunks:
            raise RagIndexError("增量更新后没有可检索的文本片段。")

        matrix = sparse.vstack(
            [retained_matrix, replacement_matrix],
            format="csr",
        )
        return cls(
            vectorizer=existing.vectorizer,
            matrix=matrix,
            chunks=chunks,
            knowledge_hash=knowledge_hash,
            document_manifest=_build_manifest(current_documents, chunks),
            duplicate_documents=duplicate_documents,
        )

    def save(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": INDEX_VERSION,
            "knowledge_hash": self.knowledge_hash,
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "document_manifest": {
                source: entry.model_dump()
                for source, entry in self.document_manifest.items()
            },
            "duplicate_documents": [
                item.model_dump() for item in self.duplicate_documents
            ],
        }
        temporary_path = index_path.with_name(
            f".{index_path.name}.{uuid4().hex}.tmp"
        )
        try:
            joblib.dump(payload, temporary_path)
            os.replace(temporary_path, index_path)
        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            raise RagIndexError(
                f"无法原子写入本地向量索引：{type(error).__name__}"
            ) from error

    @classmethod
    def load(cls, index_path: Path) -> "LocalVectorStore":
        # joblib 基于 pickle，只能加载本项目自己生成且受信任的索引文件。
        try:
            payload = joblib.load(index_path)
            if payload.get("version") != INDEX_VERSION:
                raise RagIndexError("向量索引版本不兼容。")

            chunks = [
                KnowledgeChunk.model_validate(item) for item in payload["chunks"]
            ]
            manifest = {
                source: DocumentManifestEntry.model_validate(item)
                for source, item in payload["document_manifest"].items()
            }
            duplicates = [
                DuplicateDocument.model_validate(item)
                for item in payload["duplicate_documents"]
            ]
            return cls(
                vectorizer=payload["vectorizer"],
                matrix=payload["matrix"],
                chunks=chunks,
                knowledge_hash=payload["knowledge_hash"],
                document_manifest=manifest,
                duplicate_documents=duplicates,
            )
        except RagIndexError:
            raise
        except Exception as error:
            raise RagIndexError(
                f"无法读取本地向量索引：{type(error).__name__}"
            ) from error

    def search(
        self,
        question: str,
        top_k: int = 3,
        min_score: float = 0.10,
    ) -> list[SearchResult]:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("检索问题不能为空。")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score 必须位于 0 到 1 之间。")

        try:
            query_vector = self.vectorizer.transform([clean_question])
            scores = linear_kernel(query_vector, self.matrix).ravel()
        except Exception as error:
            raise RagRetrievalError(
                f"向量检索失败：{type(error).__name__}"
            ) from error

        ranked_indices = scores.argsort()[::-1]
        results: list[SearchResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                break
            chunk = self.chunks[int(index)]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_md5=chunk.document_md5,
                    document_title=chunk.document_title,
                    section_title=chunk.section_title,
                    source_file=chunk.source_file,
                    text=chunk.text,
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def _validate_internal_state(self) -> None:
        if self.matrix.shape[0] != len(self.chunks):
            raise RagIndexError("向量行数与文本片段数量不一致。")
        chunk_ids = {chunk.chunk_id for chunk in self.chunks}
        manifest_chunk_ids = {
            chunk_id
            for entry in self.document_manifest.values()
            for chunk_id in entry.chunk_ids
        }
        if chunk_ids != manifest_chunk_ids:
            raise RagIndexError("文档清单与文本片段不一致。")


def _build_manifest(
    documents: list[KnowledgeDocument],
    chunks: list[KnowledgeChunk],
) -> dict[str, DocumentManifestEntry]:
    chunk_ids_by_source: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_ids_by_source.setdefault(chunk.source_file, []).append(chunk.chunk_id)

    return {
        document.source_file: DocumentManifestEntry(
            document_id=document.document_id,
            content_md5=document.content_md5,
            title=document.title,
            source_file=document.source_file,
            chunk_ids=chunk_ids_by_source.get(document.source_file, []),
        )
        for document in documents
    }
