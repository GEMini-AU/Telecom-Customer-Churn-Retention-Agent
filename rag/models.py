"""Data contracts used by the local RAG pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    content_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str
    source_file: str
    text: str


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    document_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    document_title: str
    section_title: str
    source_file: str
    text: str


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    document_title: str
    section_title: str
    source_file: str
    text: str
    score: float = Field(ge=0, le=1)


class DuplicateDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    retained_source: str
    skipped_source: str


class DocumentLoadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[KnowledgeDocument]
    duplicates: list[DuplicateDocument]


class DocumentManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    content_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str
    source_file: str
    chunk_ids: list[str]


class IndexUpdateReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["reused", "full_rebuild", "incremental_update"]
    added_sources: list[str] = Field(default_factory=list)
    updated_sources: list[str] = Field(default_factory=list)
    deleted_sources: list[str] = Field(default_factory=list)
    skipped_duplicates: list[str] = Field(default_factory=list)
    retained_chunk_count: int = Field(default=0, ge=0)
    new_chunk_count: int = Field(default=0, ge=0)
    reason: str | None = None


class RagAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    has_sufficient_evidence: bool
    evidence: list[SearchResult]
    status: Literal[
        "answered",
        "insufficient_evidence",
        "retrieval_error",
        "generation_fallback",
    ] = "answered"
    error_type: str | None = None
