# ============================================================================
# 文件职责：定义 RAG 文档、片段、检索证据、索引更新报告和回答结果的数据结构。
# 主要调用方：knowledge_base.py、vector_store.py、service.py 与知识检索工具。
# 输入/输出：输入原始字典或 JSON；输出通过字段范围校验的 Pydantic 对象。
# 不负责：不读取文件、不计算向量、不调用大模型。
# ============================================================================
"""本地 RAG 从文档到答案的 Pydantic 数据契约。

数据依次经历 ``KnowledgeDocument``（原文）-> ``KnowledgeChunk``（切片）->
``SearchResult``（命中证据）-> ``RagAnswer``（带状态的回答）。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocument(BaseModel):
    """一篇经过演示声明校验和 MD5 计算的完整知识文档。"""

    # 原始 Markdown 文档：MD5 用于识别内容重复或内容更新。
    model_config = ConfigDict(extra="forbid")

    document_id: str
    content_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str
    source_file: str
    text: str


class KnowledgeChunk(BaseModel):
    """一篇文档按章节/窗口切分后的最小向量检索单元。"""

    # 文档切分后的最小检索单元，保留标题和文件名以便最终引用来源。
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    document_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    document_title: str
    section_title: str
    source_file: str
    text: str


class SearchResult(BaseModel):
    """问题检索命中的一条证据，含来源与 0~1 相似度。"""

    # 向量检索结果：score 已归一化到 0~1，并附带可展示的原文片段。
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    document_title: str
    section_title: str
    source_file: str
    text: str
    score: float = Field(ge=0, le=1)


class DuplicateDocument(BaseModel):
    """内容 MD5 相同的重复文档记录：保留哪个来源、跳过哪个来源。"""
    model_config = ConfigDict(extra="forbid")

    content_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    retained_source: str
    skipped_source: str


class DocumentLoadResult(BaseModel):
    """文档加载结果：可入库的唯一文档加重复文档审计信息。"""
    model_config = ConfigDict(extra="forbid")

    documents: list[KnowledgeDocument]
    duplicates: list[DuplicateDocument]


class DocumentManifestEntry(BaseModel):
    """一个源文件及其所有 chunk_id 的索引清单，用于增量更新。"""
    model_config = ConfigDict(extra="forbid")

    document_id: str
    content_md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str
    source_file: str
    chunk_ids: list[str]


class IndexUpdateReport(BaseModel):
    """建库、复用或增量更新完成后的可审查统计报告。"""

    # 每次建库或增量更新后返回的摘要，方便测试确认没有重复写入。
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
    """RAG 问答的统一结果，显式区分回答、证据不足和检索/生成异常。"""

    # RAG 服务的统一输出；证据不足和检索异常也用状态字段显式区分。
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
