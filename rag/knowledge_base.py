# ============================================================================
# 文件职责：加载演示政策 Markdown、校验声明、计算 MD5 去重并切分为知识片段。
# 主要调用方：rag/service.py 的 ensure_index 与 full rebuild 流程。
# 输入/输出：输入知识目录中的 Markdown；输出唯一 KnowledgeDocument/KnowledgeChunk 列表。
# 不负责：不读取客户 CSV、不向量化、不回答问题，也不把演示政策伪装为真实政策。
# ============================================================================
"""演示知识库 Markdown 的加载、去重和确定性切分。

调用链：``DemoTelecomRAG.ensure_index`` -> ``load_markdown_documents`` ->
``split_documents``。这里把 Markdown 转为带来源元数据的 ``KnowledgeChunk``，
供向量库持久化和最终引用展示。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import (
    DocumentLoadResult,
    DuplicateDocument,
    KnowledgeChunk,
    KnowledgeDocument,
)


# 每篇知识文档必须显式声明其为演示政策，防止演示内容被误认为真实运营商制度。
DEMO_MARKER = "项目演示政策"
CHUNK_DISCLAIMER = "【项目演示政策，非真实运营商内部政策】"


def load_markdown_documents(
    knowledge_dir: Path,
    project_root: Path,
) -> DocumentLoadResult:
    """加载目录下的 Markdown，校验演示声明并按规范化 MD5 去重。"""
    # 按规范化文本的 MD5 去重：相同内容只保留首个文件，并记录被跳过的来源。
    documents: list[KnowledgeDocument] = []
    duplicates: list[DuplicateDocument] = []
    retained_by_md5: dict[str, tuple[str, str]] = {}
    for path in sorted(knowledge_dir.glob("*.md")):
        text = normalize_document_text(path.read_text(encoding="utf-8"))
        # 演示声明是知识库入库的硬性门槛。
        if DEMO_MARKER not in text:
            raise ValueError(f"知识文档缺少演示政策标记：{path}")

        title = _extract_title(text, path)
        source_file = path.relative_to(project_root).as_posix()
        content_md5 = calculate_document_md5(text)
        # MD5 相同后仍比较全文，避免极低概率碰撞被静默合并。
        retained_document = retained_by_md5.get(content_md5)
        if retained_document is not None:
            retained_source, retained_text = retained_document
            if retained_text != text:
                raise ValueError(
                    f"检测到 MD5 碰撞，拒绝合并文档：{retained_source} 与 {source_file}"
                )
            duplicates.append(
                DuplicateDocument(
                    content_md5=content_md5,
                    retained_source=retained_source,
                    skipped_source=source_file,
                )
            )
            continue

        retained_by_md5[content_md5] = (source_file, text)
        documents.append(
            KnowledgeDocument(
                document_id=path.stem,
                content_md5=content_md5,
                title=title,
                source_file=source_file,
                text=text,
            )
        )

    if not documents:
        raise ValueError(f"知识目录中没有 Markdown 文档：{knowledge_dir}")
    return DocumentLoadResult(documents=documents, duplicates=duplicates)


def split_documents(
    documents: list[KnowledgeDocument],
    max_chars: int = 520,
    overlap: int = 80,
) -> list[KnowledgeChunk]:
    """按二级标题和滑动窗口切分文档，返回可独立检索的带来源片段。"""
    # 先按 Markdown 二级标题分段，再用滑动窗口切片；每个片段都带来源元数据。
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0。")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars。")

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for section_title, section_text in _split_markdown_sections(document):
            # 将演示声明、文档名和章节写入片段正文，检索结果脱离上下文时仍可追溯。
            prefix = (
                f"{CHUNK_DISCLAIMER}\n"
                f"文档：{document.title}\n"
                f"章节：{section_title}\n"
            )
            available_chars = max_chars - len(prefix)
            if available_chars <= overlap:
                raise ValueError("max_chars 太小，无法容纳片段元数据。")

            for part_index, part in enumerate(
                _sliding_windows(section_text, available_chars, overlap)
            ):
                chunk_text = f"{prefix}{part.strip()}"
                raw_id = (
                    f"{document.source_file}|{section_title}|"
                    f"{part_index}|{chunk_text}"
                )
                chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        document_md5=document.content_md5,
                        document_title=document.title,
                        section_title=section_title,
                        source_file=document.source_file,
                        text=chunk_text,
                    )
                )

    return chunks


def calculate_knowledge_hash(knowledge_dir: Path) -> str:
    """计算知识目录整体哈希，用来决定持久化向量库是否需要更新。"""
    # 目录级哈希用于判断是否需要更新持久化索引，包含文件名和原始字节内容。
    digest = hashlib.sha256()
    for path in sorted(knowledge_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def calculate_document_md5(text: str) -> str:
    """Calculate normalized-content MD5 for deduplication, not security."""

    normalized = normalize_document_text(text)
    return hashlib.md5(
        normalized.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def normalize_document_text(text: str) -> str:
    """消除换行、BOM 和行尾空格差异，使内容去重不受排版影响。"""
    # 统一换行、移除 BOM 与行尾空白，避免仅格式不同的文档无法被去重。
    normalized_newlines = (
        text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    )
    return "\n".join(
        line.rstrip() for line in normalized_newlines.splitlines()
    ).strip()


def _extract_title(text: str, path: Path) -> str:
    """读取 Markdown 的首个一级标题；缺失时拒绝入库以保留可展示来源。"""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError(f"知识文档缺少一级标题：{path}")


def _split_markdown_sections(
    document: KnowledgeDocument,
) -> list[tuple[str, str]]:
    """将一级标题后的内容按二级标题分组，返回“章节名、章节正文”列表。"""
    sections: list[tuple[str, str]] = []
    current_title = "文档说明"
    current_lines: list[str] = []

    for line in document.text.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            _append_section(sections, current_title, current_lines)
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    _append_section(sections, current_title, current_lines)
    return sections


def _append_section(
    sections: list[tuple[str, str]],
    title: str,
    lines: list[str],
) -> None:
    """把非空章节追加到结果，避免为空白章节创建无意义向量。"""
    text = "\n".join(lines).strip()
    if text:
        sections.append((title, text))


def _sliding_windows(text: str, size: int, overlap: int) -> list[str]:
    """将长文本切成重叠窗口，减少关键词刚好位于边界时的检索遗漏。"""
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    windows: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        windows.append(normalized[start:end])
        if end == len(normalized):
            break
        start = end - overlap
    return windows
