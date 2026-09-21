"""Markdown loading and deterministic text splitting for demo knowledge."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .models import (
    DocumentLoadResult,
    DuplicateDocument,
    KnowledgeChunk,
    KnowledgeDocument,
)


DEMO_MARKER = "项目演示政策"
CHUNK_DISCLAIMER = "【项目演示政策，非真实运营商内部政策】"


def load_markdown_documents(
    knowledge_dir: Path,
    project_root: Path,
) -> DocumentLoadResult:
    documents: list[KnowledgeDocument] = []
    duplicates: list[DuplicateDocument] = []
    retained_by_md5: dict[str, tuple[str, str]] = {}
    for path in sorted(knowledge_dir.glob("*.md")):
        text = normalize_document_text(path.read_text(encoding="utf-8"))
        if DEMO_MARKER not in text:
            raise ValueError(f"知识文档缺少演示政策标记：{path}")

        title = _extract_title(text, path)
        source_file = path.relative_to(project_root).as_posix()
        content_md5 = calculate_document_md5(text)
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
    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0。")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars。")

    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for section_title, section_text in _split_markdown_sections(document):
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
    normalized_newlines = (
        text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    )
    return "\n".join(
        line.rstrip() for line in normalized_newlines.splitlines()
    ).strip()


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    raise ValueError(f"知识文档缺少一级标题：{path}")


def _split_markdown_sections(
    document: KnowledgeDocument,
) -> list[tuple[str, str]]:
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
    text = "\n".join(lines).strip()
    if text:
        sections.append((title, text))


def _sliding_windows(text: str, size: int, overlap: int) -> list[str]:
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
