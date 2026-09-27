# ============================================================================
# 文件职责：命令行完整重建本地 RAG 向量索引，并打印建库摘要。
# 主要调用方：用户在终端手工运行 ``python scripts/build_rag_index.py``。
# 输入/输出：输入 knowledge/demo_telecom Markdown；输出 storage/rag 索引和控制台 JSON。
# 不负责：不查询客户、不调用 DeepSeek、不生成 Agent 方案。
# ============================================================================
"""命令行完整重建演示知识库的持久化向量索引。

运行：``python scripts/build_rag_index.py``。输入是 ``knowledge/demo_telecom`` 的
Markdown；输出是本地 joblib 索引和一份 JSON 摘要，不会调用大模型。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


# 以脚本自身定位项目根目录，使其可从任意终端工作目录运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import DemoTelecomRAG


def main() -> None:
    """强制重建索引并打印文档数、重复数、片段数和更新报告。"""
    rag = DemoTelecomRAG(PROJECT_ROOT)
    store = rag.ensure_index(force_rebuild=True)
    # 只打印可审查元数据，不打印任何客户 CSV 或 API Key。
    result = {
        "index_path": str(rag.index_path),
        "document_directory": str(rag.knowledge_dir),
        "unique_document_count": len(store.document_manifest),
        "duplicate_document_count": len(store.duplicate_documents),
        "chunk_count": len(store.chunks),
        "embedding_dimensions": store.matrix.shape[1],
        "knowledge_hash": store.knowledge_hash,
        "update_report": (
            rag.last_update_report.model_dump()
            if rag.last_update_report is not None
            else None
        ),
        "duplicate_documents": [
            item.model_dump() for item in store.duplicate_documents
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
