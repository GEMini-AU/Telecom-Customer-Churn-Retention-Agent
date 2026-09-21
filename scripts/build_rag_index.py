"""Build the persistent local vector index from demo Markdown documents."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import DemoTelecomRAG


def main() -> None:
    rag = DemoTelecomRAG(PROJECT_ROOT)
    store = rag.ensure_index(force_rebuild=True)
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
