"""Independent local RAG package for the demo telecom knowledge base."""

from .errors import RagIndexError, RagRetrievalError
from .generation import DeepSeekGroundedAnswerGenerator
from .models import RagAnswer, SearchResult
from .service import DemoTelecomRAG, RETRIEVAL_ERROR_MESSAGE

__all__ = [
    "DemoTelecomRAG",
    "DeepSeekGroundedAnswerGenerator",
    "RagAnswer",
    "RagIndexError",
    "RagRetrievalError",
    "RETRIEVAL_ERROR_MESSAGE",
    "SearchResult",
]
