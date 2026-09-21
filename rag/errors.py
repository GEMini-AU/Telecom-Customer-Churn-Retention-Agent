"""Domain exceptions for the independent RAG module."""


class RagError(RuntimeError):
    """Base exception for recoverable RAG failures."""


class RagIndexError(RagError):
    """Raised when a persisted index is invalid or cannot be written."""


class RagRetrievalError(RagError):
    """Raised when the retrieval pipeline cannot return reliable evidence."""
