"""Offline Chinese-friendly stateless embedding implementation."""

from sklearn.feature_extraction.text import HashingVectorizer


EMBEDDING_DIMENSIONS = 2**16


def create_local_embedder() -> HashingVectorizer:
    """Create fixed-size character embeddings that support incremental updates."""

    return HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        lowercase=True,
        norm="l2",
        n_features=EMBEDDING_DIMENSIONS,
        alternate_sign=False,
    )
