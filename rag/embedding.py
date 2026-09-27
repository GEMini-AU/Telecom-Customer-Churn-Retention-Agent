# ============================================================================
# 文件职责：创建本地字符 n-gram 向量器，将知识片段和问题转换为固定维度稀疏向量。
# 主要调用方：rag/vector_store.py 的建库和查询流程。
# 输入/输出：输入文本列表；输出 HashingVectorizer 产生的向量矩阵。
# 不负责：不访问 API、不保存索引、不生成自然语言答案。
# ============================================================================
"""本地 RAG 的离线向量化实现。

调用方 ``LocalVectorStore`` 用本文件创建固定维度的 ``HashingVectorizer``。
输入是文本片段或问题；输出是稀疏向量，不依赖 API Key、网络或预训练 Embedding 服务。
"""

from sklearn.feature_extraction.text import HashingVectorizer


# HashingVectorizer 不需训练词表，维度固定，因此可对新增文档做增量向量化。
EMBEDDING_DIMENSIONS = 2**16


def create_local_embedder() -> HashingVectorizer:
    """创建固定维度字符 n-gram 向量器，供建库与查询使用同一规则。"""

    # 中文按字符 n-gram 切分，避免依赖额外分词模型或在线 Embedding API。
    return HashingVectorizer(
        # ``char`` 表示按字符而不是按英文单词切词，更适合中文短语。
        analyzer="char",
        ngram_range=(2, 4),
        lowercase=True,
        norm="l2",
        n_features=EMBEDDING_DIMENSIONS,
        alternate_sign=False,
    )
