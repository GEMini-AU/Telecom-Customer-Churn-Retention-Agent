# ============================================================================
# 文件职责：RAG 包的公开导入入口，集中导出检索服务、结果、异常和可选生成器。
# 主要调用方：business_tools/knowledge_retrieval.py、命令行脚本和测试。
# 输入/输出：无运行时输入；输出可从 ``rag`` 直接导入的公开名称。
# 不负责：不处理客户数据、不计算优惠、不编排 Agent。
# ============================================================================
"""演示电信知识库 RAG 包的公开入口。

对外提供服务、证据结果、可选 DeepSeek 生成器和领域异常；具体的 Markdown 切分、
向量持久化与检索细节保留在各自实现文件中。
"""

# ``__all__`` 是包的明确公共 API，其他内部类不建议被页面或 Agent 直接依赖。
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
