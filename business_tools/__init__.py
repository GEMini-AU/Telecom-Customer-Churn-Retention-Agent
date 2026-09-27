# ============================================================================
# 文件职责：业务工具包的公开导入入口，集中导出四个工具及其输入输出类型。
# 主要调用方：agent/registry.py、命令行演示脚本和单元测试。
# 输入/输出：无运行时输入；输出可从 ``business_tools`` 直接导入的公开名称。
# 不负责：不编排工具顺序，不保存 Agent 状态，不连接大模型。
# ============================================================================
"""四个可独立测试的电信业务工具的公开入口。

这里不放业务逻辑，只把查询、预测、检索、优惠及其输入输出类型集中重导出，
使 Agent 注册表和脚本无需记住各实现文件的位置。
"""

# ``__all__`` 下面列出的名称是其他模块允许从此包直接导入的稳定接口。
from .churn_prediction import ChurnPredictionTool
from .customer_lookup import CustomerLookupTool
from .knowledge_retrieval import KnowledgeRetrievalTool
from .offer_calculator import OfferCalculationTool
from .schemas import (
    ChurnPredictionInput,
    ChurnPredictionOutput,
    CustomerLookupInput,
    CustomerLookupOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
    OfferCalculationInput,
    OfferCalculationOutput,
)

__all__ = [
    "ChurnPredictionInput",
    "ChurnPredictionOutput",
    "ChurnPredictionTool",
    "CustomerLookupInput",
    "CustomerLookupOutput",
    "CustomerLookupTool",
    "KnowledgeRetrievalTool",
    "KnowledgeSearchInput",
    "KnowledgeSearchOutput",
    "OfferCalculationInput",
    "OfferCalculationOutput",
    "OfferCalculationTool",
]
