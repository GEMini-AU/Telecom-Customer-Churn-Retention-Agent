# ============================================================================
# 文件职责：大模型包的公开导入入口，向外暴露客户端和结构化建议类型。
# 主要调用方：旧页面、独立测试脚本，未来可被其他页面复用。
# 输入/输出：无运行时输入；输出可从 ``llm`` 直接导入的公开名称。
# 不负责：不读取客户 CSV、不预测概率、不检索 RAG 或计算优惠。
# ============================================================================
"""大模型集成包的公开入口。

对外提供独立客户端和两类结构化数据；密钥读取、网络请求和 JSON 校验仍在
``deepseek_client.py`` 中，页面无需直接创建 SDK 客户端。
"""

# 只重导出调用方需要的稳定名称，避免泄露内部实现细节。
from .deepseek_client import DeepSeekRetentionClient
from .schemas import CustomerRiskContext, RetentionAdvice

__all__ = [
    "CustomerRiskContext",
    "DeepSeekRetentionClient",
    "RetentionAdvice",
]
