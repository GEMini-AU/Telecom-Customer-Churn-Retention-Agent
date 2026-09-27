# ============================================================================
# 文件职责：定义四个业务工具通用的结构化错误对象。
# 主要调用方：customer_lookup.py、churn_prediction.py、knowledge_retrieval.py 等。
# 输入/输出：输入错误码和可读消息；输出 ToolError。
# 不负责：不捕获异常；每个具体工具决定何时构造该对象。
# ============================================================================
"""四个独立业务工具共用的结构化错误协议。

调用方根据 ``code`` 做程序判断，根据 ``message`` 向页面用户展示原因，
因此业务失败无需让页面解析 Python 异常文本。
"""

from pydantic import BaseModel, ConfigDict


class ToolError(BaseModel):
    """业务工具失败时的标准输出：机器可判断的代码加可读提示。"""

    # 不允许错误对象混入未声明字段，避免不同工具返回格式漂移。
    model_config = ConfigDict(extra="forbid")

    # ``code`` 供 Agent、测试和页面分支使用；``message`` 供人阅读。
    code: str
    message: str
