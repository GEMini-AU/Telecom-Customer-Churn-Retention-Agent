# ============================================================================
# 文件职责：Agent 包的公开导入入口，集中暴露外部允许使用的 Agent 类型与函数。
# 主要调用方：agent_app.py、命令行脚本和测试。
# 输入/输出：无运行时输入；输出可从 ``agent`` 包直接导入的公开名称。
# 不负责：不执行工具、不调用模型、不写入审计记录。
# ============================================================================
"""单 Agent 编排包的公开入口。

外部页面和脚本从这里导入 ``SingleRetentionAgent``、方案结果与旧版审计导出能力；
具体工具调用、状态校验与 SQLite 审计仍分别位于同目录的实现文件。
"""

# 仅重导出外部使用的类型/函数，``__all__`` 明确 ``from agent import *`` 的公开边界。
from .audit import AuditRecord, append_decision_record
from .schemas import AgentRunResult, RetentionPlan
from .single_agent import SingleRetentionAgent

__all__ = [
    "AgentRunResult",
    "AuditRecord",
    "RetentionPlan",
    "SingleRetentionAgent",
    "append_decision_record",
]
