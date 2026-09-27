# ============================================================================
# 文件职责：把 SQLite 中已经确认或拒绝的审计记录导出为兼容的 JSONL 文件。
# 主要调用方：用户手工运行 ``python scripts/export_audit.py``。
# 输入/输出：输入 storage/audit 下的 SQLite；输出 JSONL 导出文件及条数提示。
# 不负责：不创建新方案、不改变已有审计状态、不导出仍待确认的方案。
# ============================================================================
"""把 SQLite 中已确认/已拒绝的人工决定导出为旧版 JSONL。

运行：``python scripts/export_audit.py``。待确认方案不会被导出，原 SQLite 数据不会被删除。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.sqlite_audit import export_decisions_jsonl


def main() -> None:
    """定位项目审计库和导出目标，执行导出并打印记录数。"""
    db_path = PROJECT_ROOT / "storage" / "audit" / "agent_decisions.sqlite3"
    export_path = PROJECT_ROOT / "storage" / "audit" / "agent_decisions_export.jsonl"
    count = export_decisions_jsonl(db_path, export_path)
    print(f"已导出 {count} 条已确认/已拒绝的审计记录到 {export_path}")


if __name__ == "__main__":
    main()
