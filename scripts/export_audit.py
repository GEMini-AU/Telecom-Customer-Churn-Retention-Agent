"""Export finalized SQLite human decisions to the legacy JSONL format."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.sqlite_audit import export_decisions_jsonl


def main() -> None:
    db_path = PROJECT_ROOT / "storage" / "audit" / "agent_decisions.sqlite3"
    export_path = PROJECT_ROOT / "storage" / "audit" / "agent_decisions_export.jsonl"
    count = export_decisions_jsonl(db_path, export_path)
    print(f"已导出 {count} 条已确认/已拒绝的审计记录到 {export_path}")


if __name__ == "__main__":
    main()
