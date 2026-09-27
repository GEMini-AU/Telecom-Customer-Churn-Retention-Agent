# ============================================================================
# 文件职责：把用户新建的演示客户保存在独立 SQLite，不改写公开训练 CSV。
# 主要调用方：agent_app.py 的新客户表单和 CustomerLookupTool。
# 输入/输出：输入经过 Pydantic 校验的 CustomerProfile；输出持久化后的查询结果。
# 不负责：不训练模型、不保存真实姓名或联系方式、不管理生产客户资料。
# ============================================================================
"""自定义演示客户的最小本地存储。数据库只保存模型需要的 19 个字段。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .schemas import CustomerProfile


class CustomCustomerStore:
    """隔离保存手工客户；每次查询都访问数据库以看到页面新添加的记录。"""

    def __init__(self, database_path: Path) -> None:
        """记录数据库路径；构造对象时不创建文件。"""
        self.database_path = database_path.resolve()

    def _connect(self) -> sqlite3.Connection:
        """按需建表；SQLite 参数化查询避免把客户 ID 拼接进 SQL。"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS custom_customers ("
            "customer_id TEXT PRIMARY KEY, features_json TEXT NOT NULL, "
            "created_at_utc TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        return connection

    def add(self, profile: CustomerProfile) -> None:
        """只新增，不覆盖旧客户；重复 ID 由数据库唯一约束拒绝。"""
        # closing 负责关闭连接，内层事务上下文负责成功提交/异常回滚。
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO custom_customers(customer_id, features_json) VALUES (?, ?)",
                (profile.customer_id, profile.features.model_dump_json()),
            )

    def get(self, customer_id: str) -> CustomerProfile | None:
        """按 ID 读取一个客户；数据库不存在时返回 None。"""
        # 普通公开客户查询不应仅为检查自定义客户而创建一个空数据库。
        if not self.database_path.exists():
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT features_json FROM custom_customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
        if row is None:
            return None
        return CustomerProfile(
            customer_id=customer_id,
            features=json.loads(row[0]),
        )

    def list_ids(self) -> list[str]:
        """仅返回手工客户 ID，供页面选择，不暴露全部模型特征。"""
        if not self.database_path.exists():
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT customer_id FROM custom_customers ORDER BY customer_id"
            ).fetchall()
        return [row[0] for row in rows]
