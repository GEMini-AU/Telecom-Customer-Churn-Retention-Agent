# ============================================================================
# 文件职责：定义旧版 JSONL 审计记录格式，并支持追加人工确认/拒绝记录。
# 主要调用方：SQLite 导出兼容流程和审计测试。
# 输入/输出：输入 RetentionPlan 与人工决定；输出 AuditRecord 或 JSONL 文件行。
# 不负责：当前页面不直接使用它落库；页面的主审计实现位于 sqlite_audit.py。
# ============================================================================
"""人工决定的旧版 JSONL 审计导出格式。

当前页面实际使用 SQLite 审计；本文件仍为导出兼容格式和对应测试提供摘要、指纹与
追加写入能力。它不执行方案，只记录“确认”或“拒绝”的人工决定。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .schemas import RetentionPlan


# JSONL 是旧版导出格式；进程内写锁避免多个线程把两条记录写进同一行。
_AUDIT_WRITE_LOCK = threading.Lock()


class PlanAuditSummary(BaseModel):
    """审计中保留的最小业务摘要，不保存完整对话或完整客户原始行。"""

    # 审计结构固定，避免导出数据混入页面临时字段。
    model_config = ConfigDict(extra="forbid")

    risk_level: str
    churn_probability: float = Field(ge=0, le=1)
    offer_code: str
    offer_name: str
    total_discount: str
    policy_sources: list[str]


class AuditRecord(BaseModel):
    """一条已完成决定的 JSONL 记录，包含时间、客户、方案指纹和摘要。"""
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    timestamp_utc: datetime
    customer_id: str
    decision: Literal["confirmed", "rejected"]
    previous_status: Literal["pending_confirmation"]
    result_status: Literal["confirmed_execution", "rejected"]
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_summary: PlanAuditSummary


def calculate_plan_fingerprint(plan: RetentionPlan) -> str:
    """把完整方案序列化并计算 SHA-256，用于识别页面方案是否被替换。"""
    # 对完整方案做 SHA-256，供 SQLite 审计层判断“当前页面方案是否仍是同一份内容”。
    canonical_json = plan.model_dump_json()
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def append_decision_record(
    audit_path: Path,
    plan: RetentionPlan,
    decision: Literal["confirmed", "rejected"],
) -> AuditRecord:
    """在人点击确认/拒绝后追加一行耐久 JSONL 记录，并返回该记录。"""

    # 同一政策文件可能命中多个片段，导出摘要时保留去重后的来源列表。
    sources = list(
        dict.fromkeys(item.source_file for item in plan.policy_evidence)
    )
    result_status = (
        "confirmed_execution" if decision == "confirmed" else "rejected"
    )
    record = AuditRecord(
        audit_id=str(uuid4()),
        timestamp_utc=datetime.now(timezone.utc),
        customer_id=plan.customer_basic.customer_id,
        decision=decision,
        previous_status="pending_confirmation",
        result_status=result_status,
        plan_fingerprint=calculate_plan_fingerprint(plan),
        plan_summary=PlanAuditSummary(
            risk_level=plan.risk_assessment.risk_level,
            churn_probability=plan.risk_assessment.churn_probability,
            offer_code=plan.recommended_offer.offer_code,
            offer_name=plan.recommended_offer.offer_name,
            total_discount=str(plan.recommended_offer.total_discount),
            policy_sources=sources,
        ),
    )

    audit_path = audit_path.resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # 紧凑 JSON 使每条记录严格占一行，便于后续逐行读取与导出。
    serialized = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    # flush + fsync 尽量确保人工点击后的本地记录已真正落盘。
    with _AUDIT_WRITE_LOCK:
        with audit_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(f"{serialized}\n")
            file.flush()
            os.fsync(file.fileno())
    return record
