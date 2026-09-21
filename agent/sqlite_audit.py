"""Transactional SQLite audit for human decisions on demo retention plans."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditRecord, PlanAuditSummary, calculate_plan_fingerprint
from .schemas import RetentionPlan


DecisionStatus = Literal["pending_confirmation", "confirmed_execution", "rejected"]
Decision = Literal["confirmed", "rejected"]


class DecisionConflictError(RuntimeError):
    """The plan was already decided or its fingerprint changed."""


class SqliteDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    created_at_utc: datetime
    decided_at_utc: datetime | None
    customer_id: str
    plan_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: DecisionStatus
    decision: Decision | None
    plan_summary: PlanAuditSummary


_SCHEMA = """
CREATE TABLE IF NOT EXISTS retention_decisions (
    audit_id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    decided_at_utc TEXT,
    customer_id TEXT NOT NULL,
    plan_fingerprint TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (
        status IN ('pending_confirmation', 'confirmed_execution', 'rejected')
    ),
    decision TEXT CHECK (decision IN ('confirmed', 'rejected')),
    plan_summary_json TEXT NOT NULL,
    CHECK (
        (status = 'pending_confirmation' AND decision IS NULL AND decided_at_utc IS NULL)
        OR (status = 'confirmed_execution' AND decision = 'confirmed' AND decided_at_utc IS NOT NULL)
        OR (status = 'rejected' AND decision = 'rejected' AND decided_at_utc IS NOT NULL)
    )
)
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute(_SCHEMA)
    return connection


def _summary(plan: RetentionPlan) -> PlanAuditSummary:
    return PlanAuditSummary(
        risk_level=plan.risk_assessment.risk_level,
        churn_probability=plan.risk_assessment.churn_probability,
        offer_code=plan.recommended_offer.offer_code,
        offer_name=plan.recommended_offer.offer_name,
        total_discount=str(plan.recommended_offer.total_discount),
        policy_sources=list(
            dict.fromkeys(item.source_file for item in plan.policy_evidence)
        ),
    )


def _record(row: sqlite3.Row) -> SqliteDecisionRecord:
    return SqliteDecisionRecord(
        audit_id=row["audit_id"],
        created_at_utc=row["created_at_utc"],
        decided_at_utc=row["decided_at_utc"],
        customer_id=row["customer_id"],
        plan_fingerprint=row["plan_fingerprint"],
        status=row["status"],
        decision=row["decision"],
        plan_summary=PlanAuditSummary.model_validate_json(
            row["plan_summary_json"]
        ),
    )


def create_pending_decision(
    db_path: Path,
    plan: RetentionPlan,
) -> SqliteDecisionRecord:
    """Create one pending record per exact plan, or return its existing state."""

    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = calculate_plan_fingerprint(plan)
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO retention_decisions (
                    audit_id, created_at_utc, customer_id, plan_fingerprint,
                    status, plan_summary_json
                ) VALUES (?, ?, ?, ?, 'pending_confirmation', ?)
                ON CONFLICT(plan_fingerprint) DO NOTHING""",
                (
                    str(uuid4()),
                    created_at,
                    plan.customer_basic.customer_id,
                    fingerprint,
                    _summary(plan).model_dump_json(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM retention_decisions WHERE plan_fingerprint = ?",
                (fingerprint,),
            ).fetchone()
            assert row is not None
            return _record(row)


def decide_pending_decision(
    db_path: Path,
    audit_id: str,
    plan: RetentionPlan,
    decision: Decision,
) -> SqliteDecisionRecord:
    """Atomically finalize one pending plan; repeated same clicks are idempotent."""

    db_path = db_path.resolve()
    if not db_path.is_file():
        raise LookupError("审计数据库不存在，无法确认方案。")
    target_status = (
        "confirmed_execution" if decision == "confirmed" else "rejected"
    )
    with closing(_connect(db_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM retention_decisions WHERE audit_id = ?",
                (audit_id,),
            ).fetchone()
            if row is None:
                raise LookupError("未找到待确认的方案审计记录。")
            if row["plan_fingerprint"] != calculate_plan_fingerprint(plan):
                raise DecisionConflictError("方案内容已变化，请重新生成后确认。")
            if row["status"] != "pending_confirmation":
                if row["status"] == target_status:
                    return _record(row)
                raise DecisionConflictError("方案已做出相反决定，不能重复更改。")
            connection.execute(
                """UPDATE retention_decisions
                SET status = ?, decision = ?, decided_at_utc = ?
                WHERE audit_id = ? AND status = 'pending_confirmation'""",
                (
                    target_status,
                    decision,
                    datetime.now(timezone.utc).isoformat(),
                    audit_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM retention_decisions WHERE audit_id = ?",
                (audit_id,),
            ).fetchone()
            assert updated is not None
            return _record(updated)


def list_decision_records(
    db_path: Path,
    limit: int = 10,
) -> list[SqliteDecisionRecord]:
    """Return recent persisted decisions without creating a missing database."""

    if not 1 <= limit <= 1000:
        raise ValueError("limit 必须位于 1 到 1000 之间。")
    db_path = db_path.resolve()
    if not db_path.is_file():
        return []
    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            """SELECT * FROM retention_decisions
            ORDER BY created_at_utc DESC, audit_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [_record(row) for row in rows]


def export_decisions_jsonl(db_path: Path, export_path: Path) -> int:
    """Export finalized SQLite decisions to the legacy JSONL record shape."""

    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError("审计数据库不存在，无法导出。")
    with closing(_connect(db_path)) as connection:
        rows = connection.execute(
            """SELECT * FROM retention_decisions
            WHERE status != 'pending_confirmation'
            ORDER BY decided_at_utc, audit_id"""
        ).fetchall()
    export_path = export_path.resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = export_path.with_name(f"{export_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                decision = _record(row)
                legacy = AuditRecord(
                    audit_id=decision.audit_id,
                    timestamp_utc=decision.decided_at_utc,
                    customer_id=decision.customer_id,
                    decision=decision.decision,
                    previous_status="pending_confirmation",
                    result_status=decision.status,
                    plan_fingerprint=decision.plan_fingerprint,
                    plan_summary=decision.plan_summary,
                )
                file.write(
                    json.dumps(
                        legacy.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, export_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return len(rows)
