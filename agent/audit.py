"""Append-only local audit records for human Agent decisions."""

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


_AUDIT_WRITE_LOCK = threading.Lock()


class PlanAuditSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: str
    churn_probability: float = Field(ge=0, le=1)
    offer_code: str
    offer_name: str
    total_discount: str
    policy_sources: list[str]


class AuditRecord(BaseModel):
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
    canonical_json = plan.model_dump_json()
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def append_decision_record(
    audit_path: Path,
    plan: RetentionPlan,
    decision: Literal["confirmed", "rejected"],
) -> AuditRecord:
    """Write one durable JSONL decision after the human clicks a button."""

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
    serialized = json.dumps(
        record.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with _AUDIT_WRITE_LOCK:
        with audit_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(f"{serialized}\n")
            file.flush()
            os.fsync(file.fileno())
    return record
