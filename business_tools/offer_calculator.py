"""Deterministic demo-offer calculator; no language model is involved."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from .schemas import (
    OfferCalculationInput,
    OfferCalculationOutput,
    OfferPlan,
)


DEMO_POLICY_DISCLAIMER = (
    "本方案仅为项目演示规则，不是真实运营商优惠或审批结果；"
    "执行前必须由人工核验资格、成本和授权。"
)
MONEY_STEP = Decimal("0.01")


@dataclass(frozen=True)
class _OfferRule:
    code: str
    name: str
    rate: Decimal
    months: int
    monthly_cap: Decimal
    rationale: str


class OfferCalculationTool:
    """Select one non-stackable offer from explicit Python rules."""

    def run(self, request: OfferCalculationInput) -> OfferCalculationOutput:
        rule = _select_rule(request)
        calculated_discount = request.monthly_charges * rule.rate
        monthly_discount = min(calculated_discount, rule.monthly_cap).quantize(
            MONEY_STEP,
            rounding=ROUND_HALF_UP,
        )
        total_discount = (monthly_discount * rule.months).quantize(
            MONEY_STEP,
            rounding=ROUND_HALF_UP,
        )
        offer = OfferPlan(
            offer_code=rule.code,
            offer_name=rule.name,
            discount_rate=rule.rate,
            duration_months=rule.months,
            monthly_discount=monthly_discount,
            total_discount=total_discount,
            currency_note="金额沿用公开数据集的未指定计费单位，不代表人民币或真实资费。",
            stackable=False,
            requires_manual_approval=True,
            rationale=rule.rationale,
            disclaimer=DEMO_POLICY_DISCLAIMER,
        )
        return OfferCalculationOutput(success=True, offer=offer)


def _select_rule(request: OfferCalculationInput) -> _OfferRule:
    if request.risk_level == "低风险":
        return _OfferRule(
            code="CARE_ONLY",
            name="常规服务关怀（无价格优惠）",
            rate=Decimal("0"),
            months=0,
            monthly_cap=Decimal("0"),
            rationale="低风险客户不因风险标签主动获得价格优惠。",
        )

    if request.risk_level == "中风险":
        if request.contract == "Month-to-month":
            high_usage = request.monthly_charges >= Decimal("75")
            return _OfferRule(
                code=(
                    "MEDIUM_M2M_HIGH_USAGE"
                    if high_usage
                    else "MEDIUM_M2M_STANDARD"
                ),
                name="月付客户限期体验关怀",
                rate=Decimal("0.08") if high_usage else Decimal("0.05"),
                months=2,
                monthly_cap=Decimal("8") if high_usage else Decimal("5"),
                rationale="中风险月付客户可获得一次限期、不可叠加的演示候选方案。",
            )
        return _OfferRule(
            code="MEDIUM_TERM_STANDARD",
            name="长期合同服务关怀",
            rate=Decimal("0.03"),
            months=1,
            monthly_cap=Decimal("3"),
            rationale="中风险长期合同客户以服务回访为主，价格支持保持有限。",
        )

    if request.contract == "Month-to-month":
        high_usage = request.monthly_charges >= Decimal("75")
        return _OfferRule(
            code=(
                "HIGH_M2M_HIGH_USAGE" if high_usage else "HIGH_M2M_STANDARD"
            ),
            name="高风险月付客户限期挽留方案",
            rate=Decimal("0.15") if high_usage else Decimal("0.10"),
            months=3,
            monthly_cap=Decimal("15") if high_usage else Decimal("10"),
            rationale="高风险月付客户进入人工复核后，可考虑较高但封顶的演示方案。",
        )
    if request.contract == "One year":
        return _OfferRule(
            code="HIGH_ONE_YEAR",
            name="高风险一年合同关怀方案",
            rate=Decimal("0.08"),
            months=2,
            monthly_cap=Decimal("10"),
            rationale="一年合同仍在约客户使用较低折扣和较短期限的演示方案。",
        )
    return _OfferRule(
        code="HIGH_TWO_YEAR",
        name="高风险两年合同关怀方案",
        rate=Decimal("0.05"),
        months=2,
        monthly_cap=Decimal("8"),
        rationale="两年合同客户优先解决服务问题，仅保留有限演示优惠。",
    )
