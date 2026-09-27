# ============================================================================
# 文件职责：使用普通 Python 演示规则，确定性计算单一客户允许的优惠方案。
# 主要调用方：agent/registry.py 的 calculate_retention_offer 工具适配器。
# 输入/输出：输入风险等级、合同类型、月费；输出 OfferCalculationOutput。
# 不负责：不调用大模型、不自行修改规则金额，也不代表真实运营商审批。
# ============================================================================
"""确定性的演示优惠计算工具。

调用链：预测得到风险等级且客户资料已查到后，Agent 调用本工具。
输入仅为风险等级、合同和月费；输出是固定 Python 规则算出的优惠，模型不能改写金额。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP

from .schemas import (
    OfferCalculationInput,
    OfferCalculationOutput,
    OfferPlan,
)


# 所有金额来自项目演示规则，固定使用 Decimal 计算，不能由大模型自由编造。
DEMO_POLICY_DISCLAIMER = (
    "本方案仅为项目演示规则，不是真实运营商优惠或审批结果；"
    "执行前必须由人工核验资格、成本和授权。"
)
MONEY_STEP = Decimal("0.01")


@dataclass(frozen=True)
class _OfferRule:
    """一条内部优惠规则；数据类仅保存规则参数，不执行计算。"""
    code: str
    name: str
    rate: Decimal
    months: int
    monthly_cap: Decimal
    rationale: str


class OfferCalculationTool:
    """选择一条不可叠加规则，计算月优惠与总优惠并返回方案。"""

    def run(self, request: OfferCalculationInput) -> OfferCalculationOutput:
        """根据校验后的输入计算金额；输出始终附带演示政策和人工审批声明。"""
        # 先选规则，再按月费比例和封顶金额计算；全过程是确定性的 Python 逻辑。
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
    """按风险优先、再按合同与月费选择唯一的演示规则。"""
    # 规则只依赖风险等级、合同类型和月消费，优先保证不同输入得到可复现的同一结果。
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
