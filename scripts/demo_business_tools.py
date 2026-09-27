# ============================================================================
# 文件职责：按真实业务顺序直接演示客户查询、预测、知识检索和优惠计算四个工具。
# 主要调用方：用户手工运行 ``python scripts/demo_business_tools.py``。
# 输入/输出：使用固定演示客户和问题；输出每个工具的 JSON 结果到控制台。
# 不负责：不调用 Agent、不调用大模型、不写入人工确认数据库。
# ============================================================================
"""不经过 Agent 或大模型，顺序直调四个业务工具的命令行演示。

运行：``python scripts/demo_business_tools.py``。用于验证真实 CSV、模型、RAG 和
优惠规则各自可运行，且可观察每个工具的结构化输出。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business_tools import (
    ChurnPredictionInput,
    ChurnPredictionTool,
    CustomerLookupInput,
    CustomerLookupTool,
    KnowledgeRetrievalTool,
    KnowledgeSearchInput,
    OfferCalculationInput,
    OfferCalculationTool,
)


def main() -> None:
    """查询固定演示客户，预测、检索、计算优惠，并逐段打印结果。"""
    # 后续预测和优惠均依赖此次查询成功返回的经过校验的特征。
    lookup = CustomerLookupTool(
        PROJECT_ROOT / "telco_customer_churn.csv"
    ).run(CustomerLookupInput(customer_id="7590-VHVEG"))
    print("\n[客户查询工具]")
    print(lookup.model_dump_json(indent=2))
    if not lookup.success or lookup.customer is None:
        raise SystemExit("客户查询失败，停止后续直接调用。")

    prediction = ChurnPredictionTool(
        PROJECT_ROOT / "churn_pipeline.joblib"
    ).run(ChurnPredictionInput(features=lookup.customer.features))
    print("\n[流失预测工具]")
    print(prediction.model_dump_json(indent=2))
    if not prediction.success or prediction.prediction is None:
        raise SystemExit("流失预测失败，停止优惠计算。")

    # 政策检索与该客户无关；它只返回演示知识库的来源片段。
    retrieval = KnowledgeRetrievalTool(PROJECT_ROOT).run(
        KnowledgeSearchInput(question="高风险客户如何挽留？", top_k=2)
    )
    print("\n[知识检索工具]")
    print(retrieval.model_dump_json(indent=2))

    offer = OfferCalculationTool().run(
        OfferCalculationInput(
            risk_level=prediction.prediction.risk_level,
            contract=lookup.customer.features.contract,
            monthly_charges=lookup.customer.features.monthly_charges,
        )
    )
    print("\n[优惠方案计算工具]")
    print(offer.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
