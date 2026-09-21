"""Directly call all business tools without an Agent or language model."""

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
