"""Normal, boundary, and error tests for all four business tools."""

from __future__ import annotations

import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np
from pydantic import ValidationError

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_ID = "7590-VHVEG"


class _ProbabilityPipeline:
    classes_ = np.array([0, 1])
    named_steps = {"preprocessor": object(), "classifier": object()}

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _: object) -> np.ndarray:
        return np.array([[1 - self.probability, self.probability]])


class _BrokenRAG:
    def retrieve(self, *_: object, **__: object) -> list[object]:
        raise RuntimeError("模拟检索器故障")


class CustomerLookupToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = CustomerLookupTool(
            PROJECT_ROOT / "telco_customer_churn.csv"
        )

    def test_normal_existing_customer(self) -> None:
        result = self.tool.run(CustomerLookupInput(customer_id=CUSTOMER_ID))
        self.assertTrue(result.success)
        self.assertIsNotNone(result.customer)
        assert result.customer is not None
        self.assertEqual(result.customer.customer_id, CUSTOMER_ID)
        self.assertEqual(len(type(result.customer.features).model_fields), 19)
        self.assertNotIn("Churn", result.model_dump_json(by_alias=True))

    def test_boundary_customer_id_whitespace_is_trimmed(self) -> None:
        result = self.tool.run(
            CustomerLookupInput(customer_id=f"  {CUSTOMER_ID}  ")
        )
        self.assertTrue(result.success)
        assert result.customer is not None
        self.assertEqual(result.customer.customer_id, CUSTOMER_ID)

    def test_error_customer_not_found(self) -> None:
        result = self.tool.run(
            CustomerLookupInput(customer_id="NOT-EXISTS")
        )
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "CUSTOMER_NOT_FOUND")
        self.assertIn("未找到", result.error.message)


class ChurnPredictionToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lookup = CustomerLookupTool(
            PROJECT_ROOT / "telco_customer_churn.csv"
        ).run(CustomerLookupInput(customer_id=CUSTOMER_ID))
        assert lookup.customer is not None
        cls.request = ChurnPredictionInput(features=lookup.customer.features)

    def test_normal_existing_pipeline_prediction(self) -> None:
        tool = ChurnPredictionTool(PROJECT_ROOT / "churn_pipeline.joblib")
        result = tool.run(self.request)
        self.assertTrue(result.success, result.error)
        assert result.prediction is not None
        self.assertGreaterEqual(result.prediction.churn_probability, 0)
        self.assertLessEqual(result.prediction.churn_probability, 1)
        self.assertEqual(result.prediction.classification_threshold, 0.40)
        expected_class = int(result.prediction.churn_probability >= 0.40)
        self.assertEqual(result.prediction.predicted_class, expected_class)

    def test_boundary_risk_thresholds_are_inclusive(self) -> None:
        medium = ChurnPredictionTool(
            PROJECT_ROOT / "unused.joblib",
            pipeline=_ProbabilityPipeline(0.40),
        ).run(self.request)
        high = ChurnPredictionTool(
            PROJECT_ROOT / "unused.joblib",
            pipeline=_ProbabilityPipeline(0.60),
        ).run(self.request)
        assert medium.prediction is not None
        assert high.prediction is not None
        self.assertEqual(medium.prediction.predicted_class, 1)
        self.assertEqual(medium.prediction.risk_level, "中风险")
        self.assertEqual(high.prediction.risk_level, "高风险")

    def test_error_missing_model_returns_structured_error(self) -> None:
        tool = ChurnPredictionTool(PROJECT_ROOT / "missing-model.joblib")
        result = tool.run(self.request)
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "MODEL_LOAD_ERROR")


class KnowledgeRetrievalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = KnowledgeRetrievalTool(PROJECT_ROOT)

    def test_normal_returns_fragments_and_sources(self) -> None:
        result = self.tool.run(
            KnowledgeSearchInput(question="高风险客户如何挽留？")
        )
        self.assertTrue(result.success)
        self.assertTrue(result.has_sufficient_evidence)
        self.assertTrue(result.evidence)
        self.assertTrue(result.evidence[0].source_file)
        self.assertTrue(result.evidence[0].document_title)
        self.assertTrue(result.evidence[0].text)

    def test_boundary_score_one_returns_no_evidence(self) -> None:
        result = self.tool.run(
            KnowledgeSearchInput(
                question="高风险客户如何挽留？",
                min_score=1,
            )
        )
        self.assertTrue(result.success)
        self.assertFalse(result.has_sufficient_evidence)
        self.assertEqual(result.evidence, [])

    def test_error_retriever_failure_returns_structured_error(self) -> None:
        tool = KnowledgeRetrievalTool(
            PROJECT_ROOT,
            rag_service=_BrokenRAG(),
        )
        result = tool.run(KnowledgeSearchInput(question="任意问题"))
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "KNOWLEDGE_RETRIEVAL_ERROR")


class OfferCalculationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = OfferCalculationTool()

    def test_normal_high_risk_monthly_offer_is_capped(self) -> None:
        result = self.tool.run(
            OfferCalculationInput(
                risk_level="高风险",
                contract="Month-to-month",
                monthly_charges=Decimal("100.00"),
            )
        )
        self.assertEqual(result.offer.offer_code, "HIGH_M2M_HIGH_USAGE")
        self.assertEqual(result.offer.monthly_discount, Decimal("15.00"))
        self.assertEqual(result.offer.total_discount, Decimal("45.00"))
        self.assertFalse(result.offer.stackable)
        self.assertTrue(result.offer.requires_manual_approval)

    def test_boundary_monthly_charge_75_uses_high_usage_rule(self) -> None:
        result = self.tool.run(
            OfferCalculationInput(
                risk_level="中风险",
                contract="Month-to-month",
                monthly_charges=Decimal("75.00"),
            )
        )
        self.assertEqual(result.offer.offer_code, "MEDIUM_M2M_HIGH_USAGE")
        self.assertEqual(result.offer.monthly_discount, Decimal("6.00"))

    def test_error_negative_monthly_charge_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            OfferCalculationInput(
                risk_level="高风险",
                contract="Month-to-month",
                monthly_charges=Decimal("-0.01"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
