# ============================================================================
# 文件职责：验证四个业务工具的正常路径、关键边界和结构化错误路径。
# 主要调用方：``python -m unittest`` 自动发现或手工指定本测试模块。
# 输入/输出：输入真实项目文件和局部测试替身；输出 unittest 通过/失败结果。
# 不负责：不测试大模型网络请求，也不测试 Streamlit 页面布局。
# ============================================================================
"""四个业务工具的正常、边界与错误离线测试。

真实 CSV、模型和本地 RAG 会被调用；概率 Pipeline 与 RAG 异常采用小型替身，
以便精确覆盖阈值和错误分支。
"""

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
    """只返回给定正类概率的极简 Pipeline 替身，用于风险边界测试。"""
    classes_ = np.array([0, 1])
    named_steps = {"preprocessor": object(), "classifier": object()}

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, _: object) -> np.ndarray:
        """返回二分类概率矩阵，列顺序与 ``classes_=[0,1]`` 对齐。"""
        return np.array([[1 - self.probability, self.probability]])


class _BrokenRAG:
    """检索时始终失败的替身，用于验证工具将异常转换为错误码。"""
    def retrieve(self, *_: object, **__: object) -> list[object]:
        """模拟 RAG 底层异常，供工具错误包装测试调用。"""
        raise RuntimeError("模拟检索器故障")


class CustomerLookupToolTests(unittest.TestCase):
    """验证客户查询的成功、首尾空格和不存在客户三类路径。"""
    def setUp(self) -> None:
        """每个查询用例创建独立工具实例，避免缓存影响断言。"""
        self.tool = CustomerLookupTool(
            PROJECT_ROOT / "telco_customer_churn.csv"
        )

    def test_normal_existing_customer(self) -> None:
        """存在客户应返回 19 个特征，且不泄露训练标签。"""
        result = self.tool.run(CustomerLookupInput(customer_id=CUSTOMER_ID))
        self.assertTrue(result.success)
        self.assertIsNotNone(result.customer)
        assert result.customer is not None
        self.assertEqual(result.customer.customer_id, CUSTOMER_ID)
        self.assertEqual(len(type(result.customer.features).model_fields), 19)
        self.assertNotIn("Churn", result.model_dump_json(by_alias=True))

    def test_boundary_customer_id_whitespace_is_trimmed(self) -> None:
        """客户 ID 两端空格应由输入模型清理后仍能命中。"""
        result = self.tool.run(
            CustomerLookupInput(customer_id=f"  {CUSTOMER_ID}  ")
        )
        self.assertTrue(result.success)
        assert result.customer is not None
        self.assertEqual(result.customer.customer_id, CUSTOMER_ID)

    def test_error_customer_not_found(self) -> None:
        """不存在 ID 应返回结构化错误而不是抛出查询异常。"""
        result = self.tool.run(
            CustomerLookupInput(customer_id="NOT-EXISTS")
        )
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "CUSTOMER_NOT_FOUND")
        self.assertIn("未找到", result.error.message)


class ChurnPredictionToolTests(unittest.TestCase):
    """验证真实模型预测、两个风险边界和模型文件缺失错误。"""
    @classmethod
    def setUpClass(cls) -> None:
        """查询一个真实客户并准备可被各预测用例复用的合法输入。"""
        lookup = CustomerLookupTool(
            PROJECT_ROOT / "telco_customer_churn.csv"
        ).run(CustomerLookupInput(customer_id=CUSTOMER_ID))
        assert lookup.customer is not None
        cls.request = ChurnPredictionInput(features=lookup.customer.features)

    def test_normal_existing_pipeline_prediction(self) -> None:
        """真实 joblib Pipeline 应输出范围合法的概率和项目分类阈值。"""
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
        """概率恰为 0.40/0.60 时应进入中风险/高风险的包含边界。"""
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
        """模型路径不存在时工具应返回 ``MODEL_LOAD_ERROR``。"""
        tool = ChurnPredictionTool(PROJECT_ROOT / "missing-model.joblib")
        result = tool.run(self.request)
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "MODEL_LOAD_ERROR")


class KnowledgeRetrievalToolTests(unittest.TestCase):
    """验证来源保留、相似度门槛拒答和 RAG 异常包装。"""
    def setUp(self) -> None:
        """为每条检索用例创建绑定正式知识库的工具实例。"""
        self.tool = KnowledgeRetrievalTool(PROJECT_ROOT)

    def test_normal_returns_fragments_and_sources(self) -> None:
        """相关问题应返回至少一条含文件名、标题和正文的证据。"""
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
        """最高门槛 1.0 下通常没有证据，工具应明确返回空列表。"""
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
        """底层 RAG 抛异常时工具应返回统一知识检索错误码。"""
        tool = KnowledgeRetrievalTool(
            PROJECT_ROOT,
            rag_service=_BrokenRAG(),
        )
        result = tool.run(KnowledgeSearchInput(question="任意问题"))
        self.assertFalse(result.success)
        assert result.error is not None
        self.assertEqual(result.error.code, "KNOWLEDGE_RETRIEVAL_ERROR")


class OfferCalculationToolTests(unittest.TestCase):
    """验证封顶金额、75 月费边界和负金额输入校验。"""
    def setUp(self) -> None:
        """每条规则用例使用无状态的优惠计算工具。"""
        self.tool = OfferCalculationTool()

    def test_normal_high_risk_monthly_offer_is_capped(self) -> None:
        """高风险高月费月付客户应触发封顶后的固定金额。"""
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
        """月费恰为 75.00 应进入高用量规则分支。"""
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
        """负数月费在构造输入模型时就应被 Pydantic 拒绝。"""
        with self.assertRaises(ValidationError):
            OfferCalculationInput(
                risk_level="高风险",
                contract="Month-to-month",
                monthly_charges=Decimal("-0.01"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
