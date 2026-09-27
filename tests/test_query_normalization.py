# ============================================================================
# 文件职责：验证 RAG 查询扩展只作用于明确的电信服务问题。
# 主要调用方：``python -m unittest discover -s tests``。
# 输入/输出：输入口语和无关问题；输出确定性的扩展/不扩展断言。
# 不负责：不调用在线模型，也不修改知识文档。
# ============================================================================
"""口语查询扩展的边界测试。"""

import unittest

from rag.query_normalization import expand_telecom_query


class QueryNormalizationTests(unittest.TestCase):
    """防止口语改写把无关的股票问题伪装成电信政策问题。"""

    def test_fiber_quality_question_is_expanded(self) -> None:
        """同时包含光纤、网络速度和故障描述时添加政策用词。"""
        question = "光纤客户反映网速不稳定时应先核查什么？"
        expanded = expand_telecom_query(question)
        self.assertTrue(expanded.startswith(question))
        self.assertIn("光纤网络服务", expanded)

    def test_generic_network_complaint_is_expanded(self) -> None:
        """网络卡顿映射为投诉主题，但不生成处理步骤。"""
        self.assertIn(
            "网络质量投诉",
            expand_telecom_query("网络很卡，要先排查哪些？"),
        )

    def test_stock_question_is_not_expanded(self) -> None:
        """单独出现光纤词不能把股票询问变为电信政策证据。"""
        question = "光纤股票明天价格是多少？"
        self.assertEqual(expand_telecom_query(question), question)


if __name__ == "__main__":
    unittest.main(verbosity=2)
