# ============================================================================
# 文件职责：确保最终验收脚本中的全部登记案例持续通过。
# 主要调用方：``python -m unittest`` 自动发现或手工指定本测试模块。
# 输入/输出：输入 run_acceptance 的结果列表；输出 unittest 通过/失败结果。
# 不负责：不新增验收规则；具体案例定义在 evaluation 和 scripts/run_acceptance.py。
# ============================================================================
"""确保最终验收清单持续可执行的回归测试。"""

import unittest

from scripts.run_acceptance import run_acceptance


class FinalAcceptanceTests(unittest.TestCase):
    """运行验收脚本并断言所有登记案例都通过。"""
    def test_all_manifest_cases_pass(self) -> None:
        """验收结果至少十条且失败列表为空。"""
        results = run_acceptance()
        self.assertGreaterEqual(len(results), 10)
        failures = [item for item in results if not item["passed"]]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
