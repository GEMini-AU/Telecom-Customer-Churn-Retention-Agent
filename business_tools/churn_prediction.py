# ============================================================================
# 文件职责：加载现有 churn_pipeline.joblib，计算流失概率、预测标签、风险等级和风险信息。
# 主要调用方：agent/registry.py，也可由 scripts/demo_business_tools.py 单独调用。
# 输入/输出：输入已校验的 ChurnFeatures；输出 ChurnPredictionOutput。
# 不负责：不训练模型、不修改模型文件、不让大模型决定概率或风险等级。
# ============================================================================
"""流失预测工具。

调用链：``CustomerLookupTool`` 返回的 ``ChurnFeatures`` -> 本工具 -> Agent 或页面。
本模块只加载 ``churn_pipeline.joblib`` 并调用 ``predict_proba``，不训练、不保存、
也不改写原有逻辑回归 Pipeline。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .errors import ToolError
from .schemas import (
    ChurnPrediction,
    ChurnPredictionInput,
    ChurnPredictionOutput,
    RiskLevel,
)


# 预测类别、风险等级和模型结构均沿用已有项目约定；这里绝不重新训练模型。
CLASSIFICATION_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.60
EXPECTED_PIPELINE_STEPS = {"preprocessor", "classifier"}


class ChurnPredictionTool:
    """用已训练 Pipeline 计算单个客户的流失概率和风险等级。"""

    def __init__(
        self,
        model_path: Path,
        pipeline: Any | None = None,
    ) -> None:
        """保存模型位置或测试替身；真实模型在首次 ``run`` 时才加载。"""
        # 可注入 pipeline 仅用于离线测试；正常运行时按路径懒加载 joblib 模型。
        self.model_path = model_path.resolve()
        self._pipeline = pipeline

    def run(self, request: ChurnPredictionInput) -> ChurnPredictionOutput:
        """把校验后的特征转为单行表格并返回预测结果或结构化错误。"""
        # 模型加载失败与预测失败分开返回，便于页面和测试定位问题。
        try:
            pipeline = self._get_pipeline()
        except Exception as error:
            return ChurnPredictionOutput(
                success=False,
                error=ToolError(
                    code="MODEL_LOAD_ERROR",
                    message=(
                        "流失模型加载失败，请确认模型文件可信且 "
                        "scikit-learn 版本与 requirements.txt 一致："
                        f"{type(error).__name__}：{error}"
                    ),
                ),
            )

        try:
            # 使用 CSV 原始列名别名构造单行 DataFrame，保证输入与训练时 Pipeline 对齐。
            model_input = pd.DataFrame(
                [request.features.model_dump(by_alias=True)]
            )
            # ``classes_`` 记录每一列概率的类别顺序，不能凭经验硬编码列下标。
            classes = list(pipeline.classes_)
            # 不假设正类一定排在第二列，按 classes_ 找到“流失=1”的概率位置。
            positive_index = classes.index(1)
            probability = float(
                pipeline.predict_proba(model_input)[0][positive_index]
            )
            predicted_class = int(probability >= CLASSIFICATION_THRESHOLD)
            prediction = ChurnPrediction(
                churn_probability=probability,
                classification_threshold=CLASSIFICATION_THRESHOLD,
                predicted_class=predicted_class,
                prediction_label=(
                    "预计流失" if predicted_class == 1 else "预计不流失"
                ),
                risk_level=_get_risk_level(probability),
                risk_factors=_extract_risk_factors(request.features),
            )
        except Exception as error:
            return ChurnPredictionOutput(
                success=False,
                error=ToolError(
                    code="PREDICTION_ERROR",
                    message=f"流失预测失败：{type(error).__name__}：{error}",
                ),
            )

        return ChurnPredictionOutput(success=True, prediction=prediction)

    def _get_pipeline(self) -> Any:
        """惰性加载并验证 Pipeline 的两个必要步骤和概率接口。"""
        # joblib 基于 pickle，只加载项目自身生成且受信任的模型文件。
        if self._pipeline is None:
            # joblib 基于 pickle，只加载项目自身生成且受信任的模型文件。
            self._pipeline = joblib.load(self.model_path)
        named_steps = getattr(self._pipeline, "named_steps", {})
        if not EXPECTED_PIPELINE_STEPS.issubset(named_steps):
            raise ValueError("模型不是预期的预处理与分类 Pipeline。")
        if not callable(getattr(self._pipeline, "predict_proba", None)):
            raise ValueError("模型不支持 predict_proba。")
        return self._pipeline


def _get_risk_level(probability: float) -> RiskLevel:
    """按项目阈值将 0~1 概率映射为低、中、高风险。"""
    # 低/中/高风险与分类阈值分开：中风险已预测流失，但尚未达到高风险阈值。
    if probability < CLASSIFICATION_THRESHOLD:
        return "低风险"
    if probability < HIGH_RISK_THRESHOLD:
        return "中风险"
    return "高风险"


def _extract_risk_factors(features: Any) -> list[str]:
    """Return the five existing EDA-based observations used by app.py."""

    # 这些是项目已有 EDA 观察，不是模型系数，也不表示单一客户的因果解释。
    factors: list[str] = []
    if features.contract == "Month-to-month":
        factors.append(
            "合同类型为月付合同，该群体在项目数据中的流失率相对较高"
        )
    if features.tech_support == "No":
        factors.append("客户已使用互联网服务，但未开通技术支持服务")
    if features.payment_method == "Electronic check":
        factors.append(
            "付款方式为电子支票，该群体在项目数据中的流失率相对较高"
        )
    if features.tenure <= 12:
        factors.append(
            f"客户使用时长为 {features.tenure} 个月，仍处于项目定义的新客户阶段"
        )
    if features.monthly_charges >= 75:
        factors.append(
            f"客户月费为 {features.monthly_charges:.2f}，"
            "达到项目定义的较高月费标准"
        )
    return factors
