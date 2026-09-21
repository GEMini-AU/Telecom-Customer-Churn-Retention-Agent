"""Churn prediction tool using the existing trained Pipeline as-is."""

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


CLASSIFICATION_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.60
EXPECTED_PIPELINE_STEPS = {"preprocessor", "classifier"}


class ChurnPredictionTool:
    """Run predict_proba without fitting or modifying the saved model."""

    def __init__(
        self,
        model_path: Path,
        pipeline: Any | None = None,
    ) -> None:
        self.model_path = model_path.resolve()
        self._pipeline = pipeline

    def run(self, request: ChurnPredictionInput) -> ChurnPredictionOutput:
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
            model_input = pd.DataFrame(
                [request.features.model_dump(by_alias=True)]
            )
            classes = list(pipeline.classes_)
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
    if probability < CLASSIFICATION_THRESHOLD:
        return "低风险"
    if probability < HIGH_RISK_THRESHOLD:
        return "中风险"
    return "高风险"


def _extract_risk_factors(features: Any) -> list[str]:
    """Return the five existing EDA-based observations used by app.py."""

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
