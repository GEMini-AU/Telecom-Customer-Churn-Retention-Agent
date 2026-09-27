# ============================================================================
# 文件职责：按 customerID 从项目 CSV 查询唯一客户，并提取预测模型需要的 19 个字段。
# 主要调用方：agent/registry.py 的 get_customer_profile 工具适配器。
# 输入/输出：输入 CustomerLookupInput；输出 CustomerLookupOutput 或结构化错误。
# 不负责：不预测流失、不读取知识库，也不把完整 CSV 行交给模型。
# ============================================================================
"""客户查询工具。

调用链：``agent.registry.BusinessToolRegistry`` -> ``CustomerLookupTool.run``。
输入为 ``customerID``；输出为经过 Pydantic 校验的客户标识和 19 个模型特征，
绝不把完整 CSV 行或无关个人字段交给后续模型。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import ToolError
from .customer_store import CustomCustomerStore
from .schemas import (
    ChurnFeatures,
    CustomerLookupInput,
    CustomerLookupOutput,
    CustomerProfile,
)


class CustomerLookupTool:
    """从项目 CSV 找到唯一客户，并只暴露预测所需特征。

    ``run`` 被工具注册表调用；``_load_data`` 只在首次查询时读取和清洗 CSV。
    """

    def __init__(
        self,
        data_path: Path,
        custom_store: CustomCustomerStore | None = None,
    ) -> None:
        """保存 CSV 路径并初始化进程内缓存。

        参数 ``data_path`` 是数据文件位置；这里不读取文件，避免页面启动就失败。
        """
        # 只保存路径和空缓存；CSV 在第一次真正查询时才读取。
        self.data_path = data_path.resolve()
        # 新增客户放在与 CSV 分离的本地数据库；测试可注入临时数据库。
        self.custom_store = custom_store or CustomCustomerStore(
            self.data_path.parent / "storage" / "customers" / "demo_customers.sqlite3"
        )
        self._data: pd.DataFrame | None = None

    def run(self, request: CustomerLookupInput) -> CustomerLookupOutput:
        """按 ``request.customer_id`` 精确查询，并返回统一成功/失败结构。"""
        # 先查自定义客户，但数据库损坏不能被误报成“客户不存在”。
        try:
            custom = self.custom_store.get(request.customer_id)
        except Exception as error:
            return CustomerLookupOutput(
                success=False,
                error=ToolError(
                    code="CUSTOMER_DATA_ERROR",
                    message=f"自定义客户库读取失败：{type(error).__name__}：{error}",
                ),
            )
        if custom is not None:
            return CustomerLookupOutput(success=True, customer=custom)

        # 所有失败都转换为结构化结果，调用方无需依赖异常文本判断业务状态。
        try:
            data = self._load_data()
        except Exception as error:
            return CustomerLookupOutput(
                success=False,
                error=ToolError(
                    code="CUSTOMER_DATA_ERROR",
                    message=(
                        f"客户数据读取失败：{type(error).__name__}：{error}"
                    ),
                ),
            )

        # 仅允许按标准 customerID 精确匹配，不能模糊匹配到其他客户。
        matches = data.loc[data["customerID"] == request.customer_id]
        if matches.empty:
            return CustomerLookupOutput(
                success=False,
                error=ToolError(
                    code="CUSTOMER_NOT_FOUND",
                    message=f"未找到 customerID={request.customer_id} 的客户。",
                ),
            )
        if len(matches) > 1:
            return CustomerLookupOutput(
                success=False,
                error=ToolError(
                    code="DUPLICATE_CUSTOMER_ID",
                    message=f"customerID={request.customer_id} 存在重复记录。",
                ),
            )

        # 从完整 CSV 行中只提取 Pipeline 声明的 19 个特征，并做类型与取值校验。
        try:
            row = matches.iloc[0]
            features = ChurnFeatures.model_validate(
                {
                    field_name: row[field_info.alias or field_name]
                    for field_name, field_info
                    in ChurnFeatures.model_fields.items()
                }
            )
        except Exception as error:
            return CustomerLookupOutput(
                success=False,
                error=ToolError(
                    code="CUSTOMER_DATA_INVALID",
                    message=(
                        f"客户业务字段校验失败：{type(error).__name__}：{error}"
                    ),
                ),
            )

        # 成功结果不暴露完整原始行，只返回客户标识和模型所需特征。
        return CustomerLookupOutput(
            success=True,
            customer=CustomerProfile(
                customer_id=request.customer_id,
                features=features,
            ),
        )

    def _load_data(self) -> pd.DataFrame:
        """读取、校验并清洗 CSV；成功后缓存并返回 DataFrame。"""
        # 同一进程内复用已经完成字段检查和清洗的数据，避免重复磁盘读取。
        if self._data is not None:
            return self._data
        # pandas 将磁盘 CSV 解析为表格；后续所有筛选都针对这个 DataFrame。
        data = pd.read_csv(self.data_path)
        # 必需列由 ChurnFeatures 动态生成，数据字段变更时不会悄悄与模型输入脱节。
        required_columns = {
            "customerID",
            *(
                field_info.alias or field_name
                for field_name, field_info
                in ChurnFeatures.model_fields.items()
            ),
        }
        # 集合相减得到“模型需要但 CSV 没有”的字段；为空才允许继续。
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"客户数据缺少字段：{sorted(missing_columns)}"
            )

        # 不直接改 pandas 原始对象：统一清理 ID，并把空白 TotalCharges 视为 0 后转数值。
        data = data.copy()
        data["customerID"] = data["customerID"].astype(str).str.strip()
        total_charges = data["TotalCharges"].astype(str).str.strip()
        data.loc[total_charges.eq(""), "TotalCharges"] = 0
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])
        self._data = data
        return data
