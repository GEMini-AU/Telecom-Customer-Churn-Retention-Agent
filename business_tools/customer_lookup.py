"""Customer lookup tool backed by the project's existing CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import ToolError
from .schemas import (
    ChurnFeatures,
    CustomerLookupInput,
    CustomerLookupOutput,
    CustomerProfile,
)


class CustomerLookupTool:
    """Find one customer and expose only identifier plus model features."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path.resolve()
        self._data: pd.DataFrame | None = None

    def run(self, request: CustomerLookupInput) -> CustomerLookupOutput:
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

        return CustomerLookupOutput(
            success=True,
            customer=CustomerProfile(
                customer_id=request.customer_id,
                features=features,
            ),
        )

    def _load_data(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        data = pd.read_csv(self.data_path)
        required_columns = {
            "customerID",
            *(
                field_info.alias or field_name
                for field_name, field_info
                in ChurnFeatures.model_fields.items()
            ),
        }
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(
                f"客户数据缺少字段：{sorted(missing_columns)}"
            )

        data = data.copy()
        data["customerID"] = data["customerID"].astype(str).str.strip()
        total_charges = data["TotalCharges"].astype(str).str.strip()
        data.loc[total_charges.eq(""), "TotalCharges"] = 0
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"])
        self._data = data
        return data
