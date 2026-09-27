# ============================================================================
# 文件职责：按训练 Notebook 的原始参数复现训练/测试客户 ID 划分。
# 主要调用方：agent_app.py 的来源标记及离线评估脚本。
# 输入/输出：输入原始公开 CSV 路径与客户 ID；输出训练集/测试集/自定义来源。
# 不负责：不重新训练模型，也不声称训练集客户属于独立泛化评估。
# ============================================================================
"""演示数据来源标记。若 Notebook 的划分参数改变，这里必须同步更新。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_customer_ids(data_path: Path) -> tuple[set[str], set[str]]:
    """复现 02_ml_pipeline.ipynb 的 80/20 分层随机划分，不调用 fit。"""
    # Streamlit 每次交互都会重跑页面：文件未变化时复用划分；变化时按时间和大小失效。
    resolved = data_path.resolve()
    file_stat = resolved.stat()
    return _split_customer_ids_cached(
        resolved, file_stat.st_mtime_ns, file_stat.st_size
    )


@lru_cache(maxsize=4)
def _split_customer_ids_cached(
    data_path: Path, modified_ns: int, file_size: int
) -> tuple[set[str], set[str]]:
    """缓存一次划分；额外参数仅用于文件变化时让缓存键失效。"""
    del modified_ns, file_size
    # 只读取划分所需两列；保持 CSV 原始行序，与 Notebook 的 X/y 行序一致。
    data = pd.read_csv(data_path, usecols=["customerID", "Churn"])
    labels = data["Churn"].map({"Yes": 1, "No": 0})
    if labels.isna().any() or data["customerID"].duplicated().any():
        raise ValueError("无法复现划分：流失标签无效或客户 ID 重复。")
    train_index, test_index = train_test_split(
        data.index, test_size=0.2, random_state=42, stratify=labels
    )
    train_ids = set(data.loc[train_index, "customerID"].astype(str))
    test_ids = set(data.loc[test_index, "customerID"].astype(str))
    return train_ids, test_ids


def customer_source(customer_id: str, data_path: Path) -> str:
    """给一个 ID 标记来源；不在公开 CSV 中的客户仅标为“自定义/未知”。"""
    train_ids, test_ids = split_customer_ids(data_path)
    if customer_id in test_ids:
        return "独立测试集样例"
    if customer_id in train_ids:
        return "训练集样例（仅演示流程）"
    return "自定义客户或未知 ID"
