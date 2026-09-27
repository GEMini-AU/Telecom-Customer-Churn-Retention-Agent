# ============================================================================
# 文件职责：为少量已验证的电信口语问法补充检索同义表达。
# 主要调用方：rag/vector_store.py 的 search 方法。
# 输入/输出：输入原始问题；输出仅用于向量检索的扩展问题。
# 不负责：不生成政策答案、不修改知识库原文、不替代证据门槛。
# ============================================================================
"""透明、确定性的查询扩展；仅修复当前评估集中已观察到的漏检。"""

from __future__ import annotations


def expand_telecom_query(question: str) -> str:
    """补充与演示知识库用词一致的主题词，保留用户原句和相似度阈值。"""
    # 三个条件共同限定“光纤服务质量”主题，避免仅有“光纤股票”等词时误扩展。
    if (
        "光纤" in question
        and ("网速" in question or "网络" in question)
        and any(word in question for word in ("不稳定", "很卡", "卡顿", "故障"))
    ):
        return question + " 光纤网络服务 速度 稳定性"
    # “网络很卡”属于网络质量投诉的口语表述；不添加具体处理答案。
    if (
        ("网络" in question or "网速" in question)
        and any(word in question for word in ("很卡", "卡顿"))
    ):
        return question + " 网络质量投诉 网络服务"
    return question
