# ============================================================================
# 文件职责：提供一个统一 Streamlit 入口，在原预测页面和 Agent 页面之间导航。
# 主要调用方：``streamlit run main_app.py``。
# 输入/输出：输入侧栏页面选择；输出选中页面的原有 Streamlit 内容。
# 不负责：不复制业务逻辑、不训练模型、不改变 app.py 或 agent_app.py 的独立运行方式。
# ============================================================================
"""统一演示入口；两个页面仍可各自用 streamlit run 单独启动。"""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """注册两个原有脚本为页面，再运行当前选中的脚本。"""
    # st.Page 只包装脚本入口；app.py 与 agent_app.py 的业务代码无需复制。
    pages = [
        st.Page("app.py", title="原用户流失预测", icon="📊"),
        st.Page("agent_app.py", title="智能运营 Agent", icon="🧭"),
    ]
    # 单一导航入口降低演示切页成本，不把两个独立页面强行合成一个文件。
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
