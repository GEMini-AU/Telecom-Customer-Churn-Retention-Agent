# ============================================================================
# 文件职责：从命令行运行真实单 Agent，便于脱离 Streamlit 验证完整方案生成。
# 主要调用方：用户手工运行 ``python scripts/run_agent.py "含客户ID的任务"``。
# 输入/输出：输入自然语言任务和工具上限；输出 AgentRunResult JSON 和退出码。
# 不负责：不展示页面、不做人机确认；需已配置 DEEPSEEK_API_KEY 才可真实调用。
# ============================================================================
"""从命令行运行真实单 Agent。

运行：``python scripts/run_agent.py "请分析客户 7590-VHVEG"``。
此脚本需要环境变量 API Key；与离线测试脚本不同，它会发起真实大模型请求。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.single_agent import SingleRetentionAgent
from llm.deepseek_client import DeepSeekConfigurationError


def main() -> None:
    """读取任务和上限，创建 Agent，打印结构化结果；失败时用退出码 1 表示。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="包含一个 customerID 的自然语言运营任务")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    args = parser.parse_args()

    try:
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            max_tool_calls=args.max_tool_calls,
        )
    except DeepSeekConfigurationError as error:
        raise SystemExit(f"Agent 配置失败：{error}") from error

    result = agent.run(args.task)
    print(result.model_dump_json(indent=2))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
