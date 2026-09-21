"""Run the single DeepSeek Agent from the command line."""

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
