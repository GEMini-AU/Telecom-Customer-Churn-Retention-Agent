"""Run five offline end-to-end Agent tasks without network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.single_agent import SingleRetentionAgent
from llm.deepseek_client import DeepSeekSettings
from tests.offline_agent_model import OfflineAgentCompletions, build_fake_client


def main() -> None:
    cases = json.loads(
        (PROJECT_ROOT / "evaluation" / "agent_tasks.json").read_text(
            encoding="utf-8"
        )
    )
    for case in cases:
        agent = SingleRetentionAgent(
            PROJECT_ROOT,
            settings=DeepSeekSettings(api_key="offline-agent-test"),
            openai_client=build_fake_client(OfflineAgentCompletions()),
        )
        result = agent.run(case["task"])
        assert result.success, result.error
        assert result.plan is not None
        assert [item.tool_name for item in result.tool_calls] == case[
            "expected_tools"
        ]
        assert result.plan.risk_assessment.risk_level == case[
            "expected_risk_level"
        ]
        print(f"通过：{case['name']}")
    print(f"单 Agent 完整任务：{len(cases)}/{len(cases)} 通过。")


if __name__ == "__main__":
    main()
