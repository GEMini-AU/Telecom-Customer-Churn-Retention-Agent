# ============================================================================
# 文件职责：从命令行独立检索演示知识库，并打印回答和每个命中的来源。
# 主要调用方：用户手工运行 ``python scripts/query_rag.py "问题"``。
# 输入/输出：输入问题、top_k、分数门槛；输出带 RAG 状态和来源的控制台文本。
# 不负责：不读取客户资料、不计算流失风险，也不执行 Agent 工具编排。
# ============================================================================
"""命令行查询独立演示 RAG。

默认用离线摘录式回答；``--use-deepseek`` 才会调用模型润色，且仍只能使用检索证据。
运行：``python scripts/query_rag.py "高风险客户如何挽留？"``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.deepseek_client import DeepSeekConfigurationError
from rag.generation import DeepSeekGroundedAnswerGenerator
from rag.service import DemoTelecomRAG


def main() -> None:
    """解析问题与检索参数，构建索引、回答问题并逐条打印来源。"""
    # argparse 把终端文本转换为可校验的命令行参数，不涉及 Streamlit 页面。
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="要检索的电信知识问题")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.10)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--use-deepseek",
        action="store_true",
        help="使用 DeepSeek 根据检索片段生成答案；需要 DEEPSEEK_API_KEY。",
    )
    args = parser.parse_args()

    rag = DemoTelecomRAG(PROJECT_ROOT)
    rag.ensure_index(force_rebuild=args.rebuild)
    # 未显式选择 DeepSeek 时保持 None，服务层会使用确定性的本地摘录生成器。
    generator = None
    if args.use_deepseek:
        try:
            generator = DeepSeekGroundedAnswerGenerator()
        except DeepSeekConfigurationError as error:
            print(f"DeepSeek 配置不可用，已降级为本地证据式回答：{error}")
    answer = rag.answer_question(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        generator=generator,
    )

    print(f"状态：{answer.status}")
    if answer.error_type:
        print(f"异常类型：{answer.error_type}")
    print(answer.answer)
    if answer.evidence:
        print("\n检索来源：")
        for item in answer.evidence:
            print(
                f"- {item.source_file} | {item.document_title} | "
                f"{item.section_title} | score={item.score:.4f} | "
                f"md5={item.document_md5}"
            )
            print(f"  片段：{' '.join(item.text.split())}")


if __name__ == "__main__":
    main()
