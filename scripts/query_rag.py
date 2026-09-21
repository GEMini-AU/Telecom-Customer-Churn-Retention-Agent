"""Query the independent demo telecom RAG from the command line."""

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
