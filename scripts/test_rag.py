"""Offline regression tests for the independent local RAG module."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.deepseek_client import DeepSeekSettings
from rag.generation import DeepSeekGroundedAnswerGenerator
from rag.service import (
    DemoTelecomRAG,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    RETRIEVAL_ERROR_MESSAGE,
)


class _OfflineDeepSeekCompletions:
    def __init__(self, used_chunk_id: str) -> None:
        self.used_chunk_id = used_chunk_id

    def create(self, **_: object) -> SimpleNamespace:
        content = json.dumps(
            {
                "answer": "高风险客户应优先进入人工复核队列，并核验优惠资格。",
                "used_chunk_ids": [self.used_chunk_id],
            },
            ensure_ascii=False,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _BrokenGenerator:
    def generate(self, question: str, evidence: list[object]) -> str:
        del question, evidence
        raise TimeoutError("模拟模型超时")


class _BrokenStore:
    def search(self, *_: object, **__: object) -> list[object]:
        raise RuntimeError("模拟检索异常")


def _demo_document(title: str, body: str) -> str:
    return (
        f"# {title}\n\n"
        "> 本文档内容为项目演示政策，不是真实运营商内部政策。\n\n"
        f"## 规则\n\n{body}\n"
    )


def _test_incremental_updates(temp_root: Path) -> None:
    project_root = temp_root / "incremental_project"
    knowledge_dir = project_root / "knowledge" / "demo"
    knowledge_dir.mkdir(parents=True)
    index_path = project_root / "storage" / "rag" / "test.joblib"
    first_path = knowledge_dir / "a.md"
    duplicate_path = knowledge_dir / "b.md"
    original_text = _demo_document("基础规则", "高风险客户需要人工复核。")
    first_path.write_text(original_text, encoding="utf-8")
    duplicate_path.write_text(original_text, encoding="utf-8")

    rag = DemoTelecomRAG(
        project_root,
        knowledge_dir=knowledge_dir,
        index_path=index_path,
    )
    initial_store = rag.ensure_index(force_rebuild=True)
    first_source = "knowledge/demo/a.md"
    second_source = "knowledge/demo/b.md"
    third_source = "knowledge/demo/c.md"
    assert set(initial_store.document_manifest) == {first_source}
    assert len(initial_store.duplicate_documents) == 1
    assert initial_store.duplicate_documents[0].skipped_source == second_source
    retained_chunk_ids = initial_store.document_manifest[first_source].chunk_ids

    persisted_rag = DemoTelecomRAG(
        project_root,
        knowledge_dir=knowledge_dir,
        index_path=index_path,
    )
    persisted_store = persisted_rag.ensure_index()
    assert persisted_rag.last_update_report is not None
    assert persisted_rag.last_update_report.mode == "reused"
    assert len(persisted_store.duplicate_documents) == 1

    duplicate_path.write_text(
        _demo_document("新增规则", "低风险客户可以使用常规关怀流程。"),
        encoding="utf-8",
    )
    added_store = persisted_rag.ensure_index()
    assert persisted_rag.last_update_report is not None
    assert persisted_rag.last_update_report.mode == "incremental_update"
    assert persisted_rag.last_update_report.added_sources == [second_source]
    assert added_store.document_manifest[first_source].chunk_ids == retained_chunk_ids

    third_path = knowledge_dir / "c.md"
    third_path.write_text(
        _demo_document("临时规则", "投诉客户需要记录问题和处理结果。"),
        encoding="utf-8",
    )
    persisted_rag.ensure_index()
    assert persisted_rag.last_update_report is not None
    assert persisted_rag.last_update_report.added_sources == [third_source]

    duplicate_path.write_text(
        _demo_document("更新规则", "中风险客户应安排服务回访。"),
        encoding="utf-8",
    )
    updated_store = persisted_rag.ensure_index()
    assert persisted_rag.last_update_report is not None
    assert persisted_rag.last_update_report.updated_sources == [second_source]
    assert updated_store.document_manifest[first_source].chunk_ids == retained_chunk_ids

    third_path.unlink()
    deleted_store = persisted_rag.ensure_index()
    assert persisted_rag.last_update_report is not None
    assert persisted_rag.last_update_report.deleted_sources == [third_source]
    assert third_source not in deleted_store.document_manifest

    index_path.write_bytes(b"not-a-valid-joblib-index")
    recovered_rag = DemoTelecomRAG(
        project_root,
        knowledge_dir=knowledge_dir,
        index_path=index_path,
    )
    recovered_store = recovered_rag.ensure_index()
    assert recovered_rag.last_update_report is not None
    assert recovered_rag.last_update_report.mode == "full_rebuild"
    assert recovered_rag.last_update_report.reason is not None
    assert recovered_rag.last_update_report.reason.startswith("index_load_failed:")
    assert recovered_store.chunks


def main() -> None:
    cases_path = PROJECT_ROOT / "evaluation" / "rag_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="demo-rag-test-") as temp_dir:
        temp_root = Path(temp_dir)
        index_path = temp_root / "demo_telecom.joblib"
        rag = DemoTelecomRAG(PROJECT_ROOT, index_path=index_path)
        store = rag.ensure_index(force_rebuild=True)

        assert index_path.exists(), "持久化向量索引未生成"
        assert store.chunks, "知识库切分结果为空"
        assert store.document_manifest, "文档清单未写入索引"
        assert all(not item.source_file.endswith(".csv") for item in store.chunks)
        assert all("customerID" not in item.text for item in store.chunks)
        assert all(len(item.document_md5) == 32 for item in store.chunks)

        # 重新实例化并从磁盘加载，确认向量、清单和元数据均可复用。
        reloaded_rag = DemoTelecomRAG(PROJECT_ROOT, index_path=index_path)
        reloaded_store = reloaded_rag.ensure_index()
        assert len(reloaded_store.chunks) == len(store.chunks)
        assert reloaded_rag.last_update_report is not None
        assert reloaded_rag.last_update_report.mode == "reused"

        deepseek_evidence = reloaded_rag.retrieve(
            "高风险客户应该如何挽留？", top_k=1
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=_OfflineDeepSeekCompletions(deepseek_evidence[0].chunk_id)
            )
        )
        deepseek_generator = DeepSeekGroundedAnswerGenerator(
            settings=DeepSeekSettings(api_key="offline-test-key"),
            openai_client=fake_client,
        )
        generated = reloaded_rag.answer_question(
            "高风险客户应该如何挽留？",
            top_k=1,
            generator=deepseek_generator,
        )
        assert generated.status == "answered"
        assert "优先进入人工复核队列" in generated.answer

        fallback = reloaded_rag.answer_question(
            "高风险客户应该如何挽留？",
            top_k=1,
            generator=_BrokenGenerator(),
        )
        assert fallback.has_sufficient_evidence
        assert fallback.status == "generation_fallback"
        assert fallback.error_type == "TimeoutError"
        assert fallback.evidence

        broken_rag = DemoTelecomRAG(PROJECT_ROOT, index_path=index_path)
        broken_rag._store = _BrokenStore()  # type: ignore[assignment]
        retrieval_failure = broken_rag.answer_question("任意问题")
        assert retrieval_failure.status == "retrieval_error"
        assert retrieval_failure.answer == RETRIEVAL_ERROR_MESSAGE
        assert retrieval_failure.error_type == "RuntimeError"

        passed = 0
        for case in cases:
            question = case["question"]
            expected_sources = case["expected_sources"]
            answer = reloaded_rag.answer_question(
                question,
                top_k=3,
                min_score=0.10,
            )

            if expected_sources:
                actual_sources = {item.source_file for item in answer.evidence}
                assert answer.has_sufficient_evidence, question
                assert answer.status == "answered", question
                assert any(
                    source in actual_sources for source in expected_sources
                ), f"{question} 未命中预期来源；实际来源：{sorted(actual_sources)}"
                assert all(item.document_title for item in answer.evidence)
                assert all(item.section_title for item in answer.evidence)
                assert all(item.text for item in answer.evidence)
                assert all(len(item.document_md5) == 32 for item in answer.evidence)
            else:
                assert not answer.has_sufficient_evidence, question
                assert answer.status == "insufficient_evidence", question
                assert answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE

            print(f"通过：{question}")
            passed += 1

        _test_incremental_updates(temp_root)

    print(f"RAG 检索用例：{passed}/{len(cases)} 通过。")
    print("RAG 工程测试：去重、持久化、增量增删改、来源、拒答、异常降级均通过。")


if __name__ == "__main__":
    main()
