"""api/milvus_rag.py 单测：mock 外部依赖，验证编排与 thread_id 透传。"""
from unittest.mock import MagicMock

from langchain_core.documents import Document

from api import milvus_rag


def _doc(text, source="helpContent/10006"):
    return Document(page_content=text, metadata={"source": source})


def test_query_chains_retrieve_rerank_generate(monkeypatch):
    calls = []

    def fake_retrieve(query):
        calls.append(("retrieve", query))
        return [_doc("d1"), _doc("d2")]

    def fake_rerank(query, docs):
        calls.append(("rerank", query, [d.page_content for d in docs]))
        return [docs[0]]

    def fake_generate(query, context):
        calls.append(("generate", query, context))
        return "最终答案"

    monkeypatch.setattr(milvus_rag, "_retrieve_span", fake_retrieve)
    monkeypatch.setattr(milvus_rag, "_rerank_span", fake_rerank)
    monkeypatch.setattr(milvus_rag, "_generate_span", fake_generate)
    monkeypatch.setattr(milvus_rag, "update_current_trace", lambda **kwargs: None)

    answer = milvus_rag.milvus_rag_query("怎么开发票", thread_id="t-1")

    assert answer == "最终答案"
    assert calls[0][0] == "retrieve"
    assert calls[1][0] == "rerank"
    assert calls[2][0] == "generate"
    assert "d1" in calls[2][2]


def test_query_passes_thread_id_to_trace(monkeypatch):
    captured = {}

    monkeypatch.setattr(milvus_rag, "_retrieve_span", lambda q: [_doc("d1")])
    monkeypatch.setattr(milvus_rag, "_rerank_span", lambda q, docs: docs)
    monkeypatch.setattr(milvus_rag, "_generate_span", lambda q, c: "ans")

    def fake_update_trace(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(milvus_rag, "update_current_trace", fake_update_trace)

    milvus_rag.milvus_rag_query("问题", thread_id="thread-abc")

    assert captured.get("thread_id") == "thread-abc"
    assert captured.get("input") == "问题"
    assert captured.get("output") == "ans"


def test_query_thread_id_optional(monkeypatch):
    captured = {}
    monkeypatch.setattr(milvus_rag, "_retrieve_span", lambda q: [_doc("d1")])
    monkeypatch.setattr(milvus_rag, "_rerank_span", lambda q, docs: docs)
    monkeypatch.setattr(milvus_rag, "_generate_span", lambda q, c: "ans")
    monkeypatch.setattr(milvus_rag, "update_current_trace", lambda **kwargs: captured.update(kwargs))

    answer = milvus_rag.milvus_rag_query("问题")

    assert answer == "ans"
    assert captured.get("thread_id") is None
