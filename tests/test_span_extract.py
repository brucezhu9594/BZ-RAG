"""从 Phoenix span 里拆出判官要的三样东西。

fixture 用的是真实 span 形状（2026-09-10 从本机 bz-rag-ci project 取出后简化）：
根 AGENT span 的 input.value / output.value 是纯字符串；
RERANKER span 的 output.value 是 JSON 字符串，形如
[{"id": null, "metadata": {...}, "page_content": "..."}]。
"""

import json

from evaluation.phoenix.span_extract import extract_eval_input


def _span(kind, name, attrs, span_id="s1"):
    return {
        "id": span_id,
        "name": name,
        "span_kind": kind,
        "attributes": attrs,
        "context": {"trace_id": "t1", "span_id": span_id},
    }


ROOT = _span(
    "AGENT",
    "milvus-hybrid-rag",
    {"input.value": "禾蛙平台的创始人是谁？", "output.value": "创始人是何洪锴。"},
    span_id="root",
)
RERANK = _span(
    "RERANKER",
    "_rerank",
    {
        "output.value": json.dumps(
            [
                {"id": None, "metadata": {"source": "u1"}, "page_content": "片段一"},
                {"id": None, "metadata": {"source": "u2"}, "page_content": "片段二"},
            ],
            ensure_ascii=False,
        )
    },
    span_id="rr",
)


class TestHappyPath:
    def test_extracts_all_three(self):
        got = extract_eval_input(ROOT, [ROOT, RERANK])
        assert got == {
            "question": "禾蛙平台的创始人是谁？",
            "answer": "创始人是何洪锴。",
            "contexts": ["片段一", "片段二"],
        }

    def test_prefers_reranker_over_retriever(self):
        """判官必须看到真正喂给 LLM 的那份，也就是重排后的——与期 1 的取舍一致。"""
        retriever = _span(
            "RETRIEVER",
            "retrieve",
            {
                "output.value": json.dumps(
                    [{"page_content": "重排前的六条之一"}], ensure_ascii=False
                )
            },
            span_id="rt",
        )
        got = extract_eval_input(ROOT, [ROOT, retriever, RERANK])
        assert got["contexts"] == ["片段一", "片段二"]


class TestSkips:
    def test_missing_answer_returns_none(self):
        root = _span("AGENT", "milvus-hybrid-rag", {"input.value": "问题"}, span_id="root")
        assert extract_eval_input(root, [root]) is None

    def test_missing_question_returns_none(self):
        root = _span("AGENT", "milvus-hybrid-rag", {"output.value": "答案"}, span_id="root")
        assert extract_eval_input(root, [root]) is None

    def test_no_reranker_yields_empty_contexts_not_none(self):
        """没有重排 span 不算提取失败：answer_relevancy 与 refusal_check 不需要上下文。"""
        got = extract_eval_input(ROOT, [ROOT])
        assert got is not None
        assert got["contexts"] == []

    def test_malformed_reranker_json_yields_empty_contexts(self):
        bad = _span("RERANKER", "_rerank", {"output.value": "not-json"}, span_id="rr")
        got = extract_eval_input(ROOT, [ROOT, bad])
        assert got is not None
        assert got["contexts"] == []
