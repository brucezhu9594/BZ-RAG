"""把 Phoenix span 拆成判官要的输入。

评估对象是**根 AGENT span**，不是 spec §4.8 原文写的 LLM span。实测一条 trace：

    AGENT      milvus-hybrid-rag   (root)   input.value=问题  output.value=答案
      RETRIEVER  retrieve                   output.value=重排前的 6 条
      RERANKER   _rerank                    output.value=重排后喂给 LLM 的 4 条
      CHAIN      _generate
        LLM      ChatOpenAI                 langchain instrumentor 记的
        LLM      ChatCompletion             openai instrumentor 记的（同一次调用第二遍）

按 LLM 抽会把同一次生成算两次，多轮时还会多抓 _rewrite_query 的 LLM span。
根 AGENT span 一次请求恰好一条，且两个字段直接可用，不用去 prompt 里做字符串切割。

上下文取 RERANKER 而不是 RETRIEVER：判官必须看到真正喂给 LLM 的那一份。
这与 api/milvus_rag_phoenix.py 顶部记的取舍一致（MLflow 版取的是重排前的 6 条，
与生成实际用的不一致，那是那边的已知缺陷）。
"""

import json
from typing import Any

_CONTEXT_SPAN_KIND = "RERANKER"


def _contexts_from(spans: list[dict[str, Any]]) -> list[str]:
    for s in spans:
        if s.get("span_kind") != _CONTEXT_SPAN_KIND:
            continue
        raw = (s.get("attributes") or {}).get("output.value")
        if not raw:
            return []
        try:
            docs = json.loads(raw)
        except (TypeError, ValueError):
            # 形状变了就当没有上下文：faithfulness 会因此判低分，
            # 这比让整个 worker 崩掉、或静默丢样本都好。
            return []
        if not isinstance(docs, list):
            return []
        return [d.get("page_content", "") for d in docs if isinstance(d, dict)]
    return []


def extract_eval_input(
    root_span: dict[str, Any], trace_spans: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """返回 {"question", "answer", "contexts"}；缺问题或答案时返回 None。

    返回 None 表示"这条 span 不适合评"，调用方应跳过并计数，**不要**当成判官失败——
    两者的处置完全不同：前者是正常的形状过滤，后者要进错误率闸门。
    """
    attrs = root_span.get("attributes") or {}
    question = attrs.get("input.value")
    answer = attrs.get("output.value")
    if not question or not answer:
        return None
    return {
        "question": str(question),
        "answer": str(answer),
        "contexts": _contexts_from(trace_spans),
    }
