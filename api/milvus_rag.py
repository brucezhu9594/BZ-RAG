"""Milvus 混合检索流水线，DeepEval tracing 覆盖 span/trace/thread。

- 顶层 milvus_rag_query 是 agent span（= trace），设 thread_id 做会话分组
- retrieve / rerank / generate 各一个 span
- 读 CONFIDENT_API_KEY 后台异步上报 Confident AI Observatory
"""

import os

from deepeval.tracing import (
    observe,
    update_current_span,
    update_current_trace,
    update_retriever_span,
)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from common.zhipu_embed import ZhipuEmbeddings

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"
DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
# 4 而不是 2：chunk 的 p50 只有 161 字，top-2 喂给 LLM 的上下文才 ~320 字。实测存在
# 金块排在 hybrid 第 3 位却被切掉的 case；而 rerank 分数在同话题候选上会饱和
# （六条候选全 1.0，见 common/zhipu_rerank.py 的注释），并列时只能退回 RRF 名次，
# 留 2 个位置的余量太薄。
RERANK_TOP_K = 4
RRF_K = 60


@observe(type="retriever")
def _retrieve_span(query: str) -> list[Document]:
    embeddings = ZhipuEmbeddings(model="embedding-3")
    client = MilvusClient(uri=MILVUS_URI)
    query_vector = embeddings.embed_query(query)

    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=DENSE_LIMIT,
    )
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=SPARSE_LIMIT,
    )
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=RRF_K),
        limit=RETRIEVE_TOP_K,
        output_fields=["text", "source"],
    )
    docs = [
        Document(
            page_content=r["entity"].get("text", ""),
            metadata={"source": r["entity"].get("source", "")},
        )
        for r in results[0]
    ]
    # Confident AI 对 retriever span 必填 embedder（embedding 模型名）。
    update_retriever_span(embedder="embedding-3", top_k=RETRIEVE_TOP_K)
    update_current_span(input=query, retrieval_context=[d.page_content for d in docs])
    return docs


@observe(type="tool")
def _rerank_span(query: str, docs: list[Document]) -> list[Document]:
    from common.zhipu_rerank import rerank

    reranked = rerank(query, docs, top_n=RERANK_TOP_K)
    update_current_span(input=query, output=[d.page_content for d in reranked])
    return reranked


@observe(type="llm")
def _generate_span(query: str, context: str) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
    )
    answer = msg.content or ""
    update_current_span(input=query, output=answer)
    return answer


@observe(type="agent")
def milvus_rag_query(query: str, thread_id: str | None = None) -> str:
    docs = _retrieve_span(query)
    reranked = _rerank_span(query, docs)
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', '')}\nContent: {d.page_content}" for d in reranked
    )
    answer = _generate_span(query, context)
    update_current_trace(
        thread_id=thread_id,
        input=query,
        output=answer,
        name="milvus-hybrid-rag",
    )
    return answer
