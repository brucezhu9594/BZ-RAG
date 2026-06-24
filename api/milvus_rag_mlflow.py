"""Milvus 混合检索流水线，MLflow tracing 覆盖 span/trace/session。

- 顶层 milvus_rag_mlflow_query 是 AGENT span（= trace 根），设 session_id 做多轮会话分组
- retrieve / rerank / generate 各一个 span（RETRIEVER / RERANKER / LLM）
- retriever span 输出按 MLflow schema 设为 mlflow.entities.Document 列表，
  这样 RetrievalGroundedness / RetrievalRelevance 等内置 judge 能从 trace 提取上下文
- trace 上报到 MLFLOW_TRACKING_URI 指向的 tracking server，供单轮 + 多轮离线评估
"""

import os

import mlflow
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from mlflow.entities import Document as MlflowDocument
from mlflow.entities import SpanType
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"
DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2
RRF_K = 60

MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "bz-rag-milvus")

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(MLFLOW_EXPERIMENT)


@mlflow.trace(span_type=SpanType.RETRIEVER)
def _retrieve_span(query: str) -> list[Document]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
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
    # MLflow 的 RAG judge 要求 RETRIEVER span 输出为 mlflow.entities.Document 列表。
    # genai.evaluate 的 predict_fn 预检在禁用 tracing 下试跑，此时拿不到 span，需判空。
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_outputs(
            [MlflowDocument(page_content=d.page_content, metadata=d.metadata) for d in docs]
        )
    return docs


@mlflow.trace(span_type=SpanType.RERANKER)
def _rerank_span(query: str, docs: list[Document]) -> list[Document]:
    from common.zhipu_rerank import rerank

    return rerank(query, docs, top_n=RERANK_TOP_K)


@mlflow.trace(span_type=SpanType.LLM)
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
    return msg.content or ""


@mlflow.trace(name="milvus-hybrid-rag", span_type=SpanType.AGENT)
def milvus_rag_mlflow_query(query: str, session_id: str | None = None) -> str:
    # session_id 写入 trace metadata（mlflow.trace.session），多轮评估按它分组对话。
    if session_id:
        mlflow.update_current_trace(session_id=session_id)
    docs = _retrieve_span(query)
    reranked = _rerank_span(query, docs)
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', '')}\nContent: {d.page_content}" for d in reranked
    )
    return _generate_span(query, context)
