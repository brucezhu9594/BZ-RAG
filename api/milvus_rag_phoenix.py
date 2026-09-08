"""Milvus 混合检索流水线，Phoenix / OpenInference tracing。

与 api/milvus_rag_mlflow.py 平行存在：检索、重排、生成逻辑与常量完全一致，
只把 tracing 承载从 MLflow 换成 Phoenix。ChatOpenAI 的 LLM span 由
openinference-instrumentation-langchain 自动产出，不用手写。

与 MLflow 版的一处有意差异：本模块把**重排后**的片段作为返回值交出去，
判官直接用它，不再回头去 trace 里刨 RETRIEVER span。好处有两个——
判官看到的上下文与真正喂给 LLM 的完全同一份（MLflow 版的 scorer 取的是
重排前的 6 条，与生成实际用的 2 条不一致），且少一层“提取失败”的失败模式。
"""

import os

# 本机 Milvus 走 gRPC、Phoenix collector 走 HTTP，本机 Privoxy 会拦 localhost。
# 必须在任何 milvus / phoenix import 之前设置。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv

load_dotenv()

from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from phoenix.otel import register
from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from api.history_utils import build_chat_messages, build_rewrite_prompt

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"
DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2
RRF_K = 60

# project_name 由环境变量决定：CI 里 bz-rag-ci，影子 canary 上 bz-rag-canary。
_tracer_provider = register(
    project_name=os.environ.get("PHOENIX_PROJECT_NAME", "bz-rag-ci"),
    endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    + "/v1/traces",
    auto_instrument=True,  # 挂载已安装的 openinference instrumentor（LangChain）
)
_tracer = _tracer_provider.get_tracer(__name__)


@_tracer.chain
def _rewrite_query(query: str, history: list[tuple[str, str]]) -> str:
    # 用对话历史把指代追问改写成独立问题，供检索用。temperature=0 保证可复现，
    # 改写失败（空串）退回原问题兜底。
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.0, request_timeout=60)
    msg = llm.invoke([{"role": "user", "content": build_rewrite_prompt(query, history)}])
    return (msg.content or "").strip() or query


def _retrieve(query: str) -> list[Document]:
    with _tracer.start_as_current_span(
        "retrieve", openinference_span_kind="retriever"
    ) as span:
        span.set_input(query)
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
        span.set_output([{"page_content": d.page_content, "metadata": d.metadata} for d in docs])
        return docs


@_tracer.reranker
def _rerank(query: str, docs: list[Document]) -> list[Document]:
    from common.zhipu_rerank import rerank

    return rerank(query, docs, top_n=RERANK_TOP_K)


@_tracer.chain
def _generate(query: str, context: str, history: list[tuple[str, str]] | None = None) -> str:
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke(build_chat_messages(system_prompt, query, history))
    return msg.content or ""


def milvus_rag_phoenix_query_with_context(
    query: str,
    session_id: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> tuple[str, list[str]]:
    """跑一次检索 + 生成，返回 (答案, 重排后喂给 LLM 的片段列表)。"""
    with _tracer.start_as_current_span(
        "milvus-hybrid-rag", openinference_span_kind="agent"
    ) as span:
        span.set_input(query)
        if session_id:
            # OpenInference 的会话分组约定属性，多轮评估按它分组。
            span.set_attribute("session.id", session_id)
        search_query = _rewrite_query(query, history) if history else query
        docs = _retrieve(search_query)
        reranked = _rerank(search_query, docs)
        context = "\n\n".join(
            f"Source: {d.metadata.get('source', '')}\nContent: {d.page_content}"
            for d in reranked
        )
        answer = _generate(query, context, history)
        span.set_output(answer)
    return answer, [d.page_content for d in reranked]


def milvus_rag_phoenix_query(
    query: str,
    session_id: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    answer, _ = milvus_rag_phoenix_query_with_context(query, session_id, history)
    return answer
