import os

from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from common.zhipu_rerank import rerank
from common.contextual_rewriter import contextual_rewrite

load_dotenv()
MODEL = os.environ["MODEL_ID"]

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"

DENSE_LIMIT = 10
SPARSE_LIMIT = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2
RRF_K = 60
MAX_HISTORY_ROUNDS = 3  # 保留最近 N 轮对话


def _retrieve(query: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    # 历史感知查询改写
    rewritten = contextual_rewrite(query, history or [])
    if rewritten != query:
        print(f"[历史感知改写] {query} → {rewritten}")

    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    client = MilvusClient(uri=MILVUS_URI)

    query_vector = embeddings.embed_query(rewritten)

    # 路径 1: Dense 向量语义检索
    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=DENSE_LIMIT,
    )

    # 路径 2: Sparse BM25 关键词检索
    sparse_req = AnnSearchRequest(
        data=[rewritten],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=SPARSE_LIMIT,
    )

    # RRF 融合排序（多取候选，交给 Rerank 精排）
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=RRF_K),
        limit=RETRIEVE_TOP_K,
        output_fields=["text", "source"],
    )

    # 转换为 Document 供 Rerank 使用
    candidates = [
        Document(
            page_content=r["entity"].get("text", ""),
            metadata={"source": r["entity"].get("source", "")},
        )
        for r in results[0]
    ]

    # Rerank 精排
    reranked = rerank(query, candidates, top_n=RERANK_TOP_K)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', '')}\nContent: {doc.page_content}"
        for doc in reranked
    )
    return serialized, reranked


def rag(user_input: str, history: list[dict] | None = None) -> str:
    serialized, _ = _retrieve(user_input, history)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{serialized}"
    )
    llm = ChatOpenAI(model=MODEL, temperature=0.7)
    messages = [{"role": "system", "content": system_prompt}]
    # 带上历史上下文，让回答连贯
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    msg = llm.invoke(messages)
    return msg.content or ""


def main():
    print("Chat with AI (type 'exit' to quit)")
    history: list[dict] = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            print("请输入内容，不能为空\n")
            continue

        # 滑动窗口：只保留最近 N 轮（每轮 = 1条user + 1条assistant = 2条）
        recent_history = history[-(MAX_HISTORY_ROUNDS * 2):]

        answer = rag(user_input, recent_history)
        print(f"AI: {answer}")

        # 记录本轮对话
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
