import os
from collections import defaultdict

from langchain.tools import tool
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from app.chroma.bm25_index import build_bm25_index, _tokenize

load_dotenv()
MODEL = os.environ["MODEL_ID"]

DENSE_K = 10
BM25_K = 10
RRF_K = 60
FINAL_TOP_K = 6
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5


def _rrf_merge(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    k: int = RRF_K,
    dense_w: float = DENSE_WEIGHT,
    bm25_w: float = BM25_WEIGHT,
    top_n: int = FINAL_TOP_K,
) -> list[Document]:
    """Reciprocal Rank Fusion: 将两路检索结果按 RRF 公式融合排序。"""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        uid = f"{doc.metadata}|{doc.page_content[:80]}"
        scores[uid] += dense_w / (k + rank + 1)
        doc_map[uid] = doc

    for rank, doc in enumerate(bm25_docs):
        uid = f"{doc.metadata}|{doc.page_content[:80]}"
        scores[uid] += bm25_w / (k + rank + 1)
        doc_map[uid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[uid] for uid, _ in ranked[:top_n]]


def _retrieve_for_query(query: str) -> tuple[str, list]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    vector_store = Chroma(
        collection_name="hewa_help_collection",
        embedding_function=embeddings,
        persist_directory="./db",
    )

    # --- 路径 1: Dense 向量检索 ---
    dense_docs = vector_store.similarity_search(query, k=DENSE_K)
    print(f"[Dense] query: {query}, 返回 {len(dense_docs)} 条")

    # --- 路径 2: BM25 关键词检索 ---
    bm25, all_docs = build_bm25_index(vector_store)
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:BM25_K]
    bm25_docs = [all_docs[i] for i in top_indices]
    print(f"[BM25]  query tokens: {query_tokens}, 返回 {len(bm25_docs)} 条")

    # --- RRF 融合 ---
    retrieved_docs = _rrf_merge(dense_docs, bm25_docs)
    print(f"[Hybrid] RRF 融合后取 top-{FINAL_TOP_K}, 实际 {len(retrieved_docs)} 条")

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    serialized, retrieved_docs = _retrieve_for_query(query)
    return serialized, retrieved_docs


def rag(user_input: str) -> str:
    # 使用「检索 + 单轮对话」，避免 create_agent 的 tools 调用。
    # MiniMax 等 OpenAI 兼容接口在 function calling 时可能返回 choices=null。
    serialized, _ = _retrieve_for_query(user_input)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{serialized}"
    )
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0.7,
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    )
    return msg.content or ""


def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            print("请输入内容，不能为空\n")
            continue
        print(f"AI: {rag(user_input)}")


if __name__ == "__main__":
    main()
