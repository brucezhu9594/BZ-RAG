import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.chroma.bm25_index import build_bm25_index, _tokenize
from app.chroma.rrf import rrf_merge
from common.reranker import rerank
from common.query_rewriter import rewrite_query

load_dotenv()
MODEL = os.environ["MODEL_ID"]

DENSE_K = 10
BM25_K = 10
RETRIEVE_TOP_K = 6
RERANK_TOP_K = 2


def _get_vector_store() -> Chroma:
    return Chroma(
        collection_name="hewa_help_collection",
        embedding_function=ZhipuAIEmbeddings(model="embedding-3"),
        persist_directory="./db",
    )


def _retrieve(query: str) -> tuple[str, list[Document]]:
    # 查询改写
    rewritten = rewrite_query(query)
    if rewritten != query:
        print(f"[查询改写] {query} → {rewritten}")

    vector_store = _get_vector_store()

    # 路径 1: Dense 向量检索
    dense_docs = vector_store.similarity_search(rewritten, k=DENSE_K)

    # 路径 2: BM25 关键词检索
    bm25, all_docs = build_bm25_index(vector_store)
    query_tokens = _tokenize(rewritten)
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:BM25_K]
    bm25_docs = [all_docs[i] for i in top_indices]

    # RRF 融合
    merged = rrf_merge(dense_docs, bm25_docs, top_n=RETRIEVE_TOP_K)

    # Reranker 重排序
    reranked = rerank(query, merged, top_n=RERANK_TOP_K)

    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in reranked
    )
    return serialized, reranked


def rag(user_input: str) -> str:
    serialized, _ = _retrieve(user_input)
    system_prompt = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{serialized}"
    )
    llm = ChatOpenAI(model=MODEL, temperature=0.7)
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
