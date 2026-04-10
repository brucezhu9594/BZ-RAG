import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.chroma.bm25_index import build_bm25_index, _tokenize
from app.chroma.rrf import rrf_merge
from common.reranker import rerank
from common.query_expansion import expand_query

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


def _deduplicate(docs: list[Document]) -> list[Document]:
    """按 page_content 去重，保留首次出现的顺序。"""
    seen = set()
    result = []
    for doc in docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def _retrieve(query: str) -> tuple[str, list[Document]]:
    # 多查询扩展
    sub_queries = expand_query(query)
    print(f"[多查询扩展] {query} → {sub_queries}")

    vector_store = _get_vector_store()
    bm25, all_docs = build_bm25_index(vector_store)

    # 对每个子查询分别做 Dense + BM25 检索
    all_dense_docs: list[Document] = []
    all_bm25_docs: list[Document] = []
    for sub_q in sub_queries:
        # Dense
        dense_docs = vector_store.similarity_search(sub_q, k=DENSE_K)
        all_dense_docs.extend(dense_docs)

        # BM25
        query_tokens = _tokenize(sub_q)
        bm25_scores = bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:BM25_K]
        all_bm25_docs.extend(all_docs[i] for i in top_indices)

    # 去重 + RRF 融合
    all_dense_docs = _deduplicate(all_dense_docs)
    all_bm25_docs = _deduplicate(all_bm25_docs)
    merged = rrf_merge(all_dense_docs, all_bm25_docs, top_n=RETRIEVE_TOP_K)

    # Reranker 重排序（用原始 query 打分）
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
