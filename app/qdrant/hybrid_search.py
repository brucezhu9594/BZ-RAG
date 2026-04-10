import os

from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models

from app.qdrant.bm25 import query_sparse_vector
from common.keyword_expansion import expand_keywords

load_dotenv()
MODEL = os.environ["MODEL_ID"]

COLLECTION_NAME = "hewa_help_collection"

DENSE_LIMIT = 10
SPARSE_LIMIT = 10
FINAL_TOP_K = 2


def _retrieve(query: str) -> tuple[str, list]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    client = QdrantClient(host="localhost", port=6333)

    # 关键词补充（增强 BM25 路召回）
    expanded_query = expand_keywords(query)
    if expanded_query != query:
        print(f"[关键词补充] {query} → {expanded_query}")

    query_vector = embeddings.embed_query(query)
    query_sparse = query_sparse_vector(expanded_query)

    # Dense + Sparse 混合检索，RRF 融合
    prefetch = [
        models.Prefetch(
            query=query_vector,
            using="",
            limit=DENSE_LIMIT,
        ),
        models.Prefetch(
            query=query_sparse,
            using="bm25",
            limit=SPARSE_LIMIT,
        ),
    ]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=FINAL_TOP_K,
        with_payload=True,
    ).points
    print(f"results:{results}")

    serialized = "\n\n".join(
        f"Source: {p.payload.get('metadata').get('source', '')}\nContent: {p.payload.get('page_content', '')}"
        for p in results
    )
    client.close()
    return serialized, results


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
