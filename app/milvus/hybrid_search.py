import os

from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

load_dotenv()
MODEL = os.environ["MODEL_ID"]

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"

DENSE_LIMIT = 10
SPARSE_LIMIT = 10
FINAL_TOP_K = 2
RRF_K = 60


def _retrieve(query: str) -> tuple[str, list[dict]]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    client = MilvusClient(uri=MILVUS_URI)

    query_vector = embeddings.embed_query(query)

    # 路径 1: Dense 向量语义检索
    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=DENSE_LIMIT,
    )

    # 路径 2: Sparse BM25 关键词检索
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=SPARSE_LIMIT,
    )

    # RRF 融合排序
    results = client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=RRF_K),
        limit=FINAL_TOP_K,
        output_fields=["text", "source"],
    )

    serialized = "\n\n".join(
        f"Source: {r['entity'].get('source', '')}\nContent: {r['entity'].get('text', '')}"
        for r in results[0]
    )
    return serialized, results[0]


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
