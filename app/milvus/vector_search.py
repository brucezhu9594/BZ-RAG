import os

from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient

load_dotenv()
MODEL = os.environ["MODEL_ID"]

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "hewa_help_collection"


def _retrieve_for_query(query: str) -> tuple[str, list[dict]]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    client = MilvusClient(uri=MILVUS_URI)

    query_vector = embeddings.embed_query(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=2,
        output_fields=["text", "source"],
    )[0]

    print(f"results: {results}")
    serialized = "\n\n".join(
        f"Content: {r['entity'].get('text', '')}"
        for r in results
    )
    return serialized, results


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
    print(f"system_prompt: {system_prompt}")

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
