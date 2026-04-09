import os

from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

load_dotenv()
MODEL = os.environ["MODEL_ID"]

COLLECTION_NAME = "hewa_help_collection"


def _get_vector_store() -> QdrantVectorStore:
    client = QdrantClient(host="localhost", port=6333)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=ZhipuAIEmbeddings(model="embedding-3"),
        retrieval_mode=RetrievalMode.DENSE,
    )


def _retrieve(query: str) -> tuple[str, list]:
    vector_store = _get_vector_store()
    results = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', '')}\nContent: {doc.page_content}"
        for doc in results
    )
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
