import os
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.environ["MODEL_ID"]

def _retrieve_for_query(query: str) -> tuple[str, list]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    vector_store = Chroma(
        collection_name="hewa_help_collection",
        embedding_function=embeddings,
        persist_directory="./db",
    )
    retrieved_docs = vector_store.similarity_search(query, k=2)
    print(f"retrieved_docs: {retrieved_docs}")
    serialized = "\n\n".join(
        f"Content: {doc.page_content}"
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