import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.environ["MODEL_ID"]


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """根据用户问题检索知识库，返回相关文档片段。当用户提出任何问题时都应调用此工具。"""
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    vector_store = Chroma(
        collection_name="hewa_help_collection",
        embedding_function=embeddings,
        persist_directory="./db",
    )
    retrieved_docs = vector_store.similarity_search(query, k=6)
    print(f"retrieved_docs: {retrieved_docs}")
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', '')}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


SYSTEM_PROMPT = (
    "你是一个知识库检索助手。"
    "收到用户问题后，请先调用 retrieve_context 工具检索知识库。"
    "请仅依据检索结果回答用户问题。"
    "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
)


def rag_agent():
    llm = ChatOpenAI(model=MODEL, temperature=0.7)
    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def main():
    agent = rag_agent()
    print("Chat with AI Agent (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if not user_input:
            print("请输入内容，不能为空\n")
            continue
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        # 取最后一条 AI 消息作为回答
        ai_message = result["messages"][-1]
        print(f"AI: {ai_message.content}")


if __name__ == "__main__":
    main()
