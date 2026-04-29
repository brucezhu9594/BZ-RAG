import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI

from app.chroma.bm25_index import _tokenize, build_bm25_index
from app.chroma.rrf import rrf_merge

load_dotenv()
MODEL = os.environ["MODEL_ID"]

DENSE_K = 10
BM25_K = 10
FINAL_TOP_K = 2


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """根据用户问题检索知识库，返回相关文档片段。当用户提出任何问题时都应调用此工具。"""
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    vector_store = Chroma(
        collection_name="hewa_help_collection",
        embedding_function=embeddings,
        persist_directory="./db",
    )

    # 路径 1: Dense 向量检索
    dense_docs = vector_store.similarity_search(query, k=DENSE_K)
    print(f"dense docs: {dense_docs}")

    # 路径 2: BM25 关键词检索
    bm25, all_docs = build_bm25_index(vector_store)
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[
        :BM25_K
    ]
    bm25_docs = [all_docs[i] for i in top_indices]
    print(f"bm25_docs: {bm25_docs}")

    # RRF 融合
    merged = rrf_merge(dense_docs, bm25_docs, top_n=FINAL_TOP_K)
    print(f"merged docs: {merged}")

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', '')}\nContent: {doc.page_content}" for doc in merged
    )
    return serialized, merged


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
        ai_message = result["messages"][-1]
        print(f"AI: {ai_message.content}")


if __name__ == "__main__":
    main()
