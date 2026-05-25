"""LanceDB 纯向量检索 RAG。"""

import os
import pathlib

import lancedb
from dotenv import load_dotenv
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
MODEL = os.environ["MODEL_ID"]

COLLECTION_NAME = "hewa_help_collection"
DB_PATH = str(pathlib.Path(__file__).parent / "db")


def _retrieve(query: str) -> tuple[str, list]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(COLLECTION_NAME)

    query_vector = embeddings.embed_query(query)
    results = tbl.search(query_vector, vector_column_name="vector").limit(2).to_list()
    serialized = "\n\n".join(
        f"Source: {r.get('source', '')}\nContent: {r.get('text', '')}" for r in results
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
