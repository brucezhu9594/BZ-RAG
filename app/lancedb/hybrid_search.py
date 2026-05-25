"""LanceDB 原生 hybrid 检索（dense + FTS BM25），用 RRF 融合。

跟 Chroma/Qdrant 的差异：FTS/BM25 是 LanceDB 服务端原生实现（Tantivy），
不用客户端再维护 IDF/词表 JSON。
"""

import os
import pathlib
import sys

import jieba
import lancedb
from dotenv import load_dotenv
from lancedb.rerankers import RRFReranker
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.keyword_expansion import expand_keywords  # noqa: E402

load_dotenv()
MODEL = os.environ["MODEL_ID"]

COLLECTION_NAME = "hewa_help_collection"
DB_PATH = str(pathlib.Path(__file__).parent / "db")

FINAL_TOP_K = 2


def _tokenize_for_fts(text: str) -> str:
    return " ".join(w for w in jieba.cut(text) if w.strip())


def _retrieve(query: str) -> tuple[str, list]:
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    db = lancedb.connect(DB_PATH)
    tbl = db.open_table(COLLECTION_NAME)

    # 关键词补充（增强 FTS 路召回）
    expanded_query = expand_keywords(query)
    if expanded_query != query:
        print(f"[关键词补充] {query} → {expanded_query}")

    query_vector = embeddings.embed_query(query)
    query_text = _tokenize_for_fts(expanded_query)

    # 显式 vector + text，LanceDB 服务端跑 dense + BM25，再 RRF 融合
    results = (
        tbl.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .rerank(RRFReranker())
        .limit(FINAL_TOP_K)
        .to_list()
    )
    print(f"results:{results}")

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
