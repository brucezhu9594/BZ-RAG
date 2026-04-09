"""使用智谱 Rerank 模型对检索结果重排序。"""

import os

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

RERANK_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"
RERANK_MODEL = "rerank"


def _get_api_key() -> str:
    key = os.environ.get("ZHIPUAI_API_KEY", "")
    if not key:
        raise ValueError("未设置 ZHIPUAI_API_KEY 环境变量")
    return key


def rerank(
    query: str,
    documents: list[Document],
    top_n: int = 5,
) -> list[Document]:
    """调用智谱 Rerank API，按相关性重排序文档。

    Args:
        query: 用户查询
        documents: 待排序的文档列表
        top_n: 返回前 N 个最相关的文档

    Returns:
        按相关性降序排列的文档列表
    """
    if not documents:
        return []

    doc_texts = [doc.page_content for doc in documents]

    response = requests.post(
        RERANK_URL,
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
        },
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_n, len(documents)),
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    reranked: list[Document] = []
    for result in data["results"]:
        doc = documents[result["index"]]
        doc.metadata["rerank_score"] = result["relevance_score"]
        reranked.append(doc)

    return reranked
