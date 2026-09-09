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

    # 有意向 API 要回**全部**候选而不是 top_n：截断必须发生在下面的稳定排序之后。
    # 把 top_n 交给服务端截断会先按它自己的并列顺序切一刀，我们再怎么排也救不回来
    # 已经被切掉的那些（实测 top_n=2 时服务端回的就是输入的最后两条）。
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
            "top_n": len(documents),
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    # 按 (-分数, 原始下标) 排序，而不是照抄 API 的返回顺序。
    #
    # 智谱 rerank 的分数在同话题候选上会饱和：实测拿 hybrid 返回的 6 条真实片段去问，
    # 六条**全部**是 1.0（连"香蕉是一种热带水果"这种完全无关的文档都能拿 0.83，
    # 区分度只在 [0.83, 1.0] 这一小段里）。而 API 对并列项返回的是输入的**倒序**，
    # 原实现原样保留该顺序，于是整个重排等价于 documents[::-1][:top_n] ——
    # 系统性地挑中上游 hybrid/RRF 排名最差的几条。
    # 实测三条金标问题：修复前金块召回 0/3，修复后 3/3。
    #
    # 并列时回退到原始下标 = 保留上游 RRF 的名次，即"模型有区分度时听模型的，
    # 没区分度时不要破坏上游已经排好的顺序"。
    scored = sorted(
        ((r["relevance_score"], r["index"]) for r in data["results"]),
        key=lambda si: (-si[0], si[1]),
    )

    reranked: list[Document] = []
    for score, index in scored[:top_n]:
        doc = documents[index]
        doc.metadata["rerank_score"] = score
        reranked.append(doc)

    return reranked
