from collections import defaultdict

from langchain_core.documents import Document

RRF_K = 60
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5


def rrf_merge(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    k: int = RRF_K,
    dense_w: float = DENSE_WEIGHT,
    bm25_w: float = BM25_WEIGHT,
    top_n: int = 6,
) -> list[Document]:
    """Reciprocal Rank Fusion: 将两路检索结果按 RRF 公式融合排序。"""
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        uid = f"{doc.metadata}|{doc.page_content[:80]}"
        scores[uid] += dense_w / (k + rank + 1)
        doc_map[uid] = doc

    for rank, doc in enumerate(bm25_docs):
        uid = f"{doc.metadata}|{doc.page_content[:80]}"
        scores[uid] += bm25_w / (k + rank + 1)
        doc_map[uid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[uid] for uid, _ in ranked[:top_n]]
