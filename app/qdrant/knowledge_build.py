"""Qdrant 入库：dense COSINE + 客户端构建的 BM25 稀疏向量。

切到 qdrant_client 原生 upsert，避免 QdrantVectorStore 内部二次 embed；
payload 仍按 {"page_content": ..., "metadata": {...}} 摆放，跟 hybrid_search.py 对齐。
"""

import pathlib
import sys
import uuid

from qdrant_client import QdrantClient, models

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from app.qdrant.bm25 import build_and_save  # noqa: E402
from common.etl import prepare_knowledge_base  # noqa: E402

COLLECTION_NAME = "hewa_help_collection"


def etl():
    texts, metadatas, vectors = prepare_knowledge_base()
    if not texts:
        raise RuntimeError("知识库为空")

    client = QdrantClient(host="localhost", port=6333)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    dim = len(vectors[0])
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        sparse_vectors_config={"bm25": models.SparseVectorParams()},
    )

    # BM25 稀疏向量：jieba 切词 + 自建词表/IDF，保存到 bm25_meta.json 供查询时复用
    sparse_vectors = build_and_save(texts)

    # 一次 upsert 把 dense + sparse + payload 全写入；dense 用空字符串作为默认 unnamed 槽位，
    # 与 hybrid_search.py 里 `using=""` 对齐
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={"": v, "bm25": sv},
            payload={"page_content": t, "metadata": m},
        )
        for t, m, v, sv in zip(texts, metadatas, vectors, sparse_vectors, strict=False)
    ]

    BATCH = 256
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + BATCH])
        print(f"  已插入第 {i // BATCH + 1} 批，共 {len(points[i : i + BATCH])} 条")
    print(f"已插入 {len(points)} 条文档到 Qdrant（含 BM25 稀疏向量）")
    client.close()


if __name__ == "__main__":
    etl()
