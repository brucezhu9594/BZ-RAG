"""Chroma 入库：用预算好的 vectors 直接 add，不走 langchain_chroma 的二次 embed。"""

import pathlib
import sys
import uuid

import chromadb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from common.etl import prepare_knowledge_base  # noqa: E402

COLLECTION_NAME = "hewa_help_collection"
DB_PATH = str(pathlib.Path(__file__).resolve().parents[2] / "db")


def etl():
    texts, metadatas, vectors = prepare_knowledge_base()
    if not texts:
        raise RuntimeError("知识库为空")

    client = chromadb.PersistentClient(path=DB_PATH)
    # 全量重建：与 Milvus/Qdrant/LanceDB 行为对齐
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (ValueError, Exception):  # noqa: BLE001
        # collection 不存在时 chromadb 抛 ValueError 或 NotFoundError，按缺省视之
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    # chromadb 要求显式 ids，langchain_chroma 内部也是这么干的
    BATCH = 256
    total = 0
    for i in range(0, len(texts), BATCH):
        bt = texts[i : i + BATCH]
        bm = metadatas[i : i + BATCH]
        bv = vectors[i : i + BATCH]
        collection.add(
            ids=[str(uuid.uuid4()) for _ in bt],
            embeddings=bv,
            documents=bt,
            metadatas=bm,
        )
        total += len(bt)
        print(f"  已插入第 {i // BATCH + 1} 批，共 {len(bt)} 条")
    print(f"已插入 {total} 条文档到 Chroma")


if __name__ == "__main__":
    etl()
