"""LanceDB 入库：dense 向量 + 原生 FTS (Tantivy BM25)。

中文 FTS 取巧点：入库前 jieba 切词存到 `text_tokenized` 列，FTS 索引建在该列，
base_tokenizer=whitespace，关掉所有英文专用处理。原文 `text` 留作召回展示。
"""

import pathlib
import sys

import jieba
import lancedb
import pyarrow as pa

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from common.etl import prepare_knowledge_base  # noqa: E402

COLLECTION_NAME = "hewa_help_collection"
DB_PATH = str(pathlib.Path(__file__).parent / "db")


def tokenize_for_fts(text: str) -> str:
    return " ".join(w for w in jieba.cut(text) if w.strip())


def etl():
    texts, metadatas, vectors = prepare_knowledge_base()
    if not texts:
        raise RuntimeError("知识库为空")

    db = lancedb.connect(DB_PATH)
    dim = len(vectors[0])

    schema = pa.schema(
        [
            pa.field("text", pa.string()),
            pa.field("text_tokenized", pa.string()),
            pa.field("source", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
        ]
    )
    tbl = db.create_table(COLLECTION_NAME, schema=schema, mode="overwrite")

    rows = [
        {
            "text": t,
            "text_tokenized": tokenize_for_fts(t),
            "source": m.get("source", ""),
            "vector": v,
        }
        for t, m, v in zip(texts, metadatas, vectors, strict=False)
    ]
    tbl.add(rows)
    print(f"已插入 {len(rows)} 条文档到 LanceDB")

    # 建 FTS (BM25) 索引：因已 jieba 预分词，用 whitespace tokenizer + 关掉英文处理
    tbl.create_fts_index(
        "text_tokenized",
        base_tokenizer="whitespace",
        language="English",
        stem=False,
        remove_stop_words=False,
        ascii_folding=False,
        lower_case=False,
        replace=True,
    )
    print("FTS (BM25) 索引构建完成")

    # 向量索引：IVF_PQ。当前数据量很小（<1k），PQ 训练样本不足、精度会差，
    # 但建出来便于与其它向量库索引行为对齐。生产规模建议 num_partitions = num_rows/4096，
    # num_sub_vectors = dim/8（dim=2048 时取 256），届时重建即可。
    n_rows = len(rows)
    num_partitions = max(1, min(int(n_rows**0.5), 64))
    tbl.create_index(
        metric="cosine",
        vector_column_name="vector",
        num_partitions=num_partitions,
        num_sub_vectors=16,  # 2048 / 16 = 128 维/子量化器
        replace=True,
    )
    print(f"向量索引 IVF_PQ 构建完成 (num_partitions={num_partitions}, num_sub_vectors=16)")


if __name__ == "__main__":
    etl()
