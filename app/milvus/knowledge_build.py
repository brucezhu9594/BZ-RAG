"""Milvus 入库：dense HNSW + 原生稀疏 BM25 (Function)。"""

import pathlib
import sys

from pymilvus import DataType, Function, FunctionType, MilvusClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from common.etl import prepare_knowledge_base  # noqa: E402

COLLECTION_NAME = "hewa_help_collection"


def etl():
    texts, metadatas, vectors = prepare_knowledge_base()
    if not texts:
        raise RuntimeError("知识库为空")

    client = MilvusClient(uri="http://localhost:19530")
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    dim = len(vectors[0])

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    # analyzer_params 必须显式指定中文分词。不给这个参数时 Milvus 用默认的
    # standard 分析器——它按空白/标点切，对中文等于不分词，标点之间的整段中文
    # 会变成一个 token。后果是 BM25 只在「查询串恰好等于某个标点夹出来的完整
    # 片段」时才命中，实测：查 '蛙贝' 0 条（遍布全库，但总嵌在「扣除5蛙贝」这类
    # 更长的串里）、查 '人力资源供应链内容生态平台' 0 条（这串字面就在库里）。
    # 在 24 条金标问题上，BM25 返回 0 条结果的有 22 条，RRF 融合结果与纯 dense
    # 完全相同的也是 22 条——所谓混合检索在 92% 的查询上是纯 dense，稀疏这一路
    # 白建了，RRFRanker 一直在跟空结果做融合。
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 Function: 自动从 text 生成 sparse_vector
    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="HNSW")
    index_params.add_index(field_name="sparse_vector", metric_type="BM25", index_type="AUTOINDEX")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # 用预算好的 vectors 直接 insert（不再调 ZhipuAI），批大小可以放大
    BATCH = 256
    total = 0
    for i in range(0, len(texts), BATCH):
        data = [
            {"text": t, "vector": v, "source": m.get("source", "")}
            for t, m, v in zip(
                texts[i : i + BATCH],
                metadatas[i : i + BATCH],
                vectors[i : i + BATCH],
                strict=False,
            )
        ]
        client.insert(collection_name=COLLECTION_NAME, data=data)
        total += len(data)
        print(f"  已插入第 {i // BATCH + 1} 批，共 {len(data)} 条")
    print(f"已插入 {total} 条文档到 Milvus")

    # 持久化 + 等索引就绪（dense HNSW + sparse BM25 都建好并 load 到内存）
    client.flush(COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    print(f"索引就绪，load_state={client.get_load_state(COLLECTION_NAME)}")


if __name__ == "__main__":
    etl()
