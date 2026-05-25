"""LanceDB 知识库检查工具：列表 / schema / 索引 / 样本 / 三种检索冒烟测试。

用法：python -m app.lancedb.inspect
"""

import pathlib
import sys

import lancedb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

COLLECTION_NAME = "hewa_help_collection"
DB_PATH = str(pathlib.Path(__file__).parent / "db")


def _hr(title: str = "") -> None:
    print("\n" + "─" * 8 + (f" {title} " if title else "") + "─" * 60)


def main():
    db = lancedb.connect(DB_PATH)
    _hr("Database")
    print(f"path     : {DB_PATH}")
    print(f"tables   : {db.table_names()}")

    if COLLECTION_NAME not in db.table_names():
        print(f"\n[!] 表 {COLLECTION_NAME} 不存在，请先跑 knowledge_build.py")
        return

    tbl = db.open_table(COLLECTION_NAME)

    _hr("Table")
    print(f"name     : {COLLECTION_NAME}")
    print(f"rows     : {tbl.count_rows()}")
    print(f"version  : {tbl.version}")

    _hr("Schema")
    print(tbl.schema)

    _hr("Indices")
    indices = tbl.list_indices()
    if not indices:
        print("(none)")
    for idx in indices:
        print(f"  {idx}")
        try:
            stats = tbl.index_stats(idx.name)
            print(f"    stats: {stats}")
        except Exception as e:  # noqa: BLE001
            print(f"    stats: <unavailable: {type(e).__name__}: {e}>")

    _hr("Samples (前 3 条)")
    df = tbl.to_pandas().drop(columns=["vector"], errors="ignore").head(3)
    for i, row in df.iterrows():
        text = row.get("text", "")
        print(f"[{i}] source={row.get('source', '')}")
        print(f"    text={text[:80]}{'...' if len(text) > 80 else ''}")

    _hr("Source 分布 (Top 5)")
    src_counts = tbl.to_pandas()["source"].value_counts().head(5)
    for src, n in src_counts.items():
        print(f"  {n:4d}  {src}")

    # 冒烟检索：用一个常见词试试三种检索路径都活着
    probe = "登录"

    _hr(f"FTS 检索冒烟测试：query='{probe}'")
    try:
        # 跟 hybrid_search.py 保持一致，FTS 列已 jieba 预分词，查询也要切
        import jieba

        tokens = " ".join(w for w in jieba.cut(probe) if w.strip())
        results = (
            tbl.search(tokens, query_type="fts")
            .select(["text", "source"])
            .limit(2)
            .to_list()
        )
        for r in results:
            snippet = (r.get("text") or "")[:60]
            print(f"  _score={r.get('_score', 0):.3f}  {snippet}  ({r.get('source', '')})")
        if not results:
            print("  (no hits)")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] FTS 检索失败：{type(e).__name__}: {e}")

    _hr(f"向量检索冒烟测试：query='{probe}'")
    try:
        from langchain_community.embeddings import ZhipuAIEmbeddings

        qv = ZhipuAIEmbeddings(model="embedding-3").embed_query(probe)
        results = (
            tbl.search(qv, vector_column_name="vector")
            .select(["text", "source"])
            .limit(2)
            .to_list()
        )
        for r in results:
            snippet = (r.get("text") or "")[:60]
            print(f"  _distance={r.get('_distance', 0):.3f}  {snippet}  ({r.get('source', '')})")
        if not results:
            print("  (no hits)")
    except Exception as e:  # noqa: BLE001
        print(f"  [!] 向量检索失败：{type(e).__name__}: {e}")

    print()


if __name__ == "__main__":
    main()
