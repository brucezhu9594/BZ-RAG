"""BM25 稀疏向量工具：入库时构建并保存词汇表/IDF，查询时加载复用。"""

import json
import math
import pathlib
from collections import Counter

import jieba
from qdrant_client import models

_BM25_META_PATH = pathlib.Path(__file__).parent / "bm25_meta.json"

K1 = 1.5
B = 0.75


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def build_and_save(texts: list[str]) -> list[models.SparseVector]:
    """根据语料构建 BM25 元数据并保存到文件，返回每篇文档的稀疏向量。"""
    tokenized = [_tokenize(t) for t in texts]
    n_docs = len(tokenized)

    # 构建词汇表（确定性排序：按首次出现顺序）
    vocab: dict[str, int] = {}
    for tokens in tokenized:
        for w in tokens:
            if w not in vocab:
                vocab[w] = len(vocab)

    # 计算 DF
    df: dict[str, int] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    # 计算 IDF
    idf: dict[str, float] = {}
    for word, freq in df.items():
        idf[word] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    # 平均文档长度
    avg_dl = sum(len(t) for t in tokenized) / n_docs if n_docs else 1

    # 保存元数据
    meta = {"vocab": vocab, "idf": idf, "avg_dl": avg_dl, "n_docs": n_docs}
    _BM25_META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # 生成文档稀疏向量
    return [_doc_sparse_vector(tokens, vocab, idf, avg_dl) for tokens in tokenized]


def _doc_sparse_vector(
    tokens: list[str],
    vocab: dict[str, int],
    idf: dict[str, float],
    avg_dl: float,
) -> models.SparseVector:
    """为单篇文档生成 BM25 稀疏向量。"""
    dl = len(tokens)
    tf = Counter(tokens)
    indices = []
    values = []
    for word, count in tf.items():
        if word not in vocab:
            continue
        tf_norm = (count * (K1 + 1)) / (count + K1 * (1 - B + B * dl / avg_dl))
        score = idf.get(word, 0) * tf_norm
        if score > 0:
            indices.append(vocab[word])
            values.append(score)
    return models.SparseVector(indices=indices, values=values)


def query_sparse_vector(query: str) -> models.SparseVector:
    """加载保存的 BM25 元数据，为查询生成稀疏向量（TF 归一化方式与入库一致）。"""
    if not _BM25_META_PATH.exists():
        raise FileNotFoundError(
            f"BM25 元数据文件不存在: {_BM25_META_PATH}，请先运行 knowledge_build.py"
        )

    meta = json.loads(_BM25_META_PATH.read_text(encoding="utf-8"))
    vocab: dict[str, int] = meta["vocab"]
    idf: dict[str, float] = meta["idf"]
    avg_dl: float = meta["avg_dl"]

    tokens = _tokenize(query)
    # 查询也用与文档相同的 BM25 TF 归一化
    return _doc_sparse_vector(tokens, vocab, idf, avg_dl)
