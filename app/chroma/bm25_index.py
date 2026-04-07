import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_chroma import Chroma

_cache: dict[str, tuple[BM25Okapi, list[Document]]] = {}


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.cut(text) if w.strip()]


def build_bm25_index(vector_store: Chroma) -> tuple[BM25Okapi, list[Document]]:
    """从 Chroma 全量读取文档，构建 BM25 索引。首次构建后缓存，后续直接复用。"""
    collection_name = vector_store._collection.name
    if collection_name in _cache:
        return _cache[collection_name]

    col = vector_store._collection
    all_data = col.get(include=["documents", "metadatas"])

    documents: list[Document] = []
    tokenized_corpus: list[list[str]] = []
    for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
        documents.append(Document(page_content=doc_text, metadata=meta or {}))
        tokenized_corpus.append(_tokenize(doc_text))

    bm25 = BM25Okapi(tokenized_corpus)
    _cache[collection_name] = (bm25, documents)
    return bm25, documents


def invalidate_cache(collection_name: str | None = None):
    """清除缓存。传入 collection_name 清除指定集合，不传则清除全部。"""
    if collection_name:
        _cache.pop(collection_name, None)
    else:
        _cache.clear()
