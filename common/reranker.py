"""Reranker：使用 BGE-Reranker 本地模型对检索结果重排序。"""

import os

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "hub",
    "models--BAAI--bge-reranker-base",
    "snapshots", "2cfc18c9415c912f9d8155881c133215df768a70",
)
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_PATH, local_files_only=True)
        _model.eval()
    return _tokenizer, _model


def rerank(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """使用 BGE-Reranker 对文档按与查询的相关性重排序。"""
    if not docs:
        return []

    tokenizer, model = _load_model()

    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = tokenizer(
            pairs, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        scores = model(**inputs, return_dict=True).logits.view(-1).float().tolist()

    # 按 score 降序排序
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in scored_docs]

    return result[:top_n] if top_n else result
