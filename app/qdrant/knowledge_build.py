import os
import re
import sys
import pathlib
import urllib.request
from urllib.parse import urljoin, urlparse

import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.qdrant.bm25 import build_and_save

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

HELP_INDEX_URL = "https://cms.hewa.cn/content/mian/helpContent"
_HELP_CONTENT_ID_RE = re.compile(r"/content/mian/helpContent/(\d+)/?$", re.IGNORECASE)

COLLECTION_NAME = "hewa_help_collection"

load_dotenv()


def discover_help_content_article_urls(index_url: str = HELP_INDEX_URL) -> list[str]:
    """从帮助中心索引页解析所有 helpContent/{{文档id}} 链接并去重、按 id 数字排序。"""
    req = urllib.request.Request(
        index_url, headers={"User-Agent": os.environ["USER_AGENT"]}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        enc = resp.headers.get_content_charset() or "utf-8"
        html = resp.read().decode(enc, "replace")
    soup = bs4.BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(index_url, a["href"].strip())
        parsed = urlparse(full)
        if parsed.netloc and urlparse(index_url).netloc != parsed.netloc:
            continue
        m = _HELP_CONTENT_ID_RE.search(parsed.path)
        if not m:
            continue
        doc_id = m.group(1)
        normalized = f"{parsed.scheme}://{parsed.netloc}/content/mian/helpContent/{doc_id}"
        seen.add(normalized)
    return sorted(seen, key=lambda u: int(_HELP_CONTENT_ID_RE.search(urlparse(u).path).group(1)))


def _help_page_loader_kwargs():
    return dict(
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                id=lambda i: i in ("content-header", "help-content-detail")
            )
        ),
    )


def etl():
    web_paths = discover_help_content_article_urls()
    if not web_paths:
        raise RuntimeError("未从索引页解析到任何 helpContent/{{id}} 链接，请检查页面结构或网络。")

    # E-提取
    loader = WebBaseLoader(
        web_paths=web_paths,
        requests_per_second=2,
        continue_on_failure=True,
        **_help_page_loader_kwargs(),
    )
    docs = loader.load()
    print(f"共加载 {len(docs)} 个页面（期望 {len(web_paths)} 个 URL）")

    # 对纯图片页面用 GLM-4V-Flash OCR 补充文本
    from common.image_ocr import ocr_page_images
    for d in docs:
        src = d.metadata.get("source", "")
        n = len(d.page_content or "")
        if n < 50:
            print(f"  {src}  ->  {n} 字符 (文本过短，尝试 OCR 图片)")
            ocr_text = ocr_page_images(src)
            if ocr_text:
                d.page_content = ocr_text
                print(f"    OCR 后: {len(ocr_text)} 字符")
            else:
                print(f"    OCR 未提取到内容")
        else:
            print(f"  {src}  ->  {n} 字符")

    # T-加工
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"切分后共 {len(all_splits)} 段")

    # L-加载（向量化存储）
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    client = QdrantClient(host="localhost", port=6333)

    # 如已存在则先删除
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    # 先 embed 一条拿到维度
    sample_vec = embeddings.embed_query(all_splits[0].page_content)
    dim = len(sample_vec)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        sparse_vectors_config={"bm25": models.SparseVectorParams()},
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # 分批插入（ZhipuAI 单次最多 64 条）
    BATCH_SIZE = 64
    total = 0
    for i in range(0, len(all_splits), BATCH_SIZE):
        batch = all_splits[i:i + BATCH_SIZE]
        vector_store.add_documents(documents=batch)
        total += len(batch)
        print(f"  已插入第 {i // BATCH_SIZE + 1} 批，共 {len(batch)} 条")

    # 补充 BM25 稀疏向量（构建并保存词汇表/IDF 到 bm25_meta.json）
    texts = [doc.page_content for doc in all_splits]
    sparse_vectors = build_and_save(texts)

    # 获取已插入的 point ids
    all_points = []
    offset = None
    while True:
        result = client.scroll(collection_name=COLLECTION_NAME, limit=100, offset=offset)
        points, offset = result
        all_points.extend(points)
        if offset is None:
            break

    # 按插入顺序更新 sparse 向量
    for point, sparse_vec in zip(all_points, sparse_vectors):
        client.update_vectors(
            collection_name=COLLECTION_NAME,
            points=[models.PointVectors(id=point.id, vector={"bm25": sparse_vec})],
        )

    print(f"已插入 {total} 条文档到 Qdrant（含 BM25 稀疏向量）")
    client.close()


if __name__ == "__main__":
    etl()
