"""帮助中心 ETL 公共流程：fetch → ocr → split → embed。

供 chroma / milvus / qdrant / lancedb 四个 knowledge_build 复用，
默认开 parquet 磁盘缓存，第一次构建走完整流程，后续构建（即使切到别的向量库）
直接读缓存秒级返回，避免重复爬网页 + OCR + 调智谱 embedding API。

强制刷新：传 use_cache=False，或手动删 db/_kb_cache.parquet。
"""

import os
import re
import urllib.request
from pathlib import Path
from urllib.parse import urljoin, urlparse

import bs4
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

HELP_INDEX_URL = "https://cms.hewa.cn/content/mian/helpContent"
_HELP_CONTENT_ID_RE = re.compile(r"/content/mian/helpContent/(\d+)/?$", re.IGNORECASE)

# 缓存放仓库根目录的 db/ 下，三家向量库的本地存储也在此处
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "db" / "_kb_cache.parquet"

EMBED_MODEL = "embedding-3"
EMBED_BATCH_SIZE = 64  # ZhipuAI 单次最大 64
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


def discover_help_content_article_urls(index_url: str = HELP_INDEX_URL) -> list[str]:
    """从帮助中心索引页解析所有 helpContent/{{文档id}} 链接并去重、按 id 数字排序。"""
    req = urllib.request.Request(index_url, headers={"User-Agent": os.environ["USER_AGENT"]})
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
            parse_only=bs4.SoupStrainer(id=lambda i: i in ("content-header", "help-content-detail"))
        ),
    )


def fetch_help_docs() -> list[Document]:
    """爬取帮助中心所有文章，对正文过短的页面用 GLM-4V-Flash OCR 补全。"""
    web_paths = discover_help_content_article_urls()
    if not web_paths:
        raise RuntimeError("未从索引页解析到任何 helpContent/{{id}} 链接，请检查页面结构或网络。")

    loader = WebBaseLoader(
        web_paths=web_paths,
        requests_per_second=2,
        continue_on_failure=True,
        **_help_page_loader_kwargs(),
    )
    docs = loader.load()
    print(f"共加载 {len(docs)} 个页面（期望 {len(web_paths)} 个 URL）")

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
                print("    OCR 未提取到内容")
        else:
            print(f"  {src}  ->  {n} 字符")

    return docs


def split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    print(f"切分后共 {len(chunks)} 段")
    return chunks


def embed_chunks(chunks: list[Document]) -> list[list[float]]:
    """分批调用 ZhipuAI embedding（单次上限 64）。"""
    embeddings = ZhipuAIEmbeddings(model=EMBED_MODEL)
    texts = [c.page_content for c in chunks]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))
        print(f"  embed 第 {i // EMBED_BATCH_SIZE + 1} 批，共 {len(batch)} 条")
    return vectors


def prepare_knowledge_base(
    use_cache: bool = True,
    cache_path: Path = _DEFAULT_CACHE_PATH,
) -> tuple[list[str], list[dict], list[list[float]]]:
    """一站式：fetch → ocr → split → embed。

    返回 (texts, metadatas, vectors)，三者按位置对齐，可直接喂给任一向量库。
    use_cache=True 时落盘 parquet；下次跑（即使切到别的向量库）直接读缓存。
    """
    if use_cache and cache_path.exists():
        print(f"[cache] 从 {cache_path} 读取已构建的知识库")
        table = pq.read_table(cache_path)
        texts = table.column("text").to_pylist()
        sources = table.column("source").to_pylist()
        vectors = table.column("vector").to_pylist()
        metadatas = [{"source": s} for s in sources]
        dim = len(vectors[0]) if vectors else 0
        print(f"[cache] 命中：{len(texts)} 段，向量维度 {dim}")
        return texts, metadatas, vectors

    docs = fetch_help_docs()
    chunks = split_docs(docs)
    vectors = embed_chunks(chunks)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        sources = [m.get("source", "") for m in metadatas]
        table = pa.table(
            {"text": texts, "source": sources, "vector": vectors},
            schema=pa.schema(
                [
                    pa.field("text", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("vector", pa.list_(pa.float32())),
                ]
            ),
        )
        pq.write_table(table, cache_path)
        print(f"[cache] 已写入 {cache_path}")

    return texts, metadatas, vectors


def clear_cache(cache_path: Path = _DEFAULT_CACHE_PATH) -> None:
    """删除知识库缓存，下次 prepare_knowledge_base 会重新走全流程。"""
    if cache_path.exists():
        cache_path.unlink()
        print(f"[cache] 已删除 {cache_path}")


if __name__ == "__main__":
    # 直接跑此模块 = 预热缓存
    prepare_knowledge_base()
