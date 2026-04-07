import os
import re
import urllib.request
from urllib.parse import urljoin, urlparse

import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, DataType, Function, FunctionType

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
)

HELP_INDEX_URL = "https://cms.hewa.cn/content/mian/helpContent"
_HELP_CONTENT_ID_RE = re.compile(r"/content/mian/helpContent/(\d+)/?$", re.IGNORECASE)

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

    # E-提取（加载）文档（该站为 Nuxt SSR，正文在 content-header / help-content-detail）
    loader = WebBaseLoader(
        web_paths=web_paths,
        requests_per_second=2,
        continue_on_failure=True,
        **_help_page_loader_kwargs(),
    )
    docs = loader.load()
    print(f"共加载 {len(docs)} 个页面（期望 {len(web_paths)} 个 URL）")

    # 对纯图片页面用 GLM-4V-Flash OCR 补充文本
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
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
    client = MilvusClient(uri="http://localhost:19530")

    collection_name = "hewa_help_collection"
    # 如已存在则先删除，保证全量重建
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # 先 embed 一条拿到维度
    sample_vec = embeddings.embed_query(all_splits[0].page_content)
    dim = len(sample_vec)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=65535,
                     enable_analyzer=True, enable_match=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 Function: 自动从 text 生成 sparse_vector
    bm25_fn = Function(
        name="text_bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_fn)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE", index_type="FLAT")
    index_params.add_index(field_name="sparse_vector", metric_type="BM25",
                           index_type="AUTOINDEX")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    # 分批 embed + 插入（ZhipuAI 单次最多 64 条）
    BATCH_SIZE = 64
    texts = [doc.page_content for doc in all_splits]
    metadatas = [doc.metadata for doc in all_splits]
    total = 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]
        batch_vectors = embeddings.embed_documents(batch_texts)
        data = [
            {"text": t, "vector": v, "source": m.get("source", "")}
            for t, v, m in zip(batch_texts, batch_vectors, batch_metas)
        ]
        client.insert(collection_name=collection_name, data=data)
        total += len(data)
        print(f"  已插入第 {i // BATCH_SIZE + 1} 批，共 {len(data)} 条")
    print(f"已插入 {total} 条文档到 Milvus")


if __name__ == "__main__":
    etl()
