import os
import re
import urllib.request
from urllib.parse import urljoin, urlparse

import bs4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    # 向量数据库chroma
    vector_store = Chroma(
        collection_name="hewa_help_collection",
        embedding_function=embeddings,
        persist_directory="./db",
    )
    # 分批插入（ZhipuAI 单次最多 64 条）
    BATCH_SIZE = 64
    for i in range(0, len(all_splits), BATCH_SIZE):
        batch = all_splits[i:i + BATCH_SIZE]
        vector_store.add_documents(documents=batch)
        print(f"  已插入第 {i // BATCH_SIZE + 1} 批，共 {len(batch)} 条")
    print(f"已插入 {len(all_splits)} 条文档到 Chroma")


if __name__ == "__main__":
    etl()
