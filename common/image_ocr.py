"""使用 GLM-4V-Flash 识别网页中图片的文字内容。"""

import os
import urllib.request
from urllib.parse import urljoin

import bs4
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

_client: ZhipuAI | None = None


def _get_client() -> ZhipuAI:
    global _client
    if _client is None:
        _client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))
    return _client


def extract_images_from_page(url: str) -> list[str]:
    """从页面的 help-content-detail 区域提取所有图片 URL。"""
    req = urllib.request.Request(
        url, headers={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", "replace")
    soup = bs4.BeautifulSoup(html, "html.parser")
    detail = soup.find(id="help-content-detail")
    if not detail:
        return []
    return [urljoin(url, img["src"]) for img in detail.find_all("img", src=True)]


def ocr_image(image_url: str) -> str:
    """调用 GLM-4V-Flash 识别单张图片中的文字。"""
    client = _get_client()
    resp = client.chat.completions.create(
        model="glm-4v-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "请提取这张图片中的所有文字内容，保持原始结构，只输出文字不要解释。",
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


def ocr_page_images(page_url: str) -> str:
    """识别页面中所有图片的文字，拼接为一段完整文本。"""
    image_urls = extract_images_from_page(page_url)
    if not image_urls:
        return ""
    texts = []
    for i, img_url in enumerate(image_urls):
        text = ocr_image(img_url)
        if text.strip():
            texts.append(text.strip())
        print(f"  [{i + 1}/{len(image_urls)}] OCR 完成: {img_url[-40:]}")
    return "\n\n".join(texts)
