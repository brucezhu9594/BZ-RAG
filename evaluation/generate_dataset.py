"""离线工具：从帮助中心生成 (question, ground_truth) 对，写入 test_dataset.draft.json。"""
import json
import os
import pathlib
import re
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from evaluation.deepeval_judge import GLMJudge

load_dotenv()

DRAFT_PATH = pathlib.Path(__file__).parent / "test_dataset.draft.json"
CONTENT_TRUNC = 3000

PROMPT = """你是禾蛙平台帮助中心的 QA 出题助手。下面是一页帮助文档。
请仅基于这页内容，生成 1 个用户视角的事实类问题，以及完整、忠于原文的答案。

要求：
- question 用自然问句，贴近真实用户提问口吻
- ground_truth 必须用原文表述或紧密改写，不允许加入页外信息
- 不要"根据文档"、"如上所述"这类套话
- 输出严格 JSON，键名固定为 question 和 ground_truth

页面内容：
{content}
"""


def _short_source(source_url: str) -> str:
    m = re.search(r"helpContent/\d+", source_url)
    if not m:
        raise ValueError(f"无法从 source 抽取 helpContent/{{id}}: {source_url}")
    return m.group(0)


def generate_one(doc, llm) -> dict | None:
    """对单页生成 1 个 Q&A item。失败返回 None。"""
    prompt = PROMPT.format(content=(doc.page_content or "")[:CONTENT_TRUNC])
    try:
        msg = llm.invoke([{"role": "user", "content": prompt}])
        parsed = json.loads(GLMJudge._extract_json(msg.content or ""))
        short_src = _short_source(doc.metadata["source"])
    except Exception as e:
        print(f"  生成/解析失败 ({doc.metadata.get('source', '?')}): {e}", file=sys.stderr)
        return None
    return {
        "question": parsed["question"],
        "ground_truth": parsed["ground_truth"],
        "expected_source": short_src,
    }


def main() -> None:
    from common.etl import fetch_help_docs

    llm = ChatOpenAI(
        model=os.environ["MODEL_ID"],
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.3,
        request_timeout=60,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    docs = fetch_help_docs()
    items: list[dict] = []
    for i, doc in enumerate(docs):
        print(f"[{i + 1}/{len(docs)}] {doc.metadata.get('source', '?')}")
        item = generate_one(doc, llm)
        if item:
            items.append(item)
    if not items:
        print("没有任何条目生成成功，退出。", file=sys.stderr)
        sys.exit(1)
    DRAFT_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n完成 {len(items)}/{len(docs)} 项 → {DRAFT_PATH}")
    print("人工 spot-check 后：mv test_dataset.draft.json test_dataset.json")


if __name__ == "__main__":
    main()
