"""BZ-RAG 评测入口：用 deepeval 评测 milvus_hybrid 方案。"""
import json
import os
import pathlib
import sys

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evaluation.deepeval_judge import MiniMaxJudge

load_dotenv()

DATASET_PATH = pathlib.Path(__file__).parent / "test_dataset.json"


def generate_answer(query: str, context: str) -> str:
    """与 hybrid_search.rag 等价的回答生成（不带 history）。"""
    llm = ChatOpenAI(model=os.environ["MODEL_ID"], temperature=0.7, request_timeout=60)
    system = (
        "你是一个知识库检索助手。"
        "下面「检索结果」来自知识库片段，请仅依据这些内容回答用户问题。"
        "如果检索结果不足以回答，请明确说明知识库中没有相关信息，不要编造。"
        f"\n\n--- 检索结果 ---\n{context}"
    )
    msg = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
    )
    return msg.content or ""


def build_test_cases(dataset: list[dict]) -> list[LLMTestCase]:
    from app.milvus.hybrid_search import _retrieve

    cases: list[LLMTestCase] = []
    for i, item in enumerate(dataset):
        q = item["question"]
        print(f"[{i + 1}/{len(dataset)}] 构建样本: {q}")
        try:
            _, docs = _retrieve(q)
        except Exception as e:
            print(f"  检索失败，跳过: {e}", file=sys.stderr)
            continue
        ctx = [d.page_content for d in docs]
        try:
            ans = generate_answer(q, "\n\n".join(ctx))
        except Exception as e:
            print(f"  生成失败，跳过: {e}", file=sys.stderr)
            continue
        cases.append(
            LLMTestCase(
                input=q,
                actual_output=ans,
                expected_output=item["ground_truth"],
                retrieval_context=ctx,
            )
        )
    return cases


def main() -> None:
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    judge = MiniMaxJudge()
    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        ContextualPrecisionMetric(model=judge, threshold=0.7),
        ContextualRecallMetric(model=judge, threshold=0.7),
    ]

    if not os.environ.get("CONFIDENT_API_KEY"):
        print(
            "[提示] 未设置 CONFIDENT_API_KEY，仅本地评测。"
            "如需云端报告：deepeval login --confident-api-key=<key>"
        )

    test_cases = build_test_cases(dataset)
    if not test_cases:
        print("没有可评测的样本，退出。", file=sys.stderr)
        sys.exit(1)

    evaluate(test_cases=test_cases, metrics=metrics)


if __name__ == "__main__":
    main()
