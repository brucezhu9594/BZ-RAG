"""BZ-RAG 评测入口：拉 Langfuse dataset，回放 pipeline 产生 trace，DeepEval 评测后上 Confident AI。"""
import os
import pathlib
import sys
from datetime import datetime, timezone

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, observe

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evaluation.build_dataset import DATASET_NAME
from evaluation.deepeval_judge import MiniMaxJudge

load_dotenv()


@observe(as_type="generation")
def generate_answer(query: str, context: str) -> str:
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


def build_test_cases(dataset, run_name: str) -> list[LLMTestCase]:
    from app.milvus.hybrid_search import _retrieve

    cases: list[LLMTestCase] = []
    for i, item in enumerate(dataset.items):
        question = item.input
        print(f"[{i + 1}/{len(dataset.items)}] 回放: {question}")
        try:
            with item.observe(run_name=run_name) as trace:
                _, docs = _retrieve(question)
                ctx = [d.page_content for d in docs]
                ans = generate_answer(question, "\n\n".join(ctx))
                trace.update(output=ans)
        except Exception as e:
            print(f"  pipeline 抛错，跳过: {e}", file=sys.stderr)
            continue
        cases.append(
            LLMTestCase(
                input=question,
                actual_output=ans,
                expected_output=item.expected_output,
                retrieval_context=ctx,
            )
        )
    return cases


def main() -> None:
    langfuse = Langfuse()
    try:
        dataset = langfuse.get_dataset(DATASET_NAME)
    except Exception as e:
        print(
            f"无法获取 Langfuse dataset {DATASET_NAME!r}: {e}\n"
            f"请先跑: python evaluation/build_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = f"deepeval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"评测 run: {run_name}")

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

    cases = build_test_cases(dataset, run_name)
    langfuse.flush()

    if not cases:
        print("没有可评测的样本，退出。", file=sys.stderr)
        sys.exit(1)

    evaluate(test_cases=cases, metrics=metrics)


if __name__ == "__main__":
    main()
