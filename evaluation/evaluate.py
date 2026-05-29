"""BZ-RAG 评测入口：用 Langfuse run_experiment 跑 milvus_hybrid pipeline 产生 trace，DeepEval 评测后上 Confident AI。"""
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

    # Side-channel collection of (input, output, retrieval_context) per item
    # so DeepEval can evaluate after the experiment finishes.
    collected: list[dict] = []

    from app.milvus.hybrid_search import _retrieve

    def task(*, item, **kwargs):
        """Pipeline runner. run_experiment passes DatasetItem via keyword arg 'item'."""
        question = item.input
        expected = item.expected_output

        print(f"  pipeline: {question}")
        _, docs = _retrieve(question)
        ctx = [d.page_content for d in docs]
        ans = generate_answer(question, "\n\n".join(ctx))
        collected.append(
            {
                "input": question,
                "actual_output": ans,
                "expected_output": expected,
                "retrieval_context": ctx,
            }
        )
        return ans

    # Run experiment — Langfuse handles tracing and dataset run linkage.
    # Pass evaluators=[] because we use DeepEval as a separate pass after.
    dataset.run_experiment(
        name=run_name,
        run_name=run_name,
        task=task,
        evaluators=[],
    )

    langfuse.flush()

    # Build DeepEval cases from collected side data.
    cases: list[LLMTestCase] = []
    for c in collected:
        if c["expected_output"] is None:
            # Skip items without expected output
            continue
        cases.append(
            LLMTestCase(
                input=c["input"],
                actual_output=c["actual_output"],
                expected_output=c["expected_output"],
                retrieval_context=c["retrieval_context"],
            )
        )

    if not cases:
        print("没有可评测的样本，退出。", file=sys.stderr)
        sys.exit(1)

    evaluate(test_cases=cases, metrics=metrics)


if __name__ == "__main__":
    main()
