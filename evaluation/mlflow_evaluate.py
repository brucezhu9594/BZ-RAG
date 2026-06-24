"""BZ-RAG MLflow 评测入口：跑 milvus_rag_mlflow 流水线产生 trace，GLM judge 评分，结果落 MLflow。

仿照 evaluate.py（Langfuse + DeepEval + Confident AI 版）：
- 数据集：MLflow EvaluationDataset（先跑 python evaluation/mlflow_build_dataset.py 同步）
- 流水线：api/milvus_rag_mlflow.py，每条样本产出一条带 RETRIEVER span 的 trace
- 指标：与 DeepEval 四指标对齐（faithfulness / answer_relevancy / contextual_precision /
  contextual_recall），自定义 scorer 从 trace 确定性提取检索上下文，再交给 GLM judge 打分

不用 MLflow 内置 RAG scorer 的原因：mlflow 3.11-3.13 对 "openai:/" judge 走原生 adapter，
忽略 OPENAI_API_BASE，流量直打 api.openai.com；make_judge 的 base_url 参数是唯一不引入
litellm 就能指向智谱 OpenAI 兼容端点的官方口子。内置 retrieval scorer 还要求 judge 走
tool-calling 读 trace，GLM 扛不住复杂 meta-prompt，这里改为 Python 侧提取 + 简单 prompt。
"""

import os
import pathlib
import sys
from datetime import datetime, timezone

PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, PROJECT_ROOT)

# 本地 tracking server 走 HTTP，Privoxy 会拦 localhost，先兜底 NO_PROXY。
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

# 串行跑评测，防 GLM API 429（与 DeepEval 流程 max_concurrent=1 + throttle 对齐）。
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.entities import Feedback, SpanType, Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import get_dataset
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer

# import 时连 tracking server 注册实验（与 API 端点共用同一实验，trace/评测同处可比）。
from api.milvus_rag_mlflow import milvus_rag_mlflow_query
from evaluation.mlflow_build_dataset import DATASET_NAME

# judge 用 glm-4-flash（非 thinking）：thinking/M 系列模型扛不住 judge meta-prompt。
JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "glm-4-flash")
# make_judge 的 base_url 是完整 chat 端点 URL（直接作为 POST 目标，不再拼路径）。
_JUDGE_BASE_URL = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"


def _make_glm_judge(name: str, instructions: str, model: str | None = None):
    # make_judge 的英文脚手架（角色设定 + rationale 字段描述）会让 judge 默认用英文写理由，
    # instructions 是唯一能盖过默认的注入点，统一追加中文指令让 judge 的 rationale 出中文。
    # model 可选覆盖 JUDGE_MODEL_ID（如会话级 scorer 用更强的 glm-4-air），不传则沿用默认。
    return make_judge(
        name=name,
        instructions=instructions + "\n\n请始终用中文撰写评分理由（rationale）。",
        feedback_value_type=float,
        model=f"openai:/{model or JUDGE_MODEL_ID}",
        base_url=_JUDGE_BASE_URL,
    )


_FAITHFULNESS = _make_glm_judge(
    "faithfulness",
    "评估回答是否忠实于检索上下文。{{ inputs }} 中的 retrieval_context 是检索到的知识库片段，"
    "{{ outputs }} 是系统的回答。逐条检查回答中的事实陈述能否在检索上下文中找到依据："
    "全部有依据给 1.0，完全无依据或捏造给 0.0，部分有依据按比例打分。"
    "返回 0.0 到 1.0 之间的分数。",
)

_ANSWER_RELEVANCY = _make_glm_judge(
    "answer_relevancy",
    "评估回答与问题的相关性。{{ inputs }} 中的 question 是用户问题，{{ outputs }} 是系统的回答。"
    "回答是否直接、完整地回应了问题本身：完全切题给 1.0，答非所问给 0.0，"
    "部分切题或夹杂无关内容按比例打分。返回 0.0 到 1.0 之间的分数。",
)

_CONTEXTUAL_PRECISION = _make_glm_judge(
    "contextual_precision",
    "评估检索结果的排序质量。{{ inputs }} 中的 question 是用户问题，"
    "retrieval_context 是按检索排名排列的知识库片段列表，"
    "{{ expectations }} 中的 expected_response 是预期答案。"
    "判断每个片段对得出预期答案是否有用，有用的片段应排在无用片段前面："
    "有用片段全部靠前给 1.0，全部靠后给 0.0，混杂时按排序质量打分。"
    "返回 0.0 到 1.0 之间的分数。",
)

_CONTEXTUAL_RECALL = _make_glm_judge(
    "contextual_recall",
    "评估检索上下文对预期答案的覆盖度。{{ inputs }} 中的 retrieval_context 是检索到的"
    "知识库片段，{{ expectations }} 中的 expected_response 是预期答案。"
    "逐条检查预期答案中的信息点能否在检索上下文中找到出处："
    "全部能找到给 1.0，完全找不到给 0.0，部分能找到按比例打分。"
    "返回 0.0 到 1.0 之间的分数。",
)


def _retrieval_context(trace: Trace) -> list[str]:
    spans = trace.search_spans(span_type=SpanType.RETRIEVER)
    if not spans:
        raise ValueError("trace 中没有 RETRIEVER span，无法提取检索上下文")
    return [doc["page_content"] for doc in spans[-1].outputs]


@scorer
def faithfulness(outputs, trace: Trace) -> Feedback:
    return _FAITHFULNESS(
        inputs={"retrieval_context": _retrieval_context(trace)},
        outputs=outputs,
    )


@scorer
def answer_relevancy(inputs, outputs) -> Feedback:
    return _ANSWER_RELEVANCY(inputs={"question": inputs["query"]}, outputs=outputs)


@scorer
def contextual_precision(inputs, expectations, trace: Trace) -> Feedback:
    return _CONTEXTUAL_PRECISION(
        inputs={"question": inputs["query"], "retrieval_context": _retrieval_context(trace)},
        expectations={"expected_response": expectations["expected_response"]},
    )


@scorer
def contextual_recall(expectations, trace: Trace) -> Feedback:
    return _CONTEXTUAL_RECALL(
        inputs={"retrieval_context": _retrieval_context(trace)},
        expectations={"expected_response": expectations["expected_response"]},
    )


def main() -> None:
    try:
        dataset = get_dataset(name=DATASET_NAME)
    except MlflowException as e:
        print(
            f"无法获取 MLflow dataset {DATASET_NAME!r}: {e}\n"
            f"请先跑: python evaluation/mlflow_build_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    run_name = f"mlflow-eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    print(f"评测 run: {run_name}（judge: {JUDGE_MODEL_ID}）")

    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=milvus_rag_mlflow_query,
            scorers=[faithfulness, answer_relevancy, contextual_precision, contextual_recall],
        )

    print(f"run_id: {results.run_id}")
    for metric, value in sorted(results.metrics.items()):
        print(f"  {metric}: {value:.4f}")
    print(f"详情见 {mlflow.get_tracking_uri()} 对应实验的 Evaluations 页。")


if __name__ == "__main__":
    main()
