"""单轮 RAG 离线门禁。

suite → Phoenix dataset，case → example，断言结果 → 保留的 pass annotation，
pytest 退出码即 CI 门禁。挂在 marker 上的 evaluator 失败只降级成 warning，
真正让测试红的是断言；聚合阈值由 conftest 的 acceptance 在 sessionfinish 判。

parametrize 的字段名是有讲究的：插件把它们整体作为 example 的 input 传给判官
（即 input == {"question": ..., "expected": ...}），并把名为 expected 的字段
额外单独绑到判官的 expected 形参。所以字段必须叫 question / expected，
不能图省事传一个 Case 对象——那样 input 会变成 {"case": Case(...)}。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

# 门禁要的是可复现的基线，所以被测管线必须确定性——必须在 import
# api.milvus_rag_phoenix 之前设好（那边在调用时读，这里 setdefault 只是保证
# 默认值；想跑生产采样行为就在外面显式导出 GENERATION_TEMPERATURE=0.7）。
# 生产入口 api/milvus_rag.py 不受影响，仍是 0.7。
os.environ.setdefault("GENERATION_TEMPERATURE", "0")

import pytest

from api.milvus_rag_phoenix import milvus_rag_phoenix_query_with_context
from evaluation.phoenix.cases import CASES, SMOKE_IDS
from evaluation.phoenix.evaluators import EVALUATORS
from phoenix.client.pytest import log_output

_PARAMS = [
    pytest.param(
        c.question,
        c.ground_truth,
        id=c.example_id,
        marks=[pytest.mark.smoke] if c.example_id in SMOKE_IDS else [],
    )
    for c in CASES
]


@pytest.mark.phoenix(
    dataset=os.environ.get("PHOENIX_TEST_DATASET", "bz-rag-golden"),
    dataset_description="BZ-RAG 黄金集，来自 evaluation/test_dataset.json",
    experiment_description="Phoenix 离线门禁（单轮）",
    experiment_metadata={"judge": os.environ.get("JUDGE_MODEL_ID", "")},
    evaluators=EVALUATORS,
    repetitions=int(os.environ.get("EVAL_REPETITIONS", "1")),
)
@pytest.mark.parametrize("question,expected", _PARAMS)
def test_rag_single_turn(question, expected):
    answer, contexts = milvus_rag_phoenix_query_with_context(question)
    log_output({"answer": answer, "contexts": contexts})
    # 硬断言只管"管线有没有产出"，质量由 acceptance 的聚合阈值管。
    # 一条 case 答得差不会立刻红，但整体质量掉下去会红——正是 Arize 的取舍。
    assert answer and answer.strip(), "管线返回了空答案"
    assert contexts, "检索没有返回任何上下文"
