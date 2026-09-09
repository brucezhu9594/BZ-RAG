"""判官（Arize 说的 evaluator template 层）——离线门禁与线上评估共用同一份。

prompt 沿用 evaluation/mlflow_evaluate.py 里已经调过中文 rationale 的那几条，
但输出形态从"让模型吐 0-1 浮点"改成分类标签 + 分数映射：Phoenix 官方明确建议
分类优于数值评分（模型的数值推理不稳，且与人类判断相关性更差）。

每个判官除了返回插件要的 dict，还向 acceptance 登记一条记录——判官抛异常时
登记成 errored（第三态），不静默丢弃、也不当成 0 分。这一层保护必须包住"从
output/input 里取值"这个提取动作本身（见 _run），否则 output 为 None（测试在
log_output 之前就崩了）时会抛在保护范围外，该 case 就从统计里静默消失，而不是
落进 errored 第三态。

配置（JUDGE_OPENAI_API_KEY / JUDGE_OPENAI_BASE_URL / JUDGE_MODEL_ID）有意在
import 期就检查、缺失就快速失败（Ruling R20）：Task 6 的 conftest.py 会在
pytest 收集阶段 import 本模块，缺配置在收集阶段就崩比跑完全部 case × 判官的
上百次调用后才发现全 errored 便宜得多；这里只保证报错可读（写清缺哪个变量、
去哪设），不是延后检查的时机。注意这与"key 写错但存在"是两种不同的失败模式：
LLM(...) 构造不打网络，错 key 不会在 import 期暴露，会在每次调用时变成
errored 第三态、由 acceptance 的 max_error_rate 闸门抓住——两者都对，不需要
统一成同一种失败时机。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from dotenv import load_dotenv

load_dotenv()

from evaluation.phoenix import acceptance
from phoenix.evals import ClassificationEvaluator
from phoenix.evals.llm import LLM

_CHOICES = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}

_REQUIRED = ("JUDGE_OPENAI_API_KEY", "JUDGE_OPENAI_BASE_URL", "JUDGE_MODEL_ID")
_missing = [k for k in _REQUIRED if not os.environ.get(k)]
if _missing:
    raise RuntimeError(
        f"判官配置缺失：{', '.join(_missing)}。请在仓库根的 .env 里设置"
        f"（见 .env.example 的 Phoenix 评估判官段）。"
        f"注意本模块有意在 import 期就失败——缺配置时立刻红比跑完上百次判官调用再红便宜。"
    )

_llm = LLM(
    provider="openai",
    model=os.environ["JUDGE_MODEL_ID"],
    base_url=os.environ["JUDGE_OPENAI_BASE_URL"],
    api_key=os.environ["JUDGE_OPENAI_API_KEY"],
)

_FAITHFULNESS_T = """判断回答是否忠实于检索上下文。
[检索上下文]
{reference}

[回答]
{output}

逐条检查回答中的事实陈述能否在检索上下文中找到依据。
correct = 全部有依据；incorrect = 完全无依据或与上下文冲突；partial = 部分有依据。
请用中文撰写理由。"""

_RELEVANCY_T = """判断回答与问题的相关性。
[问题]
{input}

[回答]
{output}

correct = 直接完整地回应了问题；incorrect = 答非所问；partial = 部分切题或夹杂无关内容。
请用中文撰写理由。"""

_PRECISION_T = """判断检索结果的排序质量。
[问题]
{input}

[按检索排名排列的知识库片段]
{reference}

[预期答案]
{expected}

对得出预期答案有用的片段应排在无用片段前面。
correct = 有用片段全部靠前；incorrect = 全部靠后；partial = 混杂。
请用中文撰写理由。"""

_RECALL_T = """判断检索上下文对预期答案的覆盖度。
[检索上下文]
{reference}

[预期答案]
{expected}

逐条检查预期答案中的信息点能否在检索上下文中找到出处。
correct = 全部能找到；incorrect = 完全找不到；partial = 部分能找到。
请用中文撰写理由。"""

_REFUSAL_T = """判断回答是不是无效答复（护栏检查）。
[问题]
{input}

[回答]
{output}

ok = 给出了实质性回答；refused = 拒答或声称无法回答；empty = 空答、纯客套、无任何信息量。
请用中文撰写理由。"""


def _classifier(name: str, template: str, choices: dict[str, float]) -> ClassificationEvaluator:
    # temperature=0：ClassificationEvaluator 的 **kwargs 会存进 invocation_parameters
    # 并透传到 llm.generate_classification(...) → generate_object(...) → SDK
    # （已读 phoenix/evals/evaluators.py:567,752 与 llm/wrapper.py:310,350 确认）。
    #
    # 诚实标注：这不是判官漂移的主因。实测把三条失败 case 的原始 payload 逐字回放
    # 各 6 次，默认温度与 temperature=0 的标签**完全相同**（且各自 6/6 稳定），
    # 但其中两条的回放结果与跑批时记录的标签不一致。剩下的嫌疑是设计文档 §8 已
    # 列为风险的判官供应商漂移（minimax-m3 经 Vercel AI Gateway 路由，实测落
    # fireworks，另有 minimax/nebius/gmicloud/morph 四个 fallback），25 分钟的
    # 跑批里会漂到不同 provider，短时回放则命中同一个。
    # 固定温度消掉的是采样这一个变量，值得设，但别指望它解决方差。
    return ClassificationEvaluator(
        name=name,
        prompt_template=template,
        llm=_llm,
        choices=choices,
        direction="maximize",
        temperature=0,
    )


_JUDGES = {
    "faithfulness": _classifier("faithfulness", _FAITHFULNESS_T, _CHOICES),
    "answer_relevancy": _classifier("answer_relevancy", _RELEVANCY_T, _CHOICES),
    "contextual_precision": _classifier("contextual_precision", _PRECISION_T, _CHOICES),
    "contextual_recall": _classifier("contextual_recall", _RECALL_T, _CHOICES),
    # 护栏型：稀有失败，期 2 里它的 sampling_rate 固定 1.0
    "refusal_check": _classifier(
        "refusal_check", _REFUSAL_T, {"empty": 0.0, "refused": 0.0, "ok": 1.0}
    ),
}


def _run(name: str, build_payload) -> dict:
    """跑一个判官并向 acceptance 登记。

    build_payload 是个零参闭包：**提取动作也必须在 try 内**，否则 output 为 None
    （测试在 log_output 之前就崩了）时会抛在保护范围外，该 case 就从统计里静默消失，
    而不是登记成 errored 第三态。异常记成 errored（第三态），不吞、也不当 0 分。
    """
    try:
        payload = build_payload()
        (score,) = _JUDGES[name].evaluate(payload)
    except Exception as e:  # noqa: BLE001 —— 判官失败必须成为可见的第三态
        acceptance.record(name, None, error=f"{type(e).__name__}: {e}")
        return {"name": name, "score": None, "label": "errored", "explanation": str(e)}
    acceptance.record(name, score.score, label=score.label)
    return {
        "name": name,
        "score": score.score,
        "label": score.label,
        "explanation": score.explanation,
    }


def _unpack(output) -> tuple[str, str]:
    """测试用 log_output 记的是 {"answer":..., "contexts":[...]}；拆成 (答案, 拼好的上下文)。"""
    answer = output["answer"]
    contexts = output.get("contexts") or []
    reference = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    return answer, reference


# 参数名由 Phoenix 插件的「按参数名绑定」约定决定：
#   output   —— log_output 记进去的值
#   input    —— 该 case 的 parametrize 字段组成的 mapping，即 {"question": ..., "expected": ...}
#   expected —— 名为 expected 的 parametrize 字段的值（这里是 ground truth 字符串）
# 一律带 **_ 兜住不消费的字段（trace_id、metadata、example 等）。


def faithfulness(output, **_) -> dict:
    def build_payload():
        answer, reference = _unpack(output)
        return {"output": answer, "reference": reference}

    return _run("faithfulness", build_payload)


def answer_relevancy(output, input=None, **_) -> dict:  # noqa: A002 —— 参数名由 Phoenix 约定
    def build_payload():
        answer, _ref = _unpack(output)
        return {"output": answer, "input": input["question"]}

    return _run("answer_relevancy", build_payload)


def contextual_precision(output, input=None, expected=None, **_) -> dict:  # noqa: A002
    def build_payload():
        _answer, reference = _unpack(output)
        return {"input": input["question"], "reference": reference, "expected": expected}

    return _run("contextual_precision", build_payload)


def contextual_recall(output, expected=None, **_) -> dict:
    def build_payload():
        _answer, reference = _unpack(output)
        return {"reference": reference, "expected": expected}

    return _run("contextual_recall", build_payload)


def refusal_check(output, input=None, **_) -> dict:  # noqa: A002
    def build_payload():
        answer, _ref = _unpack(output)
        return {"output": answer, "input": input["question"]}

    return _run("refusal_check", build_payload)


EVALUATORS = [
    faithfulness,
    answer_relevancy,
    contextual_precision,
    contextual_recall,
    refusal_check,
]
