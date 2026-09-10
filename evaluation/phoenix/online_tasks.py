"""线上评估任务声明的加载与校验。

有意做成 eager 校验、失败即抛：worker 是被调度起来的，没人盯着终端，
一个被静默接受的坏配置会安静地产出几小时垃圾 annotation，比直接崩难查得多。
这与期 1 把判官配置缺失做成 import 期失败是同一个取舍（见 evaluators.py 顶部）。
"""

import pathlib
from typing import Any, NamedTuple

import yaml

DEFAULT_PATH = str(pathlib.Path(__file__).parent / "online_tasks.yaml")

# 线上可用的判官。另外两个（contextual_precision / contextual_recall）的 prompt
# 需要 expected（标准答案），线上没有 ground truth，放进来只会产出全 errored。
ONLINE_EVALUATORS = frozenset({"faithfulness", "answer_relevancy", "refusal_check"})

VALID_CADENCE = frozenset({"continuous", "historical"})
REQUIRED_KEYS = ("name", "project", "span_kind", "evaluators", "sampling_rate", "cadence")


def _check_one(raw: Any, idx: int) -> "OnlineTask":
    if not isinstance(raw, dict):
        raise ValueError(f"第 {idx} 条任务不是 mapping：{raw!r}")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"第 {idx} 条任务缺少必填键：{', '.join(missing)}")

    evaluators = raw["evaluators"]
    if not isinstance(evaluators, list) or not evaluators:
        raise ValueError(f"任务 {raw['name']} 的 evaluators 必须是非空列表")
    unknown = [e for e in evaluators if e not in ONLINE_EVALUATORS]
    if unknown:
        raise ValueError(
            f"任务 {raw['name']} 引用了线上不可用的判官：{', '.join(unknown)}。"
            f"线上没有 ground truth，可用的只有 {', '.join(sorted(ONLINE_EVALUATORS))}"
        )

    rate = raw["sampling_rate"]
    if not isinstance(rate, int | float) or isinstance(rate, bool):
        raise ValueError(f"任务 {raw['name']} 的 sampling_rate 必须是数字，实际 {rate!r}")
    if not 0.0 <= float(rate) <= 1.0:
        raise ValueError(f"任务 {raw['name']} 的 sampling_rate 必须在 [0, 1]，实际 {rate!r}")

    cadence = raw["cadence"]
    if cadence not in VALID_CADENCE:
        raise ValueError(
            f"任务 {raw['name']} 的 cadence 必须是 {'/'.join(sorted(VALID_CADENCE))}，"
            f"实际 {cadence!r}"
        )

    return OnlineTask(
        name=str(raw["name"]),
        project=str(raw["project"]),
        span_kind=str(raw["span_kind"]),
        evaluators=tuple(evaluators),
        sampling_rate=float(rate),
        cadence=cadence,
        window_minutes=int(raw.get("window_minutes", 60)),
    )


class OnlineTask(NamedTuple):
    name: str
    project: str
    span_kind: str
    evaluators: tuple[str, ...]
    sampling_rate: float
    cadence: str
    window_minutes: int


def load_tasks(path: str = DEFAULT_PATH) -> list[OnlineTask]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} 必须是非空的任务列表")

    tasks = [_check_one(item, i) for i, item in enumerate(raw)]

    seen: set[str] = set()
    for t in tasks:
        if t.name in seen:
            raise ValueError(f"任务名重复：{t.name}")
        seen.add(t.name)
    return tasks
