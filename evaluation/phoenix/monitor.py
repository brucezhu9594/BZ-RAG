"""线上监控与决策：把 span annotation 聚合成 rollback / hold / promote 三态。

**复用期 1 的聚合引擎**：criteria.yaml 的 online 段与 offline 段用的是同一套
阈值语言、同一份 acceptance.py 实现。区别只在输入来源——离线是 experiment run
的 annotation，线上是 project span 的 annotation。这是 spec 强调的一致性。

**本模块只判定，不动作。** 权重归零由 rollback.py 执行，晋级放行由
promote-stable.yml 的前置门做。决策与执行分开，才能在 CI 之外单独跑一次
monitor 看当前状态而不产生任何副作用。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import pathlib
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

from evaluation.phoenix import acceptance
from evaluation.phoenix.online_worker import otel_span_id

CRITERIA_PATH = str(pathlib.Path(__file__).parent / "criteria.yaml")
PULL_LIMIT = 1000

# 退出码给 workflow 用。0/1 是惯例的成功/失败，hold 单独占 2——
# CI 步骤要能区分"不达标"和"还判不了"，两者的处置完全不同。
EXIT_PROMOTE = 0
EXIT_ROLLBACK = 1
EXIT_HOLD = 2


class Decision(NamedTuple):
    state: str  # "rollback" | "hold" | "promote"
    outcomes: list[Any]
    scoreboard: str
    reason: str


def decide(outcomes: list[Any]) -> Decision:
    """三态判定。**hold 优先于 rollback。**

    判"样本不足"用 samples < criterion.min_samples，不匹配 reason 字符串——
    那是给人看的显示文本，改一个字就会把判定改坏。
    """
    scoreboard = acceptance.format_scoreboard(outcomes) if outcomes else "(无 criteria)"

    if not outcomes:
        return Decision("hold", outcomes, scoreboard, "没有任何 criterion 可判")

    short = [o for o in outcomes if o.samples < o.criterion.min_samples]
    if short:
        detail = "；".join(
            f"{o.criterion.annotation} 只有 {o.samples} 条，需要 {o.criterion.min_samples}"
            for o in short
        )
        # 样本不足意味着"还判不了"，此时任何质量结论都不可信——
        # 贸然 rollback 会把好版本也打回去，所以 hold 必须优先于 rollback。
        return Decision("hold", outcomes, scoreboard, f"样本不足：{detail}")

    failed = [o for o in outcomes if not o.passed]
    if failed:
        detail = "；".join(
            f"{o.criterion.annotation}/{o.criterion.metric} 实测 {o.observed} 未达 {o.required}"
            for o in failed
        )
        return Decision("rollback", outcomes, scoreboard, f"未达标：{detail}")

    return Decision("promote", outcomes, scoreboard, "全部达标且样本量足够")


def collect(client: Any, project: str, window_minutes: int) -> dict[str, int]:
    """拉窗口内的 span annotation，喂进 acceptance 的记录表。

    返回 {annotation_name: 记录数}，供调用方打印排查。
    先 reset()：acceptance 的记录表是模块级全局，不清会把上一轮的混进来。
    """
    acceptance.reset()
    now = datetime.now(timezone.utc)
    spans = client.spans.get_spans(
        project_identifier=project,
        start_time=now - timedelta(minutes=window_minutes),
        end_time=now,
        limit=PULL_LIMIT,
    )
    roots = [s for s in spans if not s.get("parent_id")]
    if not roots:
        return {}

    anns = client.spans.get_span_annotations(
        span_ids=[otel_span_id(s) for s in roots],
        project_identifier=project,
        limit=PULL_LIMIT,
    )
    counts: dict[str, int] = {}
    for a in anns:
        name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        result = a.get("result") if isinstance(a, dict) else getattr(a, "result", None)
        if not name or not isinstance(result, dict):
            continue
        acceptance.record(str(name), result.get("score"), label=result.get("label"))
        counts[str(name)] = counts.get(str(name), 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="聚合线上评分并输出三态判定")
    p.add_argument("--project", default="bz-rag-canary")
    p.add_argument("--window-minutes", type=int, default=180)
    p.add_argument("--criteria", default=CRITERIA_PATH)
    args = p.parse_args(argv)

    from phoenix.client import Client

    client = Client(base_url=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006"))

    counts = collect(client, args.project, args.window_minutes)
    print(f"窗口 {args.window_minutes} 分钟，收集到 annotation：{counts or '无'}")

    criteria = acceptance.load_criteria(args.criteria, "online")
    outcomes = acceptance.evaluate_all(criteria)
    d = decide(outcomes)

    print(d.scoreboard)
    print(f">>> 判定：{d.state}  （{d.reason}）")

    return {"promote": EXIT_PROMOTE, "rollback": EXIT_ROLLBACK, "hold": EXIT_HOLD}[d.state]


if __name__ == "__main__":
    raise SystemExit(main())
