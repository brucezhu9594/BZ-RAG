"""线上评估 worker：拉 span → 去重 → 抽样 → 跑判官 → 把分数写回 span。

与期 1 离线门禁的关系：**判官只有一份**（evaluation/phoenix/evaluators.py），
离线线上共用。区别只在输入从哪来、结果写到哪去——
离线是 dataset example → experiment run annotation，
线上是 project span → span annotation。这正是 Arize template/task 分离的好处。

期 2 的 worker **不做任何决策**：不 rollback、不 promote、不阻断。
它只负责把分数写上去。聚合与决策是期 3 的 monitor.py。
"""

import os

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

import argparse
import logging
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

import pandas as pd

from evaluation.phoenix.online_tasks import OnlineTask, load_tasks
from evaluation.phoenix.sampling import should_sample
from evaluation.phoenix.span_extract import extract_eval_input

logger = logging.getLogger(__name__)

PULL_LIMIT = 1000

# 上下文来自重排后的片段——判官必须看到真正喂给 LLM 的那一份。
_CONTEXT_SPAN_KIND = "RERANKER"


class RoundStats(NamedTuple):
    task: str
    pulled: int  # 从 Phoenix 拉回来的 span 总数
    skipped: int  # 形状不合适（非根 span / 缺问题或答案）——正常过滤，不是失败
    deduped: int  # 已有全部同名 annotation，本轮跳过
    unsampled: int  # 通过了去重但没被采样率选中
    sampled: int  # 真正送去评的
    annotated: int  # 成功写回的 annotation 条数
    errored: int  # 判官失败（label == "errored"）

    @property
    def accounted(self) -> int:
        """skipped + deduped + unsampled + sampled 应当等于 pulled。"""
        return self.skipped + self.deduped + self.unsampled + self.sampled


def default_judges() -> dict[str, Callable[..., dict[str, Any]]]:
    """延迟导入真判官。

    **有意不在模块顶层 import**：evaluation.phoenix.evaluators 在 import 期就要求
    JUDGE_OPENAI_API_KEY / JUDGE_OPENAI_BASE_URL / JUDGE_MODEL_ID（期 1 的有意设计，
    缺配置立刻红比跑完上百次判官调用再红便宜）。而 tests/ 要能在 test.yml 的
    ubuntu-latest 上跑——那里没有 .env 也没有 secrets。顶层 import 会让整个
    tests/ 在收集阶段就崩。测试通过 run_task(judges=...) 注入 stub。
    """
    from evaluation.phoenix.evaluators import (
        answer_relevancy,
        faithfulness,
        refusal_check,
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "refusal_check": refusal_check,
    }


def otel_span_id(span: dict[str, Any]) -> str:
    """取 OTel 的十六进制 span id。

    Phoenix 的 span 有**两个** id，用错会静默失败：
      span["id"]                 Phoenix 全局节点 ID，形如 base64 的 "Span:5898"
      span["context"]["span_id"] OTel 的十六进制 span id，形如 "3158f3e0a8011e4b"

    注解 API（读的 get_span_annotations、写的 log_span_annotations_dataframe）
    认的是**后者**。实测用节点 ID 调 get_span_annotations 直接
    404 Not Found，而端点本身是存在的（不带 span_ids 时返回 422 校验错误）——
    这个组合很有迷惑性，容易被误读成"服务端版本不支持这个端点"。
    """
    return str((span.get("context") or {}).get("span_id", ""))


def _trace_id(span: dict[str, Any]) -> str:
    return str((span.get("context") or {}).get("trace_id", ""))


def _context_spans(
    client: Any, task: OnlineTask, trace_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """按 trace 批量补拉上下文 span（RERANKER）。

    **必须单独拉一次。** 主查询带了 span_kind=AGENT 过滤，返回的只有 AGENT span，
    在那批结果里找 RERANKER 兄弟永远找不到——实测后果是 faithfulness 每条都看到
    空上下文、每条都判 incorrect，分数全是垃圾，而计数（pulled/sampled/annotated）
    看起来完全正常。这个 bug 只有去看 annotation 的**内容**才会暴露。
    """
    trace_ids = {t for t in trace_ids if t}
    if not trace_ids:
        return {}
    spans = client.spans.get_spans(
        project_identifier=task.project,
        span_kind=_CONTEXT_SPAN_KIND,
        trace_ids=sorted(trace_ids),
        limit=PULL_LIMIT,
    )
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in spans:
        out[_trace_id(s)].append(s)
    return out


def _existing_annotations(
    client: Any, project: str, span_ids: list[str], names: tuple[str, ...]
) -> set[tuple[str, str]]:
    """返回 {(span_id, annotation_name)}，用于去重。

    去重是必须的：cadence=continuous 的任务窗口重叠（window_minutes 是触发间隔的
    2 倍），同一条 span 会被连续几轮反复拉到。不去重就会对同一条 span 重复调判官
    ——重复计费，且 Phoenix 上会堆出一串同名 annotation。
    """
    if not span_ids:
        return set()
    got = client.spans.get_span_annotations(
        span_ids=span_ids,
        project_identifier=project,
        include_annotation_names=list(names),
        limit=PULL_LIMIT,
    )
    out: set[tuple[str, str]] = set()
    for a in got:
        # 返回元素在不同版本里可能是 dict 也可能是对象，两种都接住。
        sid = a.get("span_id") if isinstance(a, dict) else getattr(a, "span_id", None)
        nm = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
        if sid and nm:
            out.add((str(sid), str(nm)))
    return out


def run_task(
    client: Any,
    task: OnlineTask,
    now: datetime | None = None,
    judges: dict[str, Callable[..., dict[str, Any]]] | None = None,
) -> RoundStats:
    judges = default_judges() if judges is None else judges
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(minutes=task.window_minutes)

    spans = client.spans.get_spans(
        project_identifier=task.project,
        span_kind=task.span_kind,
        start_time=start,
        end_time=now,
        limit=PULL_LIMIT,
    )
    pulled = len(spans)

    roots = [s for s in spans if not s.get("parent_id")]
    skipped = pulled - len(roots)

    already = _existing_annotations(
        client, task.project, [otel_span_id(s) for s in roots], task.evaluators
    )

    # 先选出这一轮真正要评的，再按 trace 批量补拉上下文 span。
    candidates: list[tuple[dict[str, Any], list[str]]] = []
    deduped = unsampled = sampled = errored = 0
    for root in roots:
        span_id = otel_span_id(root)
        pending = [n for n in task.evaluators if (span_id, n) not in already]
        if not pending:
            deduped += 1
            continue
        if not should_sample(span_id, task.sampling_rate):
            unsampled += 1
            continue
        candidates.append((root, pending))

    by_trace = _context_spans(client, task, {_trace_id(root) for root, _ in candidates})

    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for root, pending in candidates:
        span_id = otel_span_id(root)
        payload = extract_eval_input(root, by_trace.get(_trace_id(root), []))
        if payload is None:
            skipped += 1
            continue

        sampled += 1
        output = {"answer": payload["answer"], "contexts": payload["contexts"]}
        judge_input = {"question": payload["question"]}
        for name in pending:
            res = judges[name](output=output, input=judge_input)
            if res.get("label") == "errored":
                # 判官失败不写 annotation：写一条 label="errored" 的记录会污染
                # 期 3 monitor 的聚合口径（它按 label 判通过）。这里只计数。
                errored += 1
                continue
            rows[name].append(
                {
                    "span_id": span_id,
                    "label": res.get("label"),
                    "score": res.get("score"),
                    "explanation": res.get("explanation"),
                }
            )

    annotated = 0
    for name, items in rows.items():
        if not items:
            continue
        client.spans.log_span_annotations_dataframe(
            dataframe=pd.DataFrame(items),
            annotator_kind="LLM",
            annotation_name=name,
        )
        annotated += len(items)

    return RoundStats(
        task=task.name,
        pulled=pulled,
        skipped=skipped,
        deduped=deduped,
        unsampled=unsampled,
        sampled=sampled,
        annotated=annotated,
        errored=errored,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑一轮线上评估")
    parser.add_argument("--tasks", default=None, help="online_tasks.yaml 路径")
    parser.add_argument("--only", default=None, help="只跑指定名字的任务（调试用）")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from phoenix.client import Client

    endpoint = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
    client = Client(base_url=endpoint)

    tasks = load_tasks(args.tasks) if args.tasks else load_tasks()
    if args.only:
        tasks = [t for t in tasks if t.name == args.only]
        if not tasks:
            logger.error("没有名为 %s 的任务", args.only)
            return 1

    judges = default_judges()
    failed = False
    header = (
        f"{'task':<20}{'pulled':>8}{'skipped':>9}{'deduped':>9}"
        f"{'unsampled':>11}{'sampled':>9}{'annotated':>11}{'errored':>9}"
    )
    print(header)
    print("-" * len(header))
    for t in tasks:
        try:
            s = run_task(client, t, judges=judges)
        except Exception:  # noqa: BLE001 —— 一个任务炸不该让整轮停摆
            logger.exception("任务 %s 抛异常", t.name)
            failed = True
            continue
        print(
            f"{s.task:<20}{s.pulled:>8}{s.skipped:>9}{s.deduped:>9}"
            f"{s.unsampled:>11}{s.sampled:>9}{s.annotated:>11}{s.errored:>9}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
