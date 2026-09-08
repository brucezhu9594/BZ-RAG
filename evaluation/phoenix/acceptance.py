"""声明式验收条件——Arize 只在 TypeScript 侧提供 acceptanceCriteria，这是 Python 版。

设计上刻意抄了 Arize 的四条取舍，它们才是价值所在：
  1. 所有 case 跑完之后才判，一次看到全部回归，而不是第一个失败就中断
  2. 缺失的指标判失败，不 vacuously pass
  3. 判官报错是第三态（errored），既不算 0 分也不算通过
  4. 结果落库失败只 warn，不影响门禁判定——所以本模块完全不读 Phoenix，只读进程内累加器
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field

import yaml

_ALLOWED_METRICS = ("average", "pass_rate")


@dataclass
class Record:
    score: float | None
    label: str | None = None
    error: str | None = None


@dataclass
class Criterion:
    annotation: str
    metric: str
    threshold: float | None = None
    direction: str = "maximize"
    pass_when: str | None = None
    min_pass_rate: float | None = None
    min_samples: int = 1

    def __post_init__(self) -> None:
        if self.metric not in _ALLOWED_METRICS:
            raise ValueError(f"metric 必须是 {_ALLOWED_METRICS} 之一，收到 {self.metric!r}")
        if self.metric == "average" and self.threshold is None:
            raise ValueError(f"{self.annotation}: metric=average 必须给 threshold")
        if self.metric == "pass_rate" and (self.pass_when is None or self.min_pass_rate is None):
            raise ValueError(f"{self.annotation}: metric=pass_rate 必须给 pass_when 与 min_pass_rate")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction 必须是 maximize/minimize，收到 {self.direction!r}")


@dataclass
class Outcome:
    criterion: Criterion
    passed: bool
    observed: float | None
    required: float
    samples: int
    reason: str


_RECORDS: dict[str, list[Record]] = {}


def reset() -> None:
    _RECORDS.clear()


def record(
    name: str, score: float | None, label: str | None = None, error: str | None = None
) -> None:
    _RECORDS.setdefault(name, []).append(Record(score=score, label=label, error=error))


# —— pass_when 求值：只允许比较运算与 score/label 两个名字，不用 eval ——

_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _eval_pass_when(expr: str, rec: Record) -> bool:
    tree = ast.parse(expr, mode="eval").body

    def val(node):
        if isinstance(node, ast.Name):
            if node.id == "score":
                return rec.score
            if node.id == "label":
                return rec.label
            raise ValueError(f"pass_when 只能引用 score / label，收到 {node.id!r}")
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError(f"pass_when 不支持的表达式节点：{type(node).__name__}")

    def run(node) -> bool:
        if isinstance(node, ast.BoolOp):
            results = [run(v) for v in node.values]
            return all(results) if isinstance(node.op, ast.And) else any(results)
        if isinstance(node, ast.Compare):
            left = val(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = val(comparator)
                if left is None or right is None:
                    return False
                cmp_fn = _CMP.get(type(op))
                if cmp_fn is None:
                    # 白名单只登记了 == != < <= > >=；is/is not/in/not in 等语法上
                    # 合法但未登记的运算符（例如把 == 手滑写成 is）如果直接查表会
                    # 抛出裸 KeyError，逃出本函数一路炸穿 evaluate_all，让所有其他
                    # criteria 也判不出来——这违反了「全部跑完再判」的取舍。这里把
                    # 它转成同风格的 ValueError，让配置错误在这一条 criterion 上
                    # 就地显形，而不是拖垮整批。
                    raise ValueError(f"pass_when 不支持的比较运算符：{type(op).__name__}")
                if not cmp_fn(left, right):
                    return False
                left = right
            return True
        raise ValueError(f"pass_when 不支持的表达式节点：{type(node).__name__}")

    return run(tree)


def _evaluate_one(c: Criterion) -> Outcome:
    recs = _RECORDS.get(c.annotation, [])
    errored = [r for r in recs if r.error is not None]
    usable = [r for r in recs if r.error is None]
    err_note = f"，{len(errored)} errored" if errored else ""

    required = c.threshold if c.metric == "average" else c.min_pass_rate

    if not usable:
        return Outcome(c, False, None, required, 0,
                       f"no {c.annotation} scores found{err_note}")

    if c.metric == "average":
        numeric = [r.score for r in usable if r.score is not None]
        if not numeric:
            return Outcome(c, False, None, required, 0,
                           f"no {c.annotation} numeric scores found{err_note}")
        if len(numeric) < c.min_samples:
            return Outcome(c, False, sum(numeric) / len(numeric), required, len(numeric),
                           f"insufficient samples: {len(numeric)} < {c.min_samples}{err_note}")
        observed = sum(numeric) / len(numeric)
        passed = observed >= c.threshold if c.direction == "maximize" else observed <= c.threshold
        cmp = ">=" if c.direction == "maximize" else "<="
        return Outcome(c, passed, observed, required, len(numeric),
                       f"mean {observed:.3f} {cmp} {c.threshold}{err_note}")

    if len(usable) < c.min_samples:
        return Outcome(c, False, None, required, len(usable),
                       f"insufficient samples: {len(usable)} < {c.min_samples}{err_note}")
    passing = sum(1 for r in usable if _eval_pass_when(c.pass_when, r))
    observed = passing / len(usable)
    passed = observed >= c.min_pass_rate
    return Outcome(c, passed, observed, required, len(usable),
                   f"pass rate {observed:.3f} >= {c.min_pass_rate}{err_note}")


def evaluate_all(criteria: list[Criterion]) -> list[Outcome]:
    """全部求值后再返回——一次看到所有回归，而不是第一个失败就短路。"""
    return [_evaluate_one(c) for c in criteria]


def load_criteria(path: str, section: str) -> list[Criterion]:
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    if section not in doc:
        raise KeyError(f"{path} 里没有 {section!r} 段")
    return [Criterion(**item) for item in doc[section]]


def format_scoreboard(outcomes: list[Outcome]) -> str:
    lines = ["", "Acceptance Criteria", "-" * 78,
             f"{'annotation':<22}{'metric':<12}{'observed':>10}{'required':>10}{'n':>6}  verdict"]
    for o in outcomes:
        obs = "n/a" if o.observed is None else f"{o.observed:.3f}"
        lines.append(
            f"{o.criterion.annotation:<22}{o.criterion.metric:<12}{obs:>10}"
            f"{o.required:>10.3f}{o.samples:>6}  {'PASS' if o.passed else 'FAIL'}"
        )
        if not o.passed:
            lines.append(f"{'':<22}└─ {o.reason}")
    lines.append("-" * 78)
    return "\n".join(lines)
