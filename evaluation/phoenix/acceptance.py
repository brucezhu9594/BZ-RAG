"""声明式验收条件——Arize 只在 TypeScript 侧提供 acceptanceCriteria，这是 Python 版。

设计上刻意抄了 Arize 的四条取舍，它们才是价值所在：
  1. 所有 case 跑完之后才判，一次看到全部回归，而不是第一个失败就中断
  2. 缺失的指标判失败，不 vacuously pass
  3. 判官报错是第三态（errored），既不算 0 分也不算通过
  4. 结果落库失败只 warn，不影响门禁判定——所以本模块完全不读 Phoenix，只读进程内累加器

补充（Task review Ruling R18）：光靠 min_samples 兜不住"判官大面积报错"这类
漏判——样本规模一大，min_samples 早就被 usable 记录数盖过去了（比如 48 条
预期里 40 errored + 8 好，usable=8 照样过关）。所以另外加了一个与规模无关的
判据 max_error_rate：errored 占比超过阈值直接 FAIL，且这件事无论最终判
PASS 还是 FAIL 都要在记分卡上无条件可见——不能让"这次绿灯不可信"这件事
在人看得见的输出里彻底隐形。
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
    max_error_rate: float | None = None
    # 缓存 pass_when 解析+校验后的 AST，避免 _evaluate_one 对同一个字符串
    # 做 n_records 次重复 ast.parse（Ruling R20 (a)）。不是公开接口，不参与
    # repr/相等比较。
    _pass_when_ast: ast.expr | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.metric not in _ALLOWED_METRICS:
            raise ValueError(f"metric 必须是 {_ALLOWED_METRICS} 之一，收到 {self.metric!r}")
        if self.metric == "average" and self.threshold is None:
            raise ValueError(f"{self.annotation}: metric=average 必须给 threshold")
        if self.metric == "pass_rate" and (self.pass_when is None or self.min_pass_rate is None):
            raise ValueError(f"{self.annotation}: metric=pass_rate 必须给 pass_when 与 min_pass_rate")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction 必须是 maximize/minimize，收到 {self.direction!r}")
        if self.max_error_rate is not None and not (0.0 <= self.max_error_rate <= 1.0):
            raise ValueError(f"max_error_rate 必须在 0..1 之间，收到 {self.max_error_rate!r}")
        if self.pass_when is not None:
            # 在构造期就把 SyntaxError（表达式写残了）和 ValueError（用了白
            # 名单外的节点/名字/运算符）挡下来——Ruling R20 (a)，详见
            # _validate_pass_when 的 docstring。
            self._pass_when_ast = _validate_pass_when(self.pass_when)


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


# —— pass_when：只允许比较运算与 score/label 两个名字，不用 eval ——
#
# 三个函数分工（Ruling R20 之后的结构，唯一一份白名单判断只活在
# _check_shape / _check_value_shape 里，_eval_tree 不重复判断，避免出现
# 两份可能互相漂移的平行实现）：
#   _validate_pass_when(expr)      解析 + 结构校验，返回 AST；在 Criterion
#                                   构造期调用一次。
#   _eval_tree(tree, rec)          对已校验过的 AST 按 rec 的值求值，不再
#                                   重复做结构检查。
#   _eval_pass_when(expr, rec)     独立入口：现场校验 + 求值，供测试/一次性
#                                   探测使用；Criterion 走的是缓存好的 AST，
#                                   不经过这里。

_CMP = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}

_ALLOWED_NAMES = ("score", "label")


def _check_value_shape(node) -> None:
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(f"pass_when 只能引用 score / label，收到 {node.id!r}")
        return
    if isinstance(node, ast.Constant):
        return
    raise ValueError(f"pass_when 不支持的表达式节点：{type(node).__name__}")


def _check_shape(node) -> None:
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check_shape(v)
        return
    if isinstance(node, ast.Compare):
        _check_value_shape(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _CMP:
                # 白名单只登记了 == != < <= > >=；is/is not/in/not in 等
                # 语法上合法但未登记的运算符（最容易发生在有人把 == 手滑
                # 写成 is）如果不在这里挡住，会在真正求值时让查表操作抛出
                # 裸 KeyError。
                raise ValueError(f"pass_when 不支持的比较运算符：{type(op).__name__}")
            _check_value_shape(comparator)
        return
    raise ValueError(f"pass_when 不支持的表达式节点：{type(node).__name__}")


def _validate_pass_when(expr: str) -> ast.expr:
    """解析并校验 pass_when 表达式，只看结构、不需要真实的 Record。

    Ruling R20：这样 Criterion 构造期（__post_init__）就能把 SyntaxError
    （表达式写残了，比如 YAML 里 "score >=" 少打一半）和 ValueError（用了
    白名单外的节点/名字/运算符）都在跑任何 case 之前挡下来——而不是像最初
    的实现那样，烧完所有判官调用、跑到 evaluate_all 最后一步才引爆，把已经
    判官打出来的其它 criteria 结果一并丢光。

    校验用的运算符表就是 _eval_tree 求值时用的同一个 _CMP，不是另一份平行
    维护的白名单。

    没法在这里挡住的是运行时才暴露的类型不匹配（比如 pass_when 里的常量
    类型和 score/label 的实际类型对不上，如 "score >= 'abc'"）——这个只有
    真的比较到具体值才会抛 TypeError，只能在 _evaluate_one 里用运行时
    try/except 兜底（见那里的注释）。
    """
    tree = ast.parse(expr, mode="eval").body
    _check_shape(tree)
    return tree


def _eval_tree(tree: ast.expr, rec: Record) -> bool:
    """对已经在 _validate_pass_when 校验过的 AST 求值。不重复做结构合法性
    检查——哪些节点/名字/运算符合法，只在 _check_shape / _check_value_shape
    里判断一次。"""

    def val(node):
        if isinstance(node, ast.Name):
            return rec.score if node.id == "score" else rec.label
        return node.value

    def run(node) -> bool:
        if isinstance(node, ast.BoolOp):
            results = [run(v) for v in node.values]
            return all(results) if isinstance(node.op, ast.And) else any(results)
        left = val(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            right = val(comparator)
            if left is None or right is None:
                return False
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True

    return run(tree)


def _eval_pass_when(expr: str, rec: Record) -> bool:
    """独立求值入口：未经过 Criterion 构造期缓存的临时表达式在这里现场校验
    + 求值（供测试/一次性探测使用）。Criterion 走的是缓存好 AST 的
    _eval_tree（见 _evaluate_one），不会对同一个字符串重复 ast.parse。"""
    tree = _validate_pass_when(expr)
    return _eval_tree(tree, rec)


def _evaluate_one(c: Criterion) -> Outcome:
    recs = _RECORDS.get(c.annotation, [])
    errored = [r for r in recs if r.error is not None]
    usable = [r for r in recs if r.error is None]
    err_note = f"，{len(errored)} errored" if errored else ""

    required = c.threshold if c.metric == "average" else c.min_pass_rate

    # Ruling R18 (b)：与样本规模无关的错误率闸门，放在 metric 逻辑之前。
    # min_samples 只保证"够几条"，规模一大就形同虚设（48 条里 40 errored +
    # 8 好，usable=8 一样能过 min_samples）；error_rate 才是不随规模漂移
    # 的判据。
    total = len(errored) + len(usable)
    if c.max_error_rate is not None and total > 0:
        error_rate = len(errored) / total
        if error_rate > c.max_error_rate:
            return Outcome(
                c, False, None, required, len(usable),
                f"error rate {len(errored)}/{total} ({error_rate:.3f}) "
                f"exceeds max_error_rate {c.max_error_rate}{err_note}",
            )

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
    try:
        passing = sum(1 for r in usable if _eval_tree(c._pass_when_ast, r))
    except (ValueError, TypeError) as e:
        # Ruling R19/R20 (b)：唯一一类没法在构造期挡住的逃逸——运行时值类型
        # 不匹配（例如常量写成字符串却拿去跟 float 型 score 比较）会抛
        # TypeError，只有真的比较到具体值才暴露。窄 try/except 把它坐实成
        # 这一条 criterion 的 FAIL Outcome，而不是让异常裸奔到 evaluate_all
        # 炸掉整批——这样"就地显形，而不是拖垮整批"这句话才是真的成立。
        return Outcome(c, False, None, required, len(usable),
                       f"invalid pass_when: {e}{err_note}")
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
        elif "errored" in o.reason:
            # Ruling R18 (a)：即使这一条最终判 PASS，只要 reason 里带了
            # errored 计数就必须无条件打出来——否则"这次绿灯不可信"会在人
            # 看得见的输出里彻底隐形（实测过：19/20 errored 时两条
            # faithfulness criteria 都 PASS，且旧版 format_scoreboard 只在
            # FAIL 行打 reason，"errored" 完全不出现在记分卡里）。
            lines.append(f"{'':<22}⚠ {o.reason}")
    lines.append("-" * 78)
    return "\n".join(lines)
