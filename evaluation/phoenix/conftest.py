"""把声明式验收条件挂进 pytest 生命周期。

判定发生在 sessionfinish：所有 case 都跑完之后再算，这样一次能看到全部回归。
"""

import pathlib

from evaluation.phoenix import acceptance

CRITERIA_PATH = str(pathlib.Path(__file__).parent / "criteria.yaml")

# 不用 3：pytest 保留 0-5 作为内置退出码（OK / TESTS_FAILED / INTERRUPTED /
# INTERNAL_ERROR / USAGE_ERROR / NO_TESTS_COLLECTED），3 正好是 INTERNAL_ERROR。
# 门禁功能上依然会红（sessionfinish 会覆盖 exitstatus），但如果下游 CI 对
# "exit 3 = pytest 内部错误，可能是基础设施抖动"做特判自动重跑，就会把一次
# 真实的 acceptance 失败误判成 infra 问题而重试掉。改用 6，落在 pytest 保留
# 区间（0-5）之外，专属本门禁的"acceptance 未达标"含义（见 docs/CD-pipeline.md）。
_EXIT_ACCEPTANCE_FAILED = 6

# 最终整支 review C2：Phoenix pytest 插件（session.py）有三处早退发生在
# run_evaluators（判官调用）之前——offline/client 为 None、experiment_id 或
# dataset_example_id 为 None、log_run 抛异常（这三处都只 logger.warning 然后
# return）。命中任一，这条 case 的五个判官一次都不跑、acceptance.record() 一次
# 都不调——不是 errored 第三态，是彻底不存在。而 acceptance 自己没有"本该有几条"
# 的概念：errored=0 让 max_error_rate 恒不触发，min_samples 在样本规模一大就形同
# 虚设（离线探针实测：48 条只落库 3 条，五条 criteria 全 PASS、n=3、零 WARN）。
#
# 这里用 pytest_runtest_logreport 数"实际跑到 call 阶段、且没被 skip 掉"的 item
# 数，作为"这次会话本该产出多少条判官记录"的基准，在 sessionfinish 跟每个
# annotation 实际记录到的条数比对，缺了就在记分卡上补一条 FAIL 行、并让退出码
# 一起变红。
#
# 为什么按"call 阶段发生且未 skip"而不是"call 阶段 passed"计数：读了
# phoenix/client/pytest/plugin.py 的 pytest_runtest_makereport 才发现，
# state.record_run(...) 在 teardown 分支调用时**没有显式传 run_evaluators**，
# 而该参数默认值是 True——也就是说测试断言失败（call 阶段 failed）甚至 teardown
# 失败，只要 setup 没失败、也没被 skip，判官依然会跑（因为 record.output 在
# 断言失败前的 log_output() 那一步就已经写好了）。真正跳过判官的只有 setup 失败
# （_record_setup_error 显式传 run_evaluators=False）和 skip（call 阶段
# report.skipped=True 时插件直接不调 record_run）。如果按"call 阶段 passed"
# 计数，会把"管线跑通但断言失败"这类合法产出记录的 case 也算成"不该有记录"，
# 在这类 case 上制造假阳性。
#
# 这个基准量随规模自动成立，不用为 smoke（N=3）与全量（N=48，含 repetitions）
# 分别调参：Phoenix 插件把每次 repetition 实现成独立的 pytest item（nodeid 里带
# phxrepN- 前缀），所以 repetitions 已经天然算在这里数到的 item 数里，不需要
# 再手动乘一遍——乘了反而会把期望值算成实际值的两倍，在 EVAL_REPETITIONS>1 时
# 对每一次正常全量跑都误报"缺样本"。
_executed_nodeids: set[str] = set()


def pytest_runtest_logreport(report) -> None:
    if report.when == "call" and not report.skipped:
        _executed_nodeids.add(report.nodeid)


def _completeness_fails(annotations: list[str], expected: int) -> list[str]:
    """比较每个 annotation 实际记录到的条数与期望条数，返回缺样本的 FAIL 行。

    没有缺样本时返回空列表。独立于 pytest 生命周期——只读 acceptance.record_count()
    和调用方给的 expected，方便离线测试：喂一个手造的 expected、往
    acceptance._RECORDS 里少塞几条记录，直接调用这个函数就能验证，不需要真的跑
    一次 pytest session。
    """
    fails = []
    for name in annotations:
        actual = acceptance.record_count(name)
        if actual < expected:
            missing = expected - actual
            fails.append(
                f"annotation {name} 期望 {expected} 条、实际 {actual} 条，缺 {missing} 条"
                f"（可能是落库失败导致判官未被调用，见 conftest.py 里 C2 的注释）"
            )
    return fails


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: PR 上只跑的小子集")
    acceptance.reset()
    _executed_nodeids.clear()
    # I4：criteria.yaml 的校验（load_criteria 内部会为每条 Criterion 跑
    # __post_init__，pass_when 表达式在这里被解析+校验）之前只在 pytest_sessionfinish
    # 里被调用一次——也就是说一个写残的 pass_when（比如 YAML 里少打一半）要等
    # 240 次判官调用 + 48 次完整管线跑完之后才会在 sessionfinish 抛 SyntaxError，
    # 变成 pytest INTERNALERROR（exit 3），而不是在跑任何 case 之前就报出来。
    # 这里提前调一次，只为了让配置错误尽早暴露；返回值不需要，也不缓存给
    # sessionfinish 用——sessionfinish 那次重新解析一遍是有意的：两次调用互相
    # 独立，不会因为缓存了一份可能过期的结果而在 criteria.yaml 被中途改动时读到
    # 旧配置（虽然同一次 pytest 会话内文件不会变化，但没有理由为省一次几毫秒的
    # YAML parse 去引入一份状态）。
    acceptance.load_criteria(CRITERIA_PATH, "offline")


def pytest_sessionfinish(session, exitstatus):
    criteria = acceptance.load_criteria(CRITERIA_PATH, "offline")
    outcomes = acceptance.evaluate_all(criteria)
    board = acceptance.format_scoreboard(outcomes)
    writer = session.config.get_terminal_writer()
    writer.write(board + "\n")

    failed = any(not o.passed for o in outcomes)

    annotations = sorted({c.annotation for c in criteria})
    sample_fails = _completeness_fails(annotations, len(_executed_nodeids))
    if sample_fails:
        lines = ["", "Sample Completeness", "-" * 78]
        lines.extend(f"FAIL {line}" for line in sample_fails)
        lines.append("-" * 78)
        writer.write("\n".join(lines) + "\n")
        failed = True

    if failed and exitstatus == 0:
        session.exitstatus = _EXIT_ACCEPTANCE_FAILED
