"""把声明式验收条件挂进 pytest 生命周期。

判定发生在 sessionfinish：所有 case 都跑完之后再算，这样一次能看到全部回归。
"""

import pathlib

import pytest

from evaluation.phoenix import acceptance

CRITERIA_PATH = str(pathlib.Path(__file__).parent / "criteria.yaml")

# 不用 3：pytest 保留 0-5 作为内置退出码（OK / TESTS_FAILED / INTERRUPTED /
# INTERNAL_ERROR / USAGE_ERROR / NO_TESTS_COLLECTED），3 正好是 INTERNAL_ERROR。
# 门禁功能上依然会红（sessionfinish 会覆盖 exitstatus），但如果下游 CI 对
# "exit 3 = pytest 内部错误，可能是基础设施抖动"做特判自动重跑，就会把一次
# 真实的 acceptance 失败误判成 infra 问题而重试掉。改用 6，落在 pytest 保留
# 区间（0-5）之外，专属本门禁的"acceptance 未达标"含义（见 docs/CD-pipeline.md）。
_EXIT_ACCEPTANCE_FAILED = 6


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: PR 上只跑的小子集")
    acceptance.reset()


def pytest_sessionfinish(session, exitstatus):
    criteria = acceptance.load_criteria(CRITERIA_PATH, "offline")
    outcomes = acceptance.evaluate_all(criteria)
    board = acceptance.format_scoreboard(outcomes)
    session.config.get_terminal_writer().write(board + "\n")
    if any(not o.passed for o in outcomes) and exitstatus == 0:
        session.exitstatus = _EXIT_ACCEPTANCE_FAILED
