"""把声明式验收条件挂进 pytest 生命周期。

判定发生在 sessionfinish：所有 case 都跑完之后再算，这样一次能看到全部回归。
"""

import pathlib

import pytest

from evaluation.phoenix import acceptance

CRITERIA_PATH = str(pathlib.Path(__file__).parent / "criteria.yaml")

_EXIT_ACCEPTANCE_FAILED = 3


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
