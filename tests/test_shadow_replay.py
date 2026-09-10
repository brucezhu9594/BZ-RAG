"""查询池的约束。

最要紧的一条：**池子必须与金标集零重合**。线上评估如果只测训练过的 case，
测出来的是"金标集上的表现"而不是线上表现，整个期 2 就失去意义。
这条约束靠 review 是守不住的（谁都可能顺手从 test_dataset.json 抄一条进来），
所以钉成测试。
"""

import json
import pathlib

from evaluation.phoenix.shadow.replay import load_pool

REPO = pathlib.Path(__file__).resolve().parents[1]


def _golden_questions() -> set[str]:
    with open(REPO / "evaluation" / "test_dataset.json", encoding="utf-8") as f:
        return {item["question"].strip() for item in json.load(f)}


class TestQueryPool:
    def test_pool_is_non_empty(self):
        assert len(load_pool()) >= 10

    def test_no_overlap_with_golden_set(self):
        """线上评估不能只测训练过的 case——设计文档 §4.7 的明确要求。"""
        overlap = {q.strip() for q in load_pool()} & _golden_questions()
        assert overlap == set(), f"查询池与金标集重合：{overlap}"

    def test_no_duplicates_within_pool(self):
        pool = load_pool()
        assert len(pool) == len(set(pool))
