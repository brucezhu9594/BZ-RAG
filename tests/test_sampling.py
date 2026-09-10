"""抽样必须是确定性的：同一条 span 在任何一轮都得到同样的抽中/不抽中判定。

理由是窗口会重叠。cadence=continuous 的任务每轮都会把最近 window_minutes
的 span 全拉回来，同一条 span 会被反复看到。若用 random.random()，它可能
这轮没抽中、下轮抽中，去重逻辑就失去意义，成本旋钮也不再线性——
采样率 0.2 跑六轮实际会评掉远超 20% 的 span。
"""

from evaluation.phoenix.sampling import should_sample


class TestDeterminism:
    def test_same_span_same_verdict(self):
        ids = [f"span-{i}" for i in range(200)]
        first = [should_sample(i, 0.3) for i in ids]
        second = [should_sample(i, 0.3) for i in ids]
        assert first == second

    def test_different_ids_not_all_equal(self):
        """不能退化成全抽或全不抽。"""
        verdicts = {should_sample(f"span-{i}", 0.5) for i in range(50)}
        assert verdicts == {True, False}


class TestRateBoundaries:
    def test_rate_one_always_samples(self):
        """护栏型任务写 sampling_rate: 1.0，必须一条都不漏。"""
        assert all(should_sample(f"span-{i}", 1.0) for i in range(500))

    def test_rate_zero_never_samples(self):
        assert not any(should_sample(f"span-{i}", 0.0) for i in range(500))

    def test_rate_roughly_proportional(self):
        """0.2 在 2000 条上应落在 20% 附近；给足容差，这里验的是量级不是精度。"""
        n = 2000
        hit = sum(should_sample(f"span-{i}", 0.2) for i in range(n))
        assert 0.15 * n < hit < 0.25 * n
