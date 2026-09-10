"""影子分流器的决策逻辑。

复刻 cf-worker/src/index.js：读 canary_weight (0-100)、Math.random()*100 < weight
则走 canary。参数形状必须与 scripts/cf-kv-update.sh 一致（0-100 整数），
这样期 3 的 monitor 输出能同时驱动影子侧和真云侧，将来真上云不用改 monitor。
"""

import json

import pytest

from evaluation.phoenix.shadow.router import pick_backend, read_weight


class TestPickBackend:
    def test_weight_zero_always_stable(self):
        assert all(pick_backend(0, r / 100) == "stable" for r in range(100))

    def test_weight_hundred_always_canary(self):
        assert all(pick_backend(100, r / 100) == "canary" for r in range(100))

    def test_boundary_is_strict_less_than(self):
        """roll*100 < weight 才走 canary，与 cf-worker 的 Math.random()*100 < weight 一致。"""
        assert pick_backend(50, 0.49) == "canary"
        assert pick_backend(50, 0.50) == "stable"


class TestReadWeight:
    def test_reads_int(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"canary_weight": 25}), encoding="utf-8")
        assert read_weight(str(p)) == 25

    def test_missing_file_defaults_to_zero(self, tmp_path):
        """读不到就当全量走 stable——失败方向要安全。"""
        assert read_weight(str(tmp_path / "nope.json")) == 0

    def test_malformed_json_defaults_to_zero(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text("not json", encoding="utf-8")
        assert read_weight(str(p)) == 0

    def test_out_of_range_rejected(self, tmp_path):
        """越界是配置错误，不能静默当 0——那会让人以为分流在跑其实没跑。"""
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"canary_weight": 150}), encoding="utf-8")
        with pytest.raises(ValueError, match="canary_weight"):
            read_weight(str(p))

    def test_non_int_rejected(self, tmp_path):
        p = tmp_path / "w.json"
        p.write_text(json.dumps({"canary_weight": "50"}), encoding="utf-8")
        with pytest.raises(ValueError, match="canary_weight"):
            read_weight(str(p))
