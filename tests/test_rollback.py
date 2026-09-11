"""权重归零的执行。

影子侧写 weight.json，云侧调 scripts/cf-kv-update.sh。
云侧现在跑不通（CF_API_TOKEN 未配置），所以默认 target=shadow，
且 token 缺失时必须**明确报错**而不是静默跳过——
"以为回滚了其实没回滚"比直接失败危险得多。
"""

import json

import pytest

from evaluation.phoenix.rollback import set_weight


class TestShadow:
    def test_writes_weight_file(self, tmp_path):
        p = tmp_path / "weight.json"
        p.write_text(json.dumps({"canary_weight": 50}), encoding="utf-8")
        set_weight(0, target="shadow", weight_path=str(p))
        assert json.loads(p.read_text(encoding="utf-8")) == {"canary_weight": 0}

    def test_creates_file_if_missing(self, tmp_path):
        p = tmp_path / "weight.json"
        set_weight(25, target="shadow", weight_path=str(p))
        assert json.loads(p.read_text(encoding="utf-8")) == {"canary_weight": 25}

    def test_rejects_out_of_range(self, tmp_path):
        with pytest.raises(ValueError, match="0-100"):
            set_weight(150, target="shadow", weight_path=str(tmp_path / "w.json"))

    def test_rejects_bool(self, tmp_path):
        """True 在 Python 里 isinstance(x, int) 为真，必须单独挡掉。"""
        with pytest.raises(ValueError, match="0-100"):
            set_weight(True, target="shadow", weight_path=str(tmp_path / "w.json"))


class TestCloud:
    def test_missing_token_raises_not_silently_skips(self, monkeypatch):
        """CF_API_TOKEN 缺失必须抛，不能当作"回滚成功"。"""
        monkeypatch.delenv("CF_API_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="CF_API_TOKEN"):
            set_weight(0, target="cloud")

    def test_with_token_invokes_the_script(self, monkeypatch):
        monkeypatch.setenv("CF_API_TOKEN", "fake")
        called = {}

        def fake_run(cmd, check):
            called["cmd"] = cmd
            called["check"] = check

        monkeypatch.setattr("evaluation.phoenix.rollback.subprocess.run", fake_run)
        set_weight(0, target="cloud")
        assert called["cmd"][0] == "bash"
        assert called["cmd"][-1] == "0"
        assert called["cmd"][1].endswith("cf-kv-update.sh")
        assert called["check"] is True


class TestTargetValidation:
    def test_unknown_target_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="target"):
            set_weight(0, target="nonesuch", weight_path=str(tmp_path / "w.json"))
