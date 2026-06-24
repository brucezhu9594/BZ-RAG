"""测试多轮 RAG 的纯历史助手：改写 prompt 组装、带历史的 chat messages 组装。"""

from api.history_utils import build_chat_messages, build_rewrite_prompt


class TestBuildRewritePrompt:
    def test_prompt_contains_query_and_history(self):
        """改写 prompt 应同时含后续问题与历史里的问答。"""
        history = [("禾蛙是什么平台？", "禾蛙是撮合交易平台")]
        prompt = build_rewrite_prompt("它怎么收费？", history)
        assert "它怎么收费？" in prompt
        assert "禾蛙是什么平台？" in prompt
        assert "禾蛙是撮合交易平台" in prompt

    def test_prompt_has_rewrite_instruction(self):
        """prompt 应包含「改写成独立问题、只输出问题」的指令。"""
        prompt = build_rewrite_prompt("它怎么收费？", [("禾蛙是什么？", "平台")])
        assert "改写" in prompt
        assert "只输出" in prompt


class TestBuildChatMessages:
    def test_empty_history_yields_system_plus_user(self):
        """history 为空列表 → 只有 system + 当前 user，两条，与单轮一致。"""
        msgs = build_chat_messages("SYS", "问题", [])
        assert msgs == [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "问题"},
        ]

    def test_none_history_same_as_empty(self):
        """history 为 None 与空列表行为一致（向后兼容）。"""
        assert build_chat_messages("SYS", "问题", None) == build_chat_messages("SYS", "问题", [])

    def test_history_expands_to_alternating_roles(self):
        """两轮历史 → system, user, assistant, user, assistant, 当前 user，共 6 条且顺序正确。"""
        history = [("Q1", "A1"), ("Q2", "A2")]
        msgs = build_chat_messages("SYS", "Q3", history)
        assert [m["role"] for m in msgs] == [
            "system", "user", "assistant", "user", "assistant", "user",
        ]
        assert msgs[0]["content"] == "SYS"
        assert msgs[1]["content"] == "Q1"
        assert msgs[2]["content"] == "A1"
        assert msgs[-1]["content"] == "Q3"
