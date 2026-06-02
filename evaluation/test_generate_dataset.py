"""evaluation/generate_dataset.py 单元测试。"""
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from evaluation import generate_dataset


class TestShortSource:
    def test_extracts_help_content_id(self):
        url = "https://cms.hewa.cn/content/mian/helpContent/10006"
        assert generate_dataset._short_source(url) == "helpContent/10006"

    def test_extracts_with_trailing_slash(self):
        url = "https://cms.hewa.cn/content/mian/helpContent/10573/"
        assert generate_dataset._short_source(url) == "helpContent/10573"

    def test_raises_on_unrecognized_url(self):
        with pytest.raises(ValueError, match="无法从 source 抽取"):
            generate_dataset._short_source("https://example.com/about")


class TestGenerateOne:
    def _make_doc(self, source: str = "https://cms.hewa.cn/content/mian/helpContent/10006") -> Document:
        return Document(
            page_content="禾蛙是一个人力资源平台。域名 hewa.cn，成立于 2020 年。",
            metadata={"source": source},
        )

    def test_returns_three_fields_on_clean_json(self):
        doc = self._make_doc()
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = '{"question": "禾蛙是什么平台", "ground_truth": "禾蛙是人力资源平台"}'
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item == {
            "question": "禾蛙是什么平台",
            "ground_truth": "禾蛙是人力资源平台",
            "expected_source": "helpContent/10006",
        }

    def test_handles_broken_json_via_extract(self):
        """LLM 返回 <think> 包裹 + markdown fence 的脏输出，_extract_json 应当救回。"""
        doc = self._make_doc()
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = (
            '<think>用户问的是这页讲什么。</think>\n'
            '```json\n'
            '{"question": "什么是禾蛙", "ground_truth": "人力资源平台"}\n'
            '```'
        )
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item is not None
        assert item["question"] == "什么是禾蛙"
        assert item["ground_truth"] == "人力资源平台"
        assert item["expected_source"] == "helpContent/10006"

    def test_returns_none_on_unparseable_llm_output(self):
        doc = self._make_doc()
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = "完全的散文，没有任何 JSON 结构。"
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item is None

    def test_returns_none_on_invalid_source_url(self):
        doc = self._make_doc(source="https://example.com/random-page")
        fake_llm = MagicMock()
        fake_msg = MagicMock()
        fake_msg.content = '{"question": "Q", "ground_truth": "A"}'
        fake_llm.invoke.return_value = fake_msg

        item = generate_dataset.generate_one(doc, fake_llm)

        assert item is None
