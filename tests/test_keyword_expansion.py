"""测试关键词补充：基于同义词词典扩展查询关键词。"""

from common.keyword_expansion import SYNONYM_DICT, expand_keywords


class TestExpandKeywords:
    def test_returns_original_when_no_synonym_match(self):
        """查询里没有同义词词典里的词时，原样返回。"""
        query = "今天天气怎么样"
        result = expand_keywords(query)
        assert result == query

    def test_expands_single_keyword(self):
        """查询命中一个同义词时，结果应包含原查询和补充词。"""
        query = "PM"
        result = expand_keywords(query)
        assert query in result
        # PM 的同义词是 ["项目经理", "工作人员", "职位PM"]
        for synonym in SYNONYM_DICT["PM"]:
            assert synonym in result

    def test_expanded_output_does_not_duplicate_existing_words(self):
        """补充词里不应再次包含原查询里已有的词。"""
        query = "PM 项目经理"
        result = expand_keywords(query)
        # "项目经理" 已在原查询里，不应在补充部分重复出现
        # 我们检查补充部分（原查询之后）不重复包含 "项目经理"
        supplement = result.replace(query, "", 1)
        assert supplement.count("项目经理") == 0

    def test_handles_empty_query(self):
        """空查询应原样返回。"""
        assert expand_keywords("") == ""

    def test_synonym_dict_has_expected_categories(self):
        """词典应至少包含已定义的几个核心类别词。"""
        assert "PM" in SYNONYM_DICT
        assert "候选人" in SYNONYM_DICT
        assert "注册" in SYNONYM_DICT
