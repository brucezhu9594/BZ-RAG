"""关键词补充：基于同义词词典扩展查询关键词，增强 BM25 检索效果。"""

import jieba

# 领域同义词词典：key 为触发词，value 为补充的同义词列表
SYNONYM_DICT: dict[str, list[str]] = {
    # 平台相关
    "禾蛙": ["hewa", "平台"],
    "平台": ["禾蛙", "系统"],
    # 角色相关
    "PM": ["项目经理", "工作人员", "职位PM"],
    "项目经理": ["PM", "工作人员"],
    "接单方": ["接单", "顾问", "猎头"],
    "发单方": ["发单", "企业", "招聘方", "HR"],
    "候选人": ["人选", "简历", "人才"],
    # 操作相关
    "注销": ["删除账号", "注销账号", "关闭账户"],
    "注册": ["开通", "创建账号", "申请账号"],
    "登录": ["登陆", "签到", "进入"],
    "发票": ["开票", "开发票", "税务"],
    "佣金": ["报酬", "费用", "结算", "薪酬"],
    # 业务相关
    "盲推": ["盲推简历", "未经沟通推荐"],
    "入驻": ["驻场", "进驻", "入驻企业"],
    "简历": ["候选人", "人选", "推荐"],
    "职位": ["岗位", "招聘", "JD"],
}


def expand_keywords(query: str) -> str:
    """对查询进行关键词补充，返回扩展后的查询文本。"""
    words = list(jieba.cut(query))
    expanded = set()

    for word in words:
        word_stripped = word.strip()
        if word_stripped in SYNONYM_DICT:
            expanded.update(SYNONYM_DICT[word_stripped])

    if not expanded:
        return query

    # 将补充关键词拼接到原查询后面
    supplement = " ".join(expanded - set(words))
    return f"{query} {supplement}"
