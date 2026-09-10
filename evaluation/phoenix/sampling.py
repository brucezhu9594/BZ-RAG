"""按 span_id 做确定性抽样。

**不用 random.random()**：cadence=continuous 的任务窗口会重叠，同一条 span
会被连续几轮反复拉回来。随机抽样下它每轮都重新掷一次骰子，采样率 0.2 跑六轮
实际评掉的远超 20%，成本旋钮失去意义；确定性抽样则让"这条 span 要不要评"
成为它自身的属性，跑多少轮都一样。

用 sha256 而不是内置 hash()：Python 的 str.__hash__ 默认带 PYTHONHASHSEED
随机化，进程间不一致——worker 每轮都是一个新进程，用内置 hash 就不确定了。
"""

import hashlib

_MAX = 0xFFFFFFFF


def should_sample(span_id: str, rate: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(span_id.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) / _MAX) < rate
