# dim3_doc_utilization.py
#
#  维度3：检索结果利用程度奖励函数
#  独立可用，遵循 verl 自定义奖励函数接口
#
#  衡量检索文档关键词在（thinking + answer）中的覆盖程度。
#  奖励"至少一篇文档被充分利用"，而非要求所有文档都被用到。
#
#  聚合公式：0.6 * top1_utilization + 0.4 * mean_utilization
#
#  分值范围 [0, 1]：
#    无文档时返回中性分 0.5
#    无推理内容时返回 0
#
#  接口：
#    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

from rubric_utils import (
    extract_answer,
    extract_retrieved_docs,
    extract_think_content,
    safe_mean,
    tokenize,
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    维度3：检索结果利用程度。

    Returns:
        float in [0, 1]
    """
    think_content = extract_think_content(solution_str)
    answer = extract_answer(solution_str) or ""
    retrieved_docs = extract_retrieved_docs(solution_str)

    if not retrieved_docs:
        return 0.50  # 无文档：中性分

    # reasoning_tokens = thinking + answer 的合并 token 集合
    reasoning_tokens = set(tokenize(think_content + " " + answer))
    if not reasoning_tokens:
        return 0.0

    utilization_scores: List[float] = []
    for doc in retrieved_docs:
        doc_tokens = set(tokenize(doc))
        if len(doc_tokens) < 3:
            continue  # 过短文档跳过，避免虚高
        coverage = len(doc_tokens & reasoning_tokens) / len(doc_tokens)
        utilization_scores.append(coverage)

    if not utilization_scores:
        return 0.0

    top1 = max(utilization_scores)
    mean_util = safe_mean(utilization_scores)
    return float(min(0.6 * top1 + 0.4 * mean_util, 1.0))
