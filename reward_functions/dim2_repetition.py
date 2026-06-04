# dim2_repetition.py
#
#  维度2：重复搜索惩罚奖励函数
#  独立可用，遵循 verl 自定义奖励函数接口
#
#  惩罚多轮次中高度相似的 query：
#    - 使用分词后的 Jaccard 相似度
#    - 相邻轮次惩罚权重更高（1.5x）
#    - 非线性映射：低相似度轻微惩罚，高相似度重惩罚
#    - 分值范围 [0, 1]：1 = 无重复，0 = 完全重复
#
#  接口：
#    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

from rubric_utils import extract_search_queries, tokenize


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    维度2：重复搜索惩罚。

    0 条或 1 条 query → 满分（无重复可罚）
    多条 query 时计算加权平均 Jaccard 相似度后映射为分数。

    Returns:
        float in [0, 1]，分数越高表示重复越少
    """
    queries = extract_search_queries(solution_str)

    # 单轮搜索无重复
    if len(queries) <= 1:
        return 1.0

    token_sets: List[set] = [set(tokenize(q)) for q in queries]

    total_penalty = 0.0
    total_weight = 0.0

    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            t1, t2 = token_sets[i], token_sets[j]
            if not t1 or not t2:
                continue

            jaccard = len(t1 & t2) / len(t1 | t2)
            # 相邻轮次惩罚 1.5x
            weight = 1.5 if j == i + 1 else 1.0
            total_penalty += weight * jaccard
            total_weight += weight

    if total_weight == 0:
        return 1.0

    avg_sim = total_penalty / total_weight

    # 非线性映射到分数
    if avg_sim < 0.30:
        # 低相似度：轻微惩罚
        score = 1.0 - avg_sim * 0.30
    elif avg_sim < 0.70:
        # 中等相似度：线性惩罚
        # 0.30→0.91, 0.70→0.30
        score = 0.91 - (avg_sim - 0.30) * 1.525
    else:
        # 高相似度：重惩罚
        score = max(0.0, 0.30 - (avg_sim - 0.70) * 1.00)

    return float(score)
