# dim1_query_quality.py
#
#  维度1：搜索 Query 质量奖励函数
#  独立可用，遵循 verl 自定义奖励函数接口
#
#  评估每条搜索 query 的质量，含三个子维度：
#    A. 长度合理性 (30%) — query 词数梯形评分
#    B. 与问题相关性 (50%) — 去停用词 Jaccard
#    C. 非直接复制惩罚 (20%) — 惩罚直接复制原问题
#
#  接口：
#    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

from rubric_utils import (
    extract_question,
    extract_search_queries,
    safe_mean,
    tokenize,
)


def _score_single_query(
    query: str,
    question_tokens: set,
) -> float:
    """对单条 query 计算质量分 [0, 1]"""
    q_tokens_list = tokenize(query)
    if not q_tokens_list:
        return 0.0
    q_token_set = set(q_tokens_list)

    # ── A. 长度合理性（梯形：理想 3-12 词） ──
    n = len(query.split())
    if n < 2:
        length_score = 0.10
    elif n <= 3:
        length_score = 0.40 + 0.20 * (n - 2)   # 0.40 → 0.60
    elif n <= 12:
        length_score = 1.00
    elif n <= 20:
        length_score = 1.00 - 0.50 * (n - 12) / 8  # 1.00 → 0.50
    else:
        length_score = 0.20

    # ── B. 与问题相关性（Jaccard） ──
    if question_tokens:
        union = question_tokens | q_token_set
        inter = question_tokens & q_token_set
        relevance = len(inter) / len(union) if union else 0.0
    else:
        relevance = 0.50  # 无法获取问题时中性分

    # ── C. 非直接复制惩罚 ──
    if question_tokens and q_token_set:
        copy_ratio = len(q_token_set & question_tokens) / len(q_token_set)
        copy_penalty = 1.0 - max(0.0, (copy_ratio - 0.85) / 0.15)
    else:
        copy_penalty = 1.0

    return 0.30 * length_score + 0.50 * relevance + 0.20 * copy_penalty


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    维度1：搜索 Query 质量。

    提取模型生成的所有 <search> query，逐条评估后取均值。
    无 query 时返回 0。

    Returns:
        float in [0, 1]
    """
    queries = extract_search_queries(solution_str)
    if not queries:
        return 0.0

    question = extract_question(solution_str, ground_truth, extra_info)
    question_tokens = set(tokenize(question))

    per_query_scores = [
        _score_single_query(q, question_tokens) for q in queries
    ]
    return float(safe_mean(per_query_scores))
