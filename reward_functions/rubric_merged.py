# rubric_merged.py
#
#  5 维度 Rubric 奖励函数（合并版）
#  遵循 verl 自定义奖励函数接口，可直接配置使用
#
#  维度：
#    1. Query 质量       (w=0.15) — 搜索 query 的相关性、长度、非复制性
#    2. 重复搜索惩罚      (w=0.15) — 惩罚多轮次中高度相似的 query
#    3. 检索结果利用度    (w=0.20) — 检索文档是否驱动了推理
#    4. 推理逻辑一致性   (w=0.15) — 答案是否由 thinking 支撑
#    5. 答案 F1          (w=0.35) — Token-level F1（SQuAD 标准）
#
#  聚合策略：加权线性求和 + EM Gate（F1=0 时过程分衰减至 30%）
#
#  使用方式（verl 配置）：
#    custom_reward_function.path=reward_functions/rubric_merged.py
#    custom_reward_function.name=compute_score
#    （name 不设置则默认使用 compute_score）


import random
import sys
import os
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dim1_query_quality import compute_score as _score_dim1
from dim2_repetition import compute_score as _score_dim2
from dim3_doc_utilization import compute_score as _score_dim3
from dim4_consistency import compute_score as _score_dim4
from dim5_answer_f1 import compute_score as _score_dim5
from rubric_utils import (
    count_answer_tags,
    extract_answer,
    extract_search_queries,
    extract_retrieved_docs,
    extract_think_content,
    extract_question,
    get_golden_answers,
)


# ══════════════════════════════════════════════════════════════
#  可调配置（可通过 extra_info 覆盖）
# ══════════════════════════════════════════════════════════════

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "query_quality":    0.15,
    "repetition":       0.15,
    "doc_utilization":  0.20,
    "consistency":      0.15,
    "answer_f1":        0.35,
}

# EM Gate：答案 F1 = 0 时，过程分（维度1-4）乘以此衰减系数
_DEFAULT_EM_DECAY: float = 0.3

# 输出缩放（verl 默认 score=1.0，此处提供覆盖入口）
_DEFAULT_SCORE_SCALE: float = 1.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    5 维度 Rubric 奖励函数主入口。

    Args:
        data_source: 数据来源标识（verl 标准参数，保留未使用）
        solution_str: 模型生成的完整 rollout 文本
        ground_truth: 标准答案 dict，需包含 target/answer 字段
        extra_info: 附加信息（可选），支持以下键：
            - weights: Dict[str, float]  自定义各维度权重
            - em_decay: float            EM Gate 衰减系数
            - score_scale: float         输出缩放
            - question: str              原始问题文本

    Returns:
        float: 最终奖励值
    """
    # ── 解析 extra_info 中的覆盖配置 ──
    if extra_info is None:
        extra_info = {}
    weights = extra_info.get("weights", _DEFAULT_WEIGHTS)
    em_decay = extra_info.get("em_decay", _DEFAULT_EM_DECAY)
    score_scale = extra_info.get("score_scale", _DEFAULT_SCORE_SCALE)

    do_print = random.randint(1, 64) == 1

    # ── Step 0：tag spam 检查 ──
    open_count, close_count = count_answer_tags(solution_str)
    tag_spam = (open_count > 10 or close_count > 10)

    # ── Step 1：答案存在性检查 ──
    answer = extract_answer(solution_str)
    if answer is None:
        if do_print:
            _print_debug(solution_str, ground_truth, extra_info, {}, 0.0, 0.0,
                         weights, False, False)
        return 0.0

    # ── Step 2：5 维度独立打分 ──
    # 每个维度调用独立的 compute_score（开销略增，但保证模块独立可替换）
    dim_scores = {
        "query_quality":   _score_dim1(data_source, solution_str, ground_truth, extra_info),
        "repetition":      _score_dim2(data_source, solution_str, ground_truth, extra_info),
        "doc_utilization": _score_dim3(data_source, solution_str, ground_truth, extra_info),
        "consistency":     _score_dim4(data_source, solution_str, ground_truth, extra_info),
        "answer_f1":       _score_dim5(data_source, solution_str, ground_truth, extra_info),
    }

    # ── Step 3：加权线性聚合 ──
    process_score = (
        weights["query_quality"]   * dim_scores["query_quality"]   +
        weights["repetition"]      * dim_scores["repetition"]      +
        weights["doc_utilization"] * dim_scores["doc_utilization"] +
        weights["consistency"]     * dim_scores["consistency"]
    )
    f1_component = weights["answer_f1"] * dim_scores["answer_f1"]
    rubric_total = process_score + f1_component

    # ── Step 4：EM Gate ──
    # F1 = 0 时过程分衰减，保留梯度信号
    em_gate_active = False
    if dim_scores["answer_f1"] == 0.0:
        rubric_total = f1_component + process_score * em_decay
        em_gate_active = True

    # ── Step 5：tag spam 惩罚 ──
    if tag_spam:
        rubric_total /= 4.0

    # ── Step 6：输出缩放 ──
    final_reward = float(rubric_total * score_scale)

    # ── Step 7：调试日志 ──
    if do_print:
        _print_debug(solution_str, ground_truth, extra_info,
                     dim_scores, rubric_total, final_reward,
                     weights, em_gate_active, tag_spam)

    return final_reward


def _print_debug(
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
    dim_scores: dict,
    rubric_total: float,
    final_reward: float,
    weights: dict,
    em_gate_active: bool,
    tag_spam: bool,
) -> None:
    """调试日志打印（~1/64 概率触发）"""
    question = extract_question(solution_str, ground_truth, extra_info)
    golden = get_golden_answers(ground_truth)
    answer = extract_answer(solution_str)
    queries = extract_search_queries(solution_str)
    docs = extract_retrieved_docs(solution_str)

    print("=" * 60)
    print(f"[Rubric] Question:     {question[:80]!r}")
    print(f"[Rubric] Golden:       {golden}")
    print(f"[Rubric] Answer:       {answer!r}")
    print(f"[Rubric] Queries ({len(queries)}): {queries}")
    print(f"[Rubric] Docs    ({len(docs)}): "
          f"{[d[:50] + '...' for d in docs]}")
    print(f"[Rubric] -- Dim Scores --"
          f"{' -' * 20}")
    for k, v in dim_scores.items():
        bar = chr(9608) * int(v * 20)  # █
        print(f"[Rubric]   {k:<22} {v:.4f}  w={weights.get(k, 0):.2f}  |{bar:<20}|")
    em_str = f" (x {_DEFAULT_EM_DECAY} EM Gate)" if em_gate_active else ""
    tag_str = " (/4 tag spam)" if tag_spam else ""
    print(f"[Rubric] -- Aggregation --"
          f"{' -' * 20}")
    print(f"[Rubric]   rubric_total = {rubric_total:.4f}{em_str}{tag_str}")
    print(f"[Rubric]   final_reward = {final_reward:.4f}  (x scale {_DEFAULT_SCORE_SCALE})")
    print("=" * 60)
