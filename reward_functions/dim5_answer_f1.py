# dim5_answer_f1.py
#
#  维度5：答案 Token-level F1 奖励函数
#  独立可用，遵循 verl 自定义奖励函数接口
#
#  SQuAD 标准 Token-level F1。支持多个 golden answer，取最大 F1。
#
#  相比硬 EM：
#    - 对部分正确的答案更宽容，提供连续梯度信号
#    - 训练早期更稳定（不会出现整批 reward 全 0 的情况）
#
#  分值范围 [0, 1]
#
#  接口：
#    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rubric_utils import compute_token_f1, extract_answer, get_golden_answers


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    维度5：答案 Token-level F1。

    从 solution_str 提取 <answer> 标签内容，
    与 ground_truth 中 target/answer 字段计算 token-level F1。

    无 <answer> 标签时返回 0。

    Returns:
        float in [0, 1]
    """
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0

    golden_answers = get_golden_answers(ground_truth)
    if not golden_answers:
        return 0.0

    return compute_token_f1(answer, golden_answers)
