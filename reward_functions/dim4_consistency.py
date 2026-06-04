# dim4_consistency.py
#
#  维度4：推理逻辑一致性奖励函数
#  独立可用，遵循 verl 自定义奖励函数接口
#
#  衡量最终答案是否有 thinking 过程支撑，以及推理长度是否合理。
#
#  子维度：
#    A. 答案有推理支撑 (60%) — answer 关键词在 thinking 中的覆盖率
#    B. 推理长度合理性 (40%) — 梯形函数，理想区间 20-500 词
#
#  分值范围 [0, 1]
#
#  接口：
#    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float


import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rubric_utils import extract_answer, extract_think_content, tokenize


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> float:
    """
    维度4：推理逻辑一致性。

    Returns:
        float in [0, 1]
    """
    think_content = extract_think_content(solution_str)
    answer = extract_answer(solution_str) or ""

    # ── A. 答案有推理支撑 ──
    if think_content and answer:
        answer_tokens = set(tokenize(answer))
        think_tokens = set(tokenize(think_content))
        grounding = (
            len(answer_tokens & think_tokens) / len(answer_tokens)
            if answer_tokens else 0.0
        )
    else:
        grounding = 0.0

    # ── B. 推理长度合理性（梯形：理想 20-500 词） ──
    n = len(think_content.split()) if think_content else 0
    if n < 20:
        length_score = n / 20.0
    elif n <= 500:
        length_score = 1.0
    elif n <= 1000:
        length_score = 1.0 - 0.30 * (n - 500) / 500
    else:
        length_score = 0.70

    return float(0.60 * grounding + 0.40 * length_score)
