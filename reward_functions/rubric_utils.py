# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# ────────────────────────────────────────────────────────────────
#  共享工具模块 — 供各维度奖励函数和合并版共同使用
# ────────────────────────────────────────────────────────────────

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Union


# ══════════════════════════════════════════════════════════════
#  停用词表（轻量，英中混合）
# ══════════════════════════════════════════════════════════════

_STOP_WORDS: set = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "will", "would", "could", "should", "may",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "it", "its", "i", "you", "he", "she", "we", "they", "this", "that",
    "what", "when", "where", "who", "how", "which", "with", "from",
}


# ══════════════════════════════════════════════════════════════
#  SQuAD 标准回答归一化
# ══════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """小写 + 去冠词 + 去标点 + 去多余空格"""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ══════════════════════════════════════════════════════════════
#  分词 & 结构化解析
# ══════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """分词 + 过滤停用词，用于各维度的词汇匹配"""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def extract_answer(solution_str: str) -> Optional[str]:
    """提取最后一个 <answer>...</answer> 中的内容"""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def extract_think_content(solution_str: str) -> str:
    """提取所有 <think>...</think> 内容，合并返回"""
    match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if not match:
        # 兼容 multi-turn：合并所有 think 块
        blocks = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
        return " ".join(b.strip() for b in blocks) if blocks else ""
    # 只有一个块时直接取，同时检查是否有多块
    blocks = re.findall(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    return " ".join(b.strip() for b in blocks) if blocks else match.group(1).strip()


def extract_search_queries(solution_str: str) -> List[str]:
    """提取所有 <search>...</search> query（按出现顺序）"""
    queries = re.findall(r"<search>(.*?)</search>", solution_str, re.DOTALL)
    return [q.strip() for q in queries if q.strip()]


def extract_retrieved_docs(solution_str: str) -> List[str]:
    """
    提取检索文档内容，兼容多种格式：
      - <information>...</information>  （Search-R1 默认）
      - <result>...</result>            （部分变体）

    返回每个文档的完整文本块（含 "Document [N](Title: ...)" 格式化前缀）。
    """
    docs = re.findall(r"<information>(.*?)</information>", solution_str, re.DOTALL)
    if not docs:
        docs = re.findall(r"<result>(.*?)</result>", solution_str, re.DOTALL)
    # 每个 <information> 块可能包含多篇文档，先按块收集再展开
    all_docs: List[str] = []
    for block in docs:
        block = block.strip()
        if not block:
            continue
        # 按 "Document [" 拆分，保留前缀
        parts = re.split(r"(?=Document \[\d+\])", block)
        for part in parts:
            part = part.strip()
            if part:
                all_docs.append(part)
    return all_docs


def extract_question(
    solution_str: str,
    ground_truth: dict,
    extra_info: dict = None,
) -> str:
    """
    提取原始问题。优先级：
      1. extra_info 中的 question / prompt
      2. ground_truth 中的 question / input / query
      3. solution_str 中解析 user turn
    """
    # extra_info 优先
    if extra_info:
        for key in ("question", "prompt", "query"):
            if key in extra_info and extra_info[key]:
                return str(extra_info[key])

    # ground_truth 其次
    for key in ("question", "input", "query"):
        if key in ground_truth and ground_truth[key]:
            return str(ground_truth[key])

    # fallback：从 solution_str 解析
    for pattern in [
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
        r"Human:\s*(.*?)\n",
        r"Question:\s*(.*?)(?:\n|$)",
    ]:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()[:500]

    return ""


def get_golden_answers(ground_truth: dict) -> Union[str, List[str]]:
    """提取标准答案，支持 target/answer/answers 字段"""
    for key in ("target", "answer", "answers"):
        if key in ground_truth and ground_truth[key]:
            return ground_truth[key]
    return ""


def count_answer_tags(solution_str: str) -> tuple:
    """返回 (open_tag_count, close_tag_count)"""
    return solution_str.count("<answer>"), solution_str.count("</answer>")


# ══════════════════════════════════════════════════════════════
#  数学工具
# ══════════════════════════════════════════════════════════════

def safe_mean(values: List[float], default: float = 0.0) -> float:
    """安全取均值，空列表返回 default"""
    if not values:
        return default
    return sum(values) / len(values)


def trapezoid_score(
    n: float,
    ideal_min: float,
    ideal_max: float,
    floor_left: float = 0.0,
    floor_right: float = 0.0,
    ramp_in: float = 0.0,
    ramp_out: float = 0.0,
) -> float:
    """
    梯形函数打分，通用化 length 类评分。

    floor_left ──ramp_in──▶ 1.0 ──ramp_out──▶ floor_right
              <─ ideal_min ── ideal_max ──>

    n 在 [ideal_min, ideal_max] → 1.0
    n 在 [ideal_min-ramp_in, ideal_min] → floor_left → 1.0 线性
    n 在 [ideal_max, ideal_max+ramp_out] → 1.0 → floor_right 线性
    n 在区间外 → 对应 floor 值
    """
    if n < ideal_min - ramp_in:
        return floor_left
    elif n < ideal_min:
        return floor_left + (1.0 - floor_left) * (n - ideal_min + ramp_in) / ramp_in
    elif n <= ideal_max:
        return 1.0
    elif n <= ideal_max + ramp_out:
        return 1.0 - (1.0 - floor_right) * (n - ideal_max) / ramp_out
    else:
        return floor_right


# ══════════════════════════════════════════════════════════════
#  Token-level F1（独立复用）
# ══════════════════════════════════════════════════════════════

def compute_token_f1(
    prediction: str,
    golden_answers: Union[str, List[str]],
) -> float:
    """SQuAD 标准 token-level F1，支持多 golden answer 取 max"""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    pred_norm = normalize_answer(prediction)
    pred_tokens = pred_norm.split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for golden in golden_answers:
        gt_norm = normalize_answer(golden)
        gt_tokens = gt_norm.split()
        if not gt_tokens:
            continue

        pred_counter = Counter(pred_tokens)
        gt_counter = Counter(gt_tokens)
        common = sum((pred_counter & gt_counter).values())

        if common == 0:
            continue

        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return float(best_f1)
