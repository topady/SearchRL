# reward_functions/__init__.py
#
#  5 维度 Rubric 奖励函数包
#
#  模块说明：
#    rubric_utils.py        — 共享工具（归一化、解析、分词）
#    dim1_query_quality.py  — 维度1：搜索 Query 质量
#    dim2_repetition.py     — 维度2：重复搜索惩罚
#    dim3_doc_utilization.py— 维度3：检索结果利用度
#    dim4_consistency.py    — 维度4：推理逻辑一致性
#    dim5_answer_f1.py      — 维度5：答案 Token-level F1
#    rubric_merged.py       — 合并版（5维度 + EM Gate + 加权聚合）
#
#  verl 配置示例：
#
#    # 使用完整 5 维度奖励
#    custom_reward_function:
#      path: reward_functions/rubric_merged.py
#      name: compute_score
#
#    # 单独使用某个维度（用于消融实验）
#    custom_reward_function:
#      path: reward_functions/dim5_answer_f1.py
#      name: compute_score
#
#    # 多试验切换
#    custom_reward_function:
#      path: reward_functions/rubric_merged.py
#      name: compute_score
#      # 通过 extra_info 传递自定义权重
#

from rubric_merged import compute_score

__all__ = ["compute_score"]
