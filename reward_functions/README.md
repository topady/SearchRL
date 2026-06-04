# 5-Dimension Rubric Reward for Search-R1

Search-R1 检索增强强化学习的多维度奖励函数，替换原有 EM reward，提供连续、细粒度的训练信号。

---

## 目录

- [设计动机](#设计动机)
- [模块总览](#模块总览)
- [快速开始](#快速开始)
- [维度详解](#维度详解)
  - [D1: Query 质量](#d1-query-质量)
  - [D2: 重复搜索惩罚](#d2-重复搜索惩罚)
  - [D3: 检索结果利用度](#d3-检索结果利用度)
  - [D4: 推理逻辑一致性](#d4-推理逻辑一致性)
  - [D5: 答案 F1](#d5-答案-f1)
- [聚合策略](#聚合策略)
- [verl 配置](#verl-配置)
- [消融实验](#消融实验)
- [自定义权重](#自定义权重)
- [输入格式约定](#输入格式约定)
- [边界行为一览](#边界行为一览)

---

## 设计动机

| 问题 | 原有 EM | Rubric 方案 |
|------|---------|-------------|
| 训练早期全零 reward | 答案全错 → 梯度消失 | EM Gate 保留 30% 过程分 |
| 忽略搜索质量 | 只看答案对错 | D1+D2 显式建模 query 质量 |
| 无检索利用信号 | 不检查文档是否被阅读 | D3 衡量 doc→reasoning 覆盖 |
| 二元信号粗糙 | 对/错，无中间态 | 5 维连续信号 + F1 替代硬 EM |

---

## 模块总览

```
reward_functions/
├── rubric_utils.py             # 共享工具
├── dim1_query_quality.py       # D1: Query 质量         (w=0.15)
├── dim2_repetition.py          # D2: 重复搜索惩罚        (w=0.15)
├── dim3_doc_utilization.py     # D3: 检索结果利用度      (w=0.20)
├── dim4_consistency.py         # D4: 推理逻辑一致性      (w=0.15)
├── dim5_answer_f1.py           # D5: 答案 F1            (w=0.35)
└── rubric_merged.py            # 合并版（加权聚合 + EM Gate）
```

每个 `dim*.py` 文件都是**独立可用**的奖励函数，遵循 verl 接口：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
```

---

## 快速开始

### 1. 完整版（推荐）

```yaml
# verl config
custom_reward_function:
  path: reward_functions/rubric_merged.py
  name: compute_score
```

奖励范围：`[0, 1]`，分数越高表示整体质量越好。

### 2. 本地测试

```python
import sys
sys.path.insert(0, 'reward_functions')

from rubric_merged import compute_score

solution_str = """<think>...</think>
<search>Eiffel Tower construction year</search>
<information>Document [1](Title: Eiffel Tower) ...</information>
<think>...</think>
<answer>1889</answer>"""

ground_truth = {"target": "1889", "question": "When was the Eiffel Tower completed?"}

reward = compute_score("nq", solution_str, ground_truth)
print(f"Reward: {reward:.4f}")  # e.g., 0.8145
```

---

## 维度详解

### D1: Query 质量

**文件**: `dim1_query_quality.py` | **权重**: 0.15

对每条 `<search>` query 评估三个子维度后取均值：

| 子维度 | 权重 | 评分方法 |
|--------|------|----------|
| 长度合理性 | 30% | 梯形函数，3-12 词为满分区间 |
| 与问题相关性 | 50% | 去停用词 Jaccard similarity |
| 非直接复制 | 20% | query 中 >85% 词来自原问题时线性扣分 |

**奖励曲线**（单 query）：

| Query 特征 | 分值范围 |
|------------|----------|
| 高相关 + 理想长度 + 非复制 | ~0.80-1.00 |
| 中等相关 / 偏短 | ~0.50-0.70 |
| 直接复制原问题 | ~0.20-0.40 |
| 无意义 / 极短 | ~0.05-0.20 |

---

### D2: 重复搜索惩罚

**文件**: `dim2_repetition.py` | **权重**: 0.15

计算 query 两两之间的 token-level Jaccard 相似度，相邻轮次惩罚权重 1.5x。

**非线性映射**：

```
相似度 < 0.30  → 轻微惩罚  (score > 0.90)
相似度 0.30~0.70 → 线性惩罚  (score: 0.91 → 0.30)
相似度 > 0.70  → 重惩罚    (score < 0.30)
```

| 场景 | 典型分值 |
|------|----------|
| 每轮搜索不同方面 | 0.90-1.00 |
| 有少量关键词重叠 | 0.80-0.90 |
| 大量重复搜索 | 0.20-0.50 |
| 完全相同 query | 0.00 |

---

### D3: 检索结果利用度

**文件**: `dim3_doc_utilization.py` | **权重**: 0.20

计算 reasoning（think + answer）tokens 对每篇检索文档 tokens 的覆盖率。

```
score = min(0.6 × max_coverage + 0.4 × mean_coverage, 1.0)
```

| 场景 | 典型分值 |
|------|----------|
| 多篇文档被深入引用 | 0.60-0.90 |
| 只用了部分文档 | 0.30-0.50 |
| 完全忽略检索结果 | 0.00-0.10 |
| 无检索文档 | 0.50（中性） |

---

### D4: 推理逻辑一致性

**文件**: `dim4_consistency.py` | **权重**: 0.15

| 子维度 | 权重 | 计算 |
|--------|------|------|
| 答案有推理支撑 | 60% | answer tokens 在 think 中的覆盖率 |
| 推理长度合理 | 40% | 梯形，20-500 词满分 |

**设计意图**：防止模型学会"跳过思考直接猜答案"或"无尽地胡言乱语"。

---

### D5: 答案 F1

**文件**: `dim5_answer_f1.py` | **权重**: 0.35

SQuAD 标准 token-level F1，支持多个 golden answer 取 max。

相比 EM 的优势：

| | EM | F1 |
|------|----|----|
| 部分正确 | 0 | 0.4-0.8 |
| 多词答案缺一词 | 0 | ~0.67 |
| 训练早期梯度 | 稀疏全零 | 连续信号 |

---

## 聚合策略

```
最终奖励 = [过程分(维度1-4) + F1分(维度5)] × EM Gate × 缩放

其中:
  过程分   = 0.15×D1 + 0.15×D2 + 0.20×D3 + 0.15×D4
  F1分     = 0.35×D5
  EM Gate  = 若 D5==0 → 过程分×0.3; 否则不衰减
```

**EM Gate 设计哲学**：

```
场景A: 答案正确 (F1>0)  → 过程分全额保留 → 模型学会"好过程→好结果"
场景B: 答案错误 (F1=0)  → 过程分衰减至30% → 仍有弱梯度，不会"死掉"
```

---

## verl 配置

### 基础配置

```yaml
# 在 verl 训练配置中添加
custom_reward_function:
  path: reward_functions/rubric_merged.py
  name: compute_score
```

### 多试验配置

```yaml
# 试验1: 完整版
custom_reward_function:
  path: reward_functions/rubric_merged.py
  name: compute_score

# 试验2: 只看答案（消融 D1-D4）
custom_reward_function:
  path: reward_functions/dim5_answer_f1.py
  # name 不设置，默认 compute_score

# 试验3: 自定义权重版（需要额外传参）
custom_reward_function:
  path: reward_functions/rubric_merged.py
  name: compute_score
```

---

## 消融实验

| 实验 | 配置 path | 研究目标 |
|------|-----------|----------|
| 完整版 | `rubric_merged.py` | 基准 |
| Answer Only | `dim5_answer_f1.py` | 过程维度是否有效 |
| Process Only | `rubric_merged.py` + F1 weight=0 | 过程信号独立性 |
| No EM Gate | `rubric_merged.py` + em_decay=1.0 | EM Gate 必要性 |
| Single Dim | 任意 `dim*.py` | 各维度单独效果 |

---

## 自定义权重

通过 `extra_info` 字典传递自定义参数（verl 自动将额外字段传入 extra_info）：

```python
# 在数据预处理阶段，向每条样本添加 custom_reward 字段
extra_info = {
    "weights": {
        "query_quality":    0.10,
        "repetition":       0.10,
        "doc_utilization":  0.25,
        "consistency":      0.10,
        "answer_f1":        0.45,
    },
    "em_decay": 0.3,
    "score_scale": 1.0,
}
```

---

## 输入格式约定

### solution_str 结构

模型生成的 rollout 必须遵循 Search-R1 格式：

```
<think>推理过程1</think>
<search>搜索 query 1</search>
<information>
Document [1](Title: 标题1) 正文1
Document [2](Title: 标题2) 正文2
</information>
<think>推理过程2（基于上轮检索结果）</think>
<search>搜索 query 2</search>
<information>
Document [1](Title: 标题3) 正文3
</information>
<think>推理过程3</think>
<answer>最终答案</answer>
```

### 标签说明

| 标签 | 用途 | D1 | D2 | D3 | D4 | D5 |
|------|------|:--:|:--:|:--:|:--:|:--:|
| `<think>` | 推理过程 | | | | ✓ | |
| `<search>` | 搜索 query | ✓ | ✓ | | | |
| `<information>` | 检索返回文档 | | | ✓ | | |
| `<answer>` | 最终答案 | | | ✓ | ✓ | ✓ |

### ground_truth 格式

```python
ground_truth = {
    "target": "1889",           # 标准答案（必填）
    "question": "...",          # 原始问题（推荐，影响 D1 质量）
    # 也支持:
    # "answer":  "1889",        # 等价于 target
    # "answers": ["1889", ...], # 多答案取 max F1
}
```

---

## 边界行为一览

| 输入情况 | 奖励 | 原因 |
|----------|------|------|
| 正常回答（答案正确 + 过程好） | ~0.80-0.90 | 各维度均衡 |
| 正常回答（答案正确 + 过程差） | ~0.50-0.60 | D1-D4 拉低 |
| 答案错误 + 过程好 | ~0.10-0.20 | EM Gate 衰减 |
| 答案错误 + 过程也差 | ~0.00-0.05 | 双低 |
| 无 `<answer>` 标签 | 0.00 | 无法提取答案 |
| 无 `<search>` 标签 | 0.50-0.85 | D1=0, D2=1, 其余正常 |
| 无 `<information>` 标签 | 0.53-0.85 | D3 走中性分 0.5 |
| 大量重复 `<answer>` (>10) | ÷4 | tag spam 惩罚 |
| 只有 1 轮搜索 | 正常 | D2 满分（无重复可罚） |

---

## 调试

合并版内置 1/64 概率打印 debug 日志，输出示例：

```
============================================================
[Rubric] Question:     'When was the Eiffel Tower completed?'
[Rubric] Golden:       1889
[Rubric] Answer:       '1889'
[Rubric] Queries (2):  ['Eiffel Tower construction...', 'Eiffel Tower finished...']
[Rubric] -- Dim Scores -- - - - - - - - - - - - - - - -
[Rubric]   query_quality          0.639  w=0.15  |████████████░░░░░░░░|
[Rubric]   repetition             0.946  w=0.15  |██████████████████░░|
[Rubric]   doc_utilization        0.384  w=0.20  |███████░░░░░░░░░░░░░|
[Rubric]   consistency            1.000  w=0.15  |████████████████████|
[Rubric]   answer_f1              1.000  w=0.35  |████████████████████|
[Rubric] -- Aggregation -- - - - - - - - - - - - - -
[Rubric]   rubric_total = 0.8145
[Rubric]   final_reward = 0.8145  (x scale 1.0)
============================================================
```

如需强制打印所有样本的日志，将 `rubric_merged.py` 中的 `random.randint(1, 64) == 1` 改为 `True`。
