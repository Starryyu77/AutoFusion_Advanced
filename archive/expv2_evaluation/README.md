# ExpV2: NAS vs Human Design - 全面对比实验

**目标**: 系统性评估自动化NAS与人工设计的多模态融合架构

**核心发现**: 简单评估器(AI2D)无法预测复杂任务性能；在MMMU等难数据集上，人工设计(FiLM)优于NAS发现架构。

---

## 实验分类

```
expv2/
├── E1_main_evaluation/        # ✅ P0: AI2D主实验 (已完成)
├── E2_cross_dataset/          # ✅ P0: 跨数据集泛化 (已完成)
├── E3_pareto_analysis/        # P1: 帕累托前沿分析
├── E4_correlation/            # P1: 3ep vs 100ep相关性
├── E5_ablation/               # P1: 消融实验
├── E6_design_patterns/        # P1: 设计模式分析
├── E7_statistical_test/       # P0: 统计显著性检验
└── shared/                    # 共享组件
    ├── baselines/             # 5个人工设计基线
    ├── discovered/            # 8个NAS发现架构
    └── evaluation/            # 统一评估框架
```

---

## 核心实验结果 (P0)

### E1: AI2D主实验 ✅ 已完成

**目标**: 验证NAS vs 人工设计的性能差距

**配置**:
- 架构: 13个 (8 NAS + 5 Baseline)
- 训练: 100 epochs, 3 runs
- 数据集: AI2D

**关键发现**:

| 类型 | 代表架构 | 准确率 | FLOPs | 结论 |
|------|---------|--------|-------|------|
| 简单架构 | CLIPFusion | **100%** | **2.36M** ✅ | 最高效 |
| 人工设计 | FiLM | **100%** | 6.29M | 准确且高效 |
| NAS发现 | arch_022 | **100%** | 12.34M | 准确但效率低 |

**启示**: AI2D过于简单，所有架构均达100%，无法区分优劣。

```bash
# 查看E1详细报告
cat E1_main_evaluation/results/E1_DETAILED_REPORT.md
```

---

### E2: 跨数据集泛化 ✅ 已完成

**目标**: 验证架构在难数据集上的泛化能力

**配置**:
- 数据集: VSR, MathVista, MMMU
- 架构: 13个 (同E1)
- 训练: 100 epochs, 3 runs

**关键发现**:

| 数据集 | 难度 | NAS最佳 | 人工最佳 | 结论 |
|--------|------|---------|----------|------|
| **VSR** | 中等 | ~50% | ~50% | 二分类随机水平 |
| **MathVista** | ? | 100% | 100% | 验证集可能过小 |
| **MMMU** | 困难 | **~33%** | **~46%** | **FiLM > arch_017** |

**MMMU详细结果**:

| 排名 | 架构 | 类型 | 平均准确率 | FLOPs |
|------|------|------|-----------|-------|
| 1 | **FiLM** | 人工设计 | **~46%** | 6.29M |
| 2 | arch_017 | NAS | ~33% | 13.20M |
| 3 | CrossModalAttention | 人工 | ~25% | 31.82M |
| 4 | arch_021 | NAS | ~21% | 13.63M |

**启示**: 在复杂任务上，人工设计的FiLM显著优于NAS架构。

```bash
# 运行E2实验
python E2_cross_dataset/scripts/run_E2.py --dataset all --gpu 0

# 查看结果
ls E2_cross_dataset/results/
```

---

## 效率对比分析

### FLOPs排名 (从低到高)

| 排名 | 架构 | 类型 | FLOPs | vs CLIPFusion |
|------|------|------|-------|---------------|
| 1 | **CLIPFusion** | 简单 | **2.36M** | 1.0× |
| 2 | BilinearPooling | 简单 | 2.88M | 1.2× |
| 3 | ConcatMLP | 简单 | 4.93M | 1.7× |
| 4 | FiLM | 人工 | 6.29M | 2.7× |
| 5 | arch_022 | NAS | 12.34M | 5.2× |
| ... | ... | ... | ... | ... |
| 13 | arch_008 | NAS | 206.00M | 87.3× |

**结论**: NAS架构平均FLOPs是人工设计的**6.5倍**。

---

## 共享组件

### 基线架构 (人工设计)

```python
from shared.baselines import (
    ConcatMLP,           # 简单拼接+MLP
    BilinearPooling,     # 双线性池化
    CrossModalAttention, # 跨模态注意力
    CLIPFusion,          # CLIP风格融合
    FiLM                 # 特征调制
)
```

### 发现架构 (NAS生成)

```python
from shared.discovered import DISCOVERED_ARCHITECTURES

# 获取arch_024 (Phase 3 Best: 0.952)
arch = DISCOVERED_ARCHITECTURES['arch_024']()
```

### 评估接口

```python
from shared.evaluation import FullEvaluator

# 创建评估器
evaluator = FullEvaluator(
    dataset='mmmu',      # ai2d, vsr, mathvista, mmmu
    epochs=100,
    num_runs=3
)

# 评估架构
results = evaluator.evaluate(arch, 'arch_name')
```

---

## 快速开始

### 1. 本地测试

```bash
cd expv2

# E1快速测试 (10 epochs)
python E1_main_evaluation/scripts/run_E1.py --mode quick --gpu 0

# E2单数据集测试
python E2_cross_dataset/scripts/run_E2.py --dataset mmmu --gpu 0
```

### 2. 服务器完整实验

```bash
# E1完整评估
bash E1_main_evaluation/scripts/run_on_server.sh 2

# E2跨数据集
bash E2_cross_dataset/scripts/run_E2_all.sh
```

### 3. 分析结果

```bash
# 查看E1详细报告
cat E1_main_evaluation/results/E1_DETAILED_REPORT.md

# 查看E2结果
ls E2_cross_dataset/results/

# 查看完整汇总
cat ../EXPERIMENTS_SUMMARY.md
```

---

## 实验状态

| 实验 | 状态 | 优先级 | 关键结果 |
|------|------|--------|----------|
| **E1** | ✅ **完成** | P0 | AI2D过于简单，无法区分 |
| **E2** | ✅ **完成** | P0 | 人工设计 > NAS (MMMU) |
| E3 | 📋 待开始 | P0 | 帕累托前沿分析 |
| E4 | 📋 待开始 | P1 | 3ep vs 100ep相关性 |
| E5 | 📋 待开始 | P1 | 消融实验 |
| E6 | 📋 待开始 | P1 | 设计模式分析 |
| E7 | 📋 待开始 | P0 | 统计显著性检验 |

---

## 关键发现总结

### 1. 评估器局限性

Phase 3使用AI2D+3epochs快速评估发现的架构，在复杂任务上表现不佳：

```
Phase 3 Best (arch_024): Reward 0.952
    ↓
E1 AI2D 100ep: 100% (所有架构)
    ↓
E2 MMMU 100ep: ~33% (NAS), ~46% (FiLM)
```

**启示**: 简单评估器无法预测复杂任务性能。

### 2. 三类架构对比

| 类型 | 简单任务 | 复杂任务 | 效率 | 代表 |
|------|---------|---------|------|------|
| **简单架构** | 完美 | 不足 | **最高** | CLIPFusion |
| **人工设计** | 完美 | **最佳** | 中等 | FiLM |
| **NAS发现** | 完美 | 一般 | 最低 | arch_017 |

### 3. 论文故事调整

**原计划**: "NAS发现优于人工设计"

**实际发现**: "人工设计在复杂任务上更鲁棒，NAS搜索空间需要改进"

---

**状态**: E1 & E2 已完成 ✅
**下一步**: E3帕累托分析，E7统计检验
