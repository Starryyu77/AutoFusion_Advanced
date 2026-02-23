# Auto-Fusion Experiment Framework

自进化多模态神经架构搜索(NAS)系统实验框架

## ⚠️ 重要发现

本实验框架系统性地比较了自动化NAS与人工设计的多模态融合架构。**核心发现：简单评估器无法预测复杂任务性能，人工设计在复杂任务上仍具优势。**

---

## 项目概述

AutoFusion 是一个用于自动设计多模态融合架构的神经网络架构搜索(NAS)系统。通过系统性对比 RL 算法、提示策略对多模态架构生成的影响，我们发现：

1. **评估器选择至关重要**：简单数据集(AI2D)无法区分架构优劣
2. **人工设计价值**：在复杂任务上，专家设计的架构表现更稳健
3. **效率差距显著**：NAS发现的架构平均FLOPs是人工设计的6.5倍

**核心流程**: Controller → Generator(LLM) → Evaluator(Sandbox) → Reward

---

## 实验结果总览

### Phase 1-3: 架构搜索 (已完成 ✅)

使用AI2D数据集+3epochs快速评估搜索架构：

| Phase | 名称 | 最佳Reward | 发现架构数 |
|-------|------|-----------|-----------|
| 1 | Prompt对比 | 0.873 | - |
| 2.5 | 评估器验证 | - | - |
| 3 | 架构发现 | **0.952** | 26个 |

**⚠️ 局限性**: Phase 3的reward基于AI2D+3epochs，在复杂任务上泛化能力不足。

### E1/E2: 完整评估 (已完成 ✅)

使用100epochs完整训练评估13个架构(8 NAS + 5 Baseline)：

| 数据集 | 难度 | NAS最佳 | 人工最佳 | 结论 |
|--------|------|---------|----------|------|
| **AI2D** | 简单 | 100% | 100% | 过于简单，无法区分 |
| **VSR** | 中等 | ~50% | ~50% | 二分类随机水平 |
| **MathVista** | ? | 100% | 100% | 验证集可能过小 |
| **MMMU** | 困难 | **~33%** | **~46%** | **人工设计 > NAS** |

**核心结论**:
- 简单任务：简单架构最高效 (CLIPFusion: 2.36M FLOPs)
- 复杂任务：人工设计更鲁棒 (FiLM > arch_017)
- NAS效率问题：平均FLOPs是人工设计的6.5倍

---

## 项目结构

```
experiment/                 # 原始实验 (Phase 1-3)
├── phase1_prompts/        # Prompt策略对比
├── phase2_controllers/    # Controller算法对比
├── phase2_5/              # 评估器验证
├── phase3_discovery/      # 架构发现 (Best: 0.952)
└── evaluators/            # RealDataFewShotEvaluator

expv2/                      # 完整评估实验 (E1-E7)
├── E1_main_evaluation/    # AI2D 100epochs完整评估
├── E2_cross_dataset/      # 跨数据集泛化测试
└── shared/                # 共享组件
    ├── baselines/         # 5个人工设计基线
    ├── discovered/        # 8个NAS发现架构
    └── evaluation/        # 统一评估框架

docs/experiments/           # 实验报告
├── PHASE1_REPORT.md
├── PHASE3_DISCOVERY_RESULTS.md
└── ...

EXPERIMENTS_SUMMARY.md      # 完整结果汇总 ⭐
```

---

## 关键发现详解

### 1. AI2D过于简单

**E1结果**: 所有13个架构均达到100%准确率

| 类型 | 代表架构 | 准确率 | FLOPs |
|------|---------|--------|-------|
| 简单架构 | CLIPFusion | 100% | **2.36M** ✅ |
| 人工设计 | FiLM | 100% | 6.29M |
| NAS发现 | arch_022 | 100% | 12.34M |

**启示**: 16-shot AI2D无法作为可靠的架构评估基准。

### 2. 复杂任务上人工设计更优

**E2-MMMU结果**: (8样本验证集)

| 排名 | 架构 | 类型 | 准确率 | FLOPs |
|------|------|------|--------|-------|
| 1 | **FiLM** | 人工设计 | **~46%** | 6.29M |
| 2 | arch_017 | NAS | ~33% | 13.20M |
| 3 | arch_021 | NAS | ~21% | 13.63M |

**启示**: 特征调制(FiLM)机制在复杂多学科问题上表现更稳健。

### 3. NAS效率问题

| 指标 | NAS平均 | 人工设计平均 | 差距 |
|------|---------|-------------|------|
| FLOPs | ~50M | ~8M | **6.5×** |
| 延迟 | ~2ms | ~0.3ms | **6.7×** |

**启示**: 当前NAS搜索空间缺乏有效的效率约束。

---

## 快速开始

### 运行完整评估 (E1)

```bash
cd expv2

# 快速测试 (10 epochs)
python E1_main_evaluation/scripts/run_E1.py --mode quick --gpu 0

# 完整评估 (100 epochs, 3 runs)
bash E1_main_evaluation/scripts/run_on_server.sh 2
```

### 跨数据集测试 (E2)

```bash
# 运行所有数据集
python E2_cross_dataset/scripts/run_E2.py --dataset all --gpu 0

# 单独运行MMMU
python E2_cross_dataset/scripts/run_E2.py --dataset mmmu --gpu 0
```

### 查看结果

```bash
# E1详细报告
cat expv2/E1_main_evaluation/results/E1_DETAILED_REPORT.md

# 完整汇总
cat EXPERIMENTS_SUMMARY.md
```

---

## 实验状态

| 实验 | 状态 | 关键结果 |
|------|------|----------|
| Phase 1 (Prompt) | ✅ 完成 | FewShot最佳 (0.873) |
| Phase 2.5 (验证器) | ✅ 完成 | AI2D+3epochs配置 |
| Phase 3 (发现) | ✅ 完成 | 26架构, Best: 0.952 |
| **E1 (完整评估)** | ✅ **完成** | **AI2D过于简单** |
| **E2 (跨数据集)** | ✅ **完成** | **人工设计 > NAS** |
| E3 (帕累托分析) | 📋 待开始 | - |

---

## 论文贡献重新定位

### 原计划 vs 实际发现

| 原计划 | 实际发现 |
|--------|----------|
| NAS发现优于人工设计 | 人工设计在复杂任务上更鲁棒 |
| AI2D是合适的评估基准 | AI2D过于简单，需要难数据集 |
| NAS架构更高效 | NAS效率是人工设计的1/6.5 |

### 调整后贡献

1. **评估器设计洞察**: 揭示了简单评估器(AI2D)与复杂任务性能之间的脱节
2. **系统性对比框架**: 提供了完整的NAS vs 人工设计对比方法论
3. **人工设计价值验证**: 证明了专家知识在复杂多模态任务上的持续重要性
4. **搜索空间分析**: 识别了当前NAS搜索空间的效率缺陷

---

## 技术架构

| 组件 | 实现 | 说明 |
|------|------|------|
| Controllers | PPO, GRPO, GDPO, Evolution, CMA-ES, Random | 6种搜索算法 |
| Generators | CoT, FewShot, Critic, Shape, RolePlay | 5种提示策略 |
| Evaluator | RealDataFewShotEvaluator | 真实数据few-shot评估 |
| Reward | MultiObjective + Exponential | 准确率+效率+有效性 |

---

## 服务器配置

- **Host**: `ntu-gpu43` / `gpu43.dynip.ntu.edu.sg`
- **GPU**: 4 × NVIDIA RTX A5000 (24GB)
- **项目路径**: `/usr1/home/s125mdg43_10/AutoFusion_Advanced/`

---

## GitHub 仓库

https://github.com/Starryyu77/AutoFusion_Advanced

---

*Last Updated: 2026-02-21*
*Status: Phase 1-3 & E1-E2 Complete ✅*
*Key Finding: Human-designed architectures outperform NAS on complex tasks*
