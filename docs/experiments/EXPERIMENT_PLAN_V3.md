# AutoFusion 实验计划 V3 (整合评估器验证)

**Version**: 3.0
**Date**: 2026-02-11
**Status**: 待确认

---

## 核心理念

评估器验证是连接 "搜索" 和 "真实性能" 的关键桥梁。本计划将数据集验证与评估器架构验证整合为统一的 **评估器可靠性验证体系**。

---

## 更新后的实验架构

```
Phase 0: 基础设施验证 (已完成 ✅)
  ├─ 0.0: API 连接验证
  └─ 0.5: Mock vs Real API 对比

Phase 1: Prompt 策略对比 (待运行 ⏳)
  ├─ 目标: 确定最佳 Prompt 策略
  ├─ Generator: Real API (DeepSeek-V3)
  └─ 输出: 最佳 Prompt 模板

Phase 2: Controller 对比 + 评估器验证 (✅ 部分完成)
  ├─ 2.1: Controller 算法对比 (Mock) ✅
  │     └─ Evolution > PPO > GRPO > GDPO > Random
  │
  └─ 2.5: 评估器可靠性验证 (整合) ⭐ NEW
        ├─ 2.5.1: 数据集泛化验证 (4 datasets)
        ├─ 2.5.2: 评估深度验证 (1 vs 10 Epochs)
        └─ 2.5.3: 架构鲁棒性验证 (8 architectures)

Phase 3: 架构发现 (规划中 📋)
  ├─ 3.1: 最佳组合探索 (Evolution + Best Prompt)
  ├─ 3.2: 消融实验
  └─ 3.3: SOTA 刷榜
```

---

## Phase 2.5: 评估器可靠性验证 (整合版)

### 目标
建立对 Surgical Sandbox 评估器的完整信任：
1. **跨数据集泛化**: 评估器在多种任务上是否一致
2. **评估深度校准**: Quick Eval (1 Epoch) 能否预测 Full Eval (10 Epochs)
3. **架构类型鲁棒性**: 评估器对不同架构风格是否公平

---

### 2.5.1: 数据集泛化验证

**问题**: 评估器在不同类型的视觉-语言任务上是否可靠？

**数据集矩阵**:

| 数据集 | 任务类型 | 核心能力 | 难度 | 样本数 |
|--------|----------|----------|------|--------|
| **MMMU** | 综合学科问答 | 跨领域知识融合 | ⭐⭐⭐⭐⭐ | ~11K |
| **VSR** | 空间关系推理 | 视觉关系理解 (左/右/包含) | ⭐⭐⭐ | ~10K |
| **MathVista** | 视觉数学 | 几何图形理解 + 逻辑推理 | ⭐⭐⭐⭐ | ~6K |
| **AI2D** | 科学图表 | 细粒度图文对齐 (箭头/标注) | ⭐⭐⭐⭐ | ~4K |

**实验设计**:
```
输入: 5 个代表性架构 (来自 Phase 2.1)
      ├─ Evolution 发现的 Top-2
      ├─ PPO 发现的 Top-2
      └─ Random 采样 1 个

过程: 每个架构在 4 个数据集上评估

输出: 4×5 性能矩阵
```

**分析指标**:
- 跨数据集排名一致性 (Kendall's τ)
- 最佳架构的迁移能力
- 数据集间的性能相关性

---

### 2.5.2: 评估深度验证

**问题**: Quick Eval (1 Epoch) 能否可靠预测 Full Eval (10 Epochs)？

**实验设计**:
```
架构选择: 8 个 (多样化)
  ├─ Attention-based × 2
  ├─ Conv-based × 2
  ├─ Transformer-based × 2
  ├─ MLP-based × 1
  └─ Hybrid × 1

评估对比:
  ├─ Quick: 1 Epoch, 5 min
  └─ Full: 10 Epochs, 30 min

指标: Kendall's τ (排名相关性)
```

**相关性等级**:
- τ > 0.8: 优秀 (完全可替代)
- τ > 0.7: 良好 (可信)
- τ > 0.5: 一般 (谨慎使用)
- τ < 0.5: 不可信 (需要调整)

**失败处理**:
若 τ < 0.7，调整评估器：
- 增加 Epochs (1 → 3)
- 调整学习率
- 增加早停 patience

---

### 2.5.3: 架构鲁棒性验证

**问题**: 评估器是否对某些架构类型有偏见？

**验证维度**:
1. **架构风格**: 同一架构在不同 seed 下评估稳定性
2. **极端架构**: 极小/极大 hidden_dim 的评估公平性
3. **失败模式**: 编译失败 vs 运行失败 vs 收敛失败

**实验设计**:
```
选择 3 个代表性架构:
  ├─ 架构 A: Attention-based (Evolution Top-1)
  ├─ 架构 B: Conv-based (Evolution Top-2)
  └─ 架构 C: Random (基线)

每个架构跑 5 个 seeds:
  ├─ 记录最佳 reward
  ├─ 记录收敛速度
  └─ 记录失败率

分析: 评估器对哪种架构最稳定？
```

---

## 整合验证流程

### Step 1: 数据集泛化验证
**输入**: 5 个架构 (来自 Phase 2.1)
**输出**: 跨数据集性能矩阵 + 排名一致性
**时间**: ~1 天
**并行**: 4 GPUs (每个数据集一个)

### Step 2: 评估深度验证
**输入**: 8 个新架构 (多样化采样)
**输出**: 1 Epoch vs 10 Epochs 相关性报告
**时间**: ~1 天
**并行**: 2 GPUs (Quick vs Full 同时跑)

### Step 3: 架构鲁棒性验证
**输入**: 3 个代表性架构 × 5 seeds = 15 次运行
**输出**: 评估稳定性报告
**时间**: ~0.5 天
**并行**: 3 GPUs

### Step 4: 整合分析
**输出**: 评估器可靠性报告
- 哪些数据集可以用 Quick Eval？
- 哪些需要 Full Eval？
- 评估器推荐的最终配置

---

## 评估器验证通过后

### 输出物
1. **评估器配置指南**: 不同场景下的推荐配置
2. **置信度指标**: 每个评估结果的可靠性分数
3. **快速筛选策略**: 先用 1 Epoch 筛选，再用 10 Epochs 精修

### 影响
- **Phase 1**: 使用验证后的评估器配置
- **Phase 3**: 大规模搜索时采用快速筛选 + 精修策略
- **论文**: 评估器验证作为 Method 部分的重要章节

---

## 完整实验时间线

| 阶段 | 任务 | 时间 | 依赖 |
|------|------|------|------|
| Week 1 | Phase 2.5.1 数据集泛化 | 1-2 天 | MMMU/VSR/MathVista/AI2D 数据准备 |
| Week 1 | Phase 2.5.2 评估深度 | 1 天 | - |
| Week 1-2 | Phase 2.5.3 鲁棒性 | 1 天 | - |
| Week 2 | Phase 2.5 整合分析 | 0.5 天 | 前三个完成 |
| Week 2-3 | Phase 1 Prompt 对比 | 3-5 天 | 评估器验证通过 |
| Week 3-4 | Phase 3 架构发现 | 5-7 天 | Phase 1 完成 |
| Week 4-5 | 论文撰写 | 7 天 | 所有实验完成 |

---

## 关键决策点

### 决策 1: 数据集选择
**必须包含**:
- MMMU: 作为综合能力的金标准
- VSR: 纯视觉关系，无文本干扰
- MathVista: 需要强视觉感知
- AI2D: 需要细粒度图文对齐

**可选添加**:
- GQA: 场景图推理
- OK-VQA: 需要外部知识

### 决策 2: 相关性阈值
**建议**: τ > 0.7 视为可信
- 若 τ = 0.6-0.7: 可用但需标注不确定性
- 若 τ < 0.6: 必须调整为 3 Epochs

### 决策 3: 评估器配置
**推荐配置**:
```yaml
evaluator:
  quick_eval:
    epochs: 1
    use_for: [screening, ranking]
    confidence: medium

  standard_eval:
    epochs: 3  # 如果 1 Epoch τ < 0.7
    use_for: [selection]
    confidence: high

  full_eval:
    epochs: 10
    use_for: [final_validation, leaderboard]
    confidence: very_high
```

---

## 下一步行动

待确认后:
1. 下载 MMMU/VSR/MathVista/AI2D 数据集
2. 实现 MultiDatasetEvaluator 类
3. 实现相关性分析脚本 (kendall_tau.py)
4. 创建评估器验证运行脚本
5. 启动 Phase 2.5.1 (数据集泛化)

---

*Last Updated: 2026-02-11*
*Status: 待确认*
