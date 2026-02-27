# AutoFusion 优化实验方案

**目标**: 通过系统性改进，使NAS自动发现的架构在性能**和**效率上均优于人工设计

**时间预估**: 2-3周

---

## 一、问题诊断

### 当前状况
```
Phase 3搜索 (AI2D+3ep)          E2验证 (MMMU+100ep)
     ↓                                ↓
  arch_024 (0.952)              ~33% (NAS avg)
  arch_017 (0.933)              ~46% (FiLM)
  arch_021 (0.933)
     ↓                                ↓
  高Reward                      低准确率
```

### 根本原因

1. **评估器与目标脱节**: 3ep/AI2D无法预测100ep/MMMU性能
2. **效率约束不足**: Reward函数偏重准确率，忽视FLOPs
3. **搜索空间偏差**: 缺乏对高效架构的显式引导
4. **对比基准不充分**: 未与SOTA方法直接对比

---

## 二、实验目标

### 主要目标
1. **性能超越**: NAS架构在MMMU上准确率 > FiLM (~46%)
2. **效率超越**: NAS架构FLOPs < FiLM (6.29M)
3. **帕累托优势**: 在准确率-效率权衡上占据前沿

### 次要目标
4. **可复现性**: 多run稳定性验证
5. **泛化性**: 在VSR/MathVista上同样有效
6. **可解释性**: 发现的设计模式具有普适性

---

## 三、实验设计

### 阶段1: 生成阶段优化 (1-2周)

#### 1.1 改进评估器 (关键 🔴)

**方案A: 难数据集直接评估**
```python
# 当前配置 (问题)
'dataset': 'ai2d', 'train_epochs': 3

# 改进配置
evaluator_v2 = {
    'dataset': 'mmmu',           # 直接使用难数据集
    'train_epochs': 10,          # 增加训练深度
    'num_shots': 32,             # 增加样本数
    'batch_size': 8,
    'early_stopping': True,      # 早停防止过拟合
    'max_training_time': 300,    # 5分钟上限
}
```

**方案B: 多数据集联合评估**
```python
evaluator_v3 = {
    'datasets': ['ai2d', 'mmmu', 'vsr'],  # 多数据集
    'epochs_per_dataset': 5,
    'ensemble_reward': True,              # 综合得分
}
```

**推荐**: 先尝试方案A (更直接)

---

#### 1.2 改进Reward函数

**当前问题**:
```python
# 当前权重
weights = {
    'accuracy': 1.0,      # 主导
    'efficiency': 0.5,    # 次要
    'compile_success': 2.0,
}
```

**改进方案**:
```python
# 方案1: 效率优先
weights_v2 = {
    'accuracy': 1.0,
    'efficiency': 1.5,           # 提升效率权重
    'flops_penalty': 'exponential',  # 指数惩罚高FLOPs
}

# 方案2: 硬约束
constraints = {
    'max_flops': 10e6,        # 硬性FLOPs上限 (FiLM=6.29M)
    'max_params': 5e6,
    'reject_if_exceed': True,  # 超出直接reject
}

# 方案3: 帕累托优化
reward_type = 'pareto'  # 使用NSGA-II风格多目标
```

**推荐**: 方案1+2组合 (软约束+硬约束)

---

#### 1.3 改进搜索空间

**当前问题**: 搜索空间偏向复杂架构

**改进方案**:
```python
search_space_v2 = {
    # 增加高效操作选项
    'fusion_type': [
        'attention', 'bilinear', 'mlp',
        'gated', 'film',              # 添加FiLM风格
        'lightweight_attention',       # 轻量注意力
        'cross_modal_efficient'        # 高效跨模态
    ],

    # 限制网络深度
    'num_layers': {'type': 'int', 'low': 1, 'high': 4},  # 原来是6

    # 限制隐藏维度
    'hidden_dim': {'type': 'int', 'low': 128, 'high': 512, 'step': 64},  # 原来是1024

    # 强制包含残差连接 (提升效率)
    'use_residual': [True],  # 强制True

    # 添加剪枝选项
    'pruning_ratio': {'type': 'float', 'low': 0.0, 'high': 0.5},
}
```

---

#### 1.4 增加训练充分度

```python
# 配置对比

# 配置A: 快速 (用于探索)
fast_config = {
    'train_epochs': 5,
    'num_shots': 16,
    'search_iterations': 100,
}

# 配置B: 充分 (用于精细化)
deep_config = {
    'train_epochs': 20,
    'num_shots': 32,
    'search_iterations': 200,
    'learning_rate_schedule': 'cosine',
}
```

**策略**: 两阶段搜索
1. 用fast_config快速探索 (100 iter)
2. 用deep_config精细化Top 10 (50 iter)

---

### 阶段2: 评估阶段改进 (1周)

#### 2.1 扩展对比基准

**当前对比**:
- 简单: CLIPFusion, BilinearPooling, ConcatMLP
- 人工: FiLM, CrossModalAttention

**扩展方案**:

**方案A: 引入SOTA论文方法**
```python
# 需要复现的SOTA方法
sota_baselines = [
    'MCAN',           # CVPR 2019 - Deep Modular Co-Attention
    'BAN',            # NeurIPS 2018 - Bilateral Attention
    'DFAF',           # CVPR 2020 - Dynamic Fusion
    'ViLBERT',        # NeurIPS 2019 - Vision-Language BERT
    'LXMERT',         # EMNLP 2019 - Learning Cross-Modality
]
```

**方案B: 引入轻量SOTA**
```python
lightweight_baselines = [
    'S-MCAN',         # 简化版MCAN
    'TinyBERT-Vision', # 轻量ViT融合
    'MobileVLM',      # 移动端优化
]
```

**实施步骤**:
1. 调研AI2D/Diagram Understanding领域SOTA
2. 复现2-3个代表性的轻量级方法
3. 统一接口，纳入对比

---

#### 2.2 AI2D领域专项研究

**研究问题**: 为什么AI2D上所有方法都100%？

**调研方向**:
1. **数据集分析**:
   - AI2D样本数量/难度分布
   - 与ScienceQA、Geometry3K等对比
   - 16-shot是否过多？

2. **SOTA方法调研**:
   ```
   关键词: "AI2D benchmark", "diagram understanding",
           "scientific figure comprehension"
   会议: CVPR, ICCV, EMNLP, NeurIPS 2022-2024
   ```

3. **关键论文**:
   - 查找AI2D leaderboard
   - 分析SOTA方法的融合层设计
   - 提取可复现的轻量级设计

**预期产出**:
- 调研报告: `docs/ai2d_sota_survey.md`
- 复现2-3个轻量级SOTA baseline
- 发现AI2D性能瓶颈原因

---

#### 2.3 跨架构对比实验

**实验设计**:

```python
# 将NAS架构与SOTA融合层互换

experiment_A = {
    'name': 'NAS_fusion_on_SOTA_backbone',
    'description': '用我们的arch_024融合层替换SOTA方法的融合层',
    'datasets': ['ai2d', 'mmmu'],
}

experiment_B = {
    'name': 'SOTA_fusion_on_NAS_backbone',
    'description': '用SOTA融合层替换我们的架构',
    'datasets': ['ai2d', 'mmmu'],
}
```

**目的**: 分离融合层本身的效果与backbone的影响

---

## 四、具体实施计划

### Week 1: 快速验证

**Day 1-2: 改进评估器**
- [ ] 修改`real_data_evaluator.py`支持MMMU直接评估
- [ ] 测试10 epochs配置
- [ ] 验证可行性 (时间/显存)

**Day 3-4: 改进Reward**
- [ ] 实现FLOPs指数惩罚
- [ ] 添加10M FLOPs硬约束
- [ ] 本地测试

**Day 5-7: 小规模搜索**
- [ ] 运行50 iteration快速搜索
- [ ] 观察是否生成更高效的架构
- [ ] 分析初步结果

### Week 2: 完整搜索

**Day 8-10: 完整架构发现**
- [ ] 运行200 iteration搜索 (MMMU+10ep)
- [ ] 保存Top 20架构

**Day 11-12: AI2D SOTA调研**
- [ ] 调研2-3篇关键论文
- [ ] 记录SOTA融合层设计

**Day 13-14: 复现SOTA**
- [ ] 复现1个轻量级SOTA baseline
- [ ] 验证在我们框架下的性能

### Week 3: 全面评估

**Day 15-17: E3扩展评估**
- [ ] 对New-NAS + SOTA + 原有Baseline进行E1/E2评估
- [ ] 生成对比表格

**Day 18-19: 消融实验**
- [ ] 验证每个改进的贡献
- [ ] Reward函数对比
- [ ] 评估器对比

**Day 20-21: 论文准备**
- [ ] 整理实验结果
- [ ] 准备可视化图表

---

## 五、预期结果与判断标准

### 成功标准 (必须满足)

| 指标 | 当前 | 目标 | 判断标准 |
|------|------|------|----------|
| **MMMU准确率** | ~33% (NAS avg) | **>46%** | 超越FiLM |
| **FLOPs** | ~50M (NAS avg) | **<6.29M** | 比FiLM更高效 |
| **帕累托前沿** | - | **占据左上** | 准确率↑ 效率↑ |

### 进阶目标 (理想情况)

| 指标 | 目标 |
|------|------|
| 多数据集优势 | AI2D/VSR/MMMU上均优于人工设计 |
| 统计显著性 | p < 0.05 (E7验证) |
| 设计模式 | 发现可解释的、普适的设计原则 |

---

## 六、风险控制

### 风险1: MMMU评估太慢
**影响**: 搜索时间从30分钟→数小时
**应对**:
- 使用早停 (验证集不提升即停)
- 限制最大训练时间 (5分钟/架构)
- 使用更轻量的backbone (如ViT-B/16代替L/14)

### 风险2: 过拟合MMMU
**影响**: 在AI2D上性能下降
**应对**:
- 多数据集联合评估
- 保留AI2D快速验证作为 sanity check

### 风险3: SOTA复现困难
**影响**: 无法公平对比
**应对**:
- 优先选择有开源代码的论文
- 简化复杂SOTA，保留核心融合机制

---

## 七、实验产出

### 代码
- `experiment/phase4_optimized/` - 改进版搜索框架
- `expv2/E2_sota_comparison/` - SOTA对比实验

### 文档
- `docs/ai2d_sota_survey.md` - SOTA调研报告
- `OPTIMIZATION_RESULTS.md` - 实验结果汇总

### 论文素材
- 对比表格 (NAS vs SOTA vs 人工设计)
- 帕累托前沿图
- 消融实验分析

---

## 八、决策节点

### Checkpoint 1: Week 1结束
**问题**: MMMU直接评估是否可行？
- **Yes**: 继续Week 2完整搜索
- **No**: 改用AI2D+20ep或VSR评估

### Checkpoint 2: Week 2结束
**问题**: 是否发现高效+高性能架构？
- **Yes**: 进入Week 3全面评估
- **No**: 分析原因，调整搜索空间/Reward

### Checkpoint 3: Week 3结束
**问题**: 是否达成目标？
- **Yes**: 准备论文投稿
- **Partial**: 补充实验，调整论文故事
- **No**: 分析根本原因，决定是否需要Phase 5

---

*Plan Version: 1.0*
*Created: 2026-02-21*
*Estimated Duration: 2-3 weeks*
