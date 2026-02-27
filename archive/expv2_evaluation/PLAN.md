# ExpV2: AutoFusion vs Human Design - 全面对比实验

## 项目核心价值

**目标**: 证明自动化NAS可以找到比人工设计更好的多模态融合层

**价值主张**:
- **BETTER**: 性能优于传统人工设计架构
- **FASTER**: 设计时间远少于人工设计（31.5分钟 vs 数小时-数天）
- **MORE**: 一次搜索发现26个高质量架构，提供更多选择

---

## 实验背景

### 已完成实验

| Phase | 内容 | 最佳结果 |
|-------|------|----------|
| Phase 1 | Prompt策略对比 | FewShot最佳 (0.873) |
| Phase 2.1 | Controller对比 | Evolution最佳 (9.8) |
| Phase 2.5 | Evaluator验证 | AI2D + 3 epochs |
| **Phase 3** | **架构发现** | **arch_024 (0.952)** - 26个架构 |

### 核心问题

Phase 3 的 **3 epochs few-shot 评估** 发现了很多架构，但：
1. **这些架构在完整训练下表现如何？**
2. **真的比人工设计的传统架构好吗？**
3. **NAS的设计效率 vs 人工设计效率？**

---

## 实验设计

### 对比维度

| 维度 | AutoFusion (NAS) | Human Design (Baselines) |
|------|------------------|--------------------------|
| **设计时间** | 31.5分钟 | 数小时-数天 |
| **架构数量** | 26个 | 5个经典设计 |
| **搜索空间** | 20+ 维度自动探索 | 专家经验设计 |
| **评估方式** | AI2D, 3 epochs → 100 epochs验证 | 直接100 epochs训练 |

### 测试架构

#### A. Phase 3 发现的架构 (Top 10)

| 排名 | 架构 | 3ep Reward | 设计类型 | 关键特点 |
|------|------|------------|----------|----------|
| 🥇 1 | **arch_024** | **0.952** | Hybrid | Bilinear+Transformer, hidden=218, layers=6 |
| 🥈 2 | arch_019 | 0.933 | Attention+MLP | cross-attention + MLP fusion, hidden=852 |
| 🥈 3 | arch_021 | 0.933 | Pure MLP | concat+MLP, hidden=1024, dense连接 |
| 4 | arch_012 | 0.906 | Cross-Modal | transformer-based, instance norm, gating |
| 5 | arch_025 | 0.899 | Hybrid | attention-based, hidden=211, 10 heads |
| 6 | arch_004 | 0.873 | MLP+Attention | serial连接, hidden=922, high dropout |
| 7 | arch_022 | 0.873 | Pure MLP | dense连接, hidden=894, residual |
| 8 | arch_015 | 0.850 | Gated | gated attention, parallel连接, mish激活 |
| 9 | arch_008 | 0.825 | Hybrid | bilinear, hidden=467, low dropout |
| 10 | arch_017 | 0.819 | Attention | attention+MLP, gating, layer scale |

**设计模式分析**:
- **Hybrid**: 4个 (024, 025, 008, 012) - NAS偏爱混合设计
- **MLP-based**: 3个 (021, 022, 004) - 简单但有效
- **Attention-based**: 3个 (019, 015, 017) - 跨模态交互

#### B. 传统人工设计基线 (5个)

| 基线 | 设计原理 | 参数量估算 | 复杂度 |
|------|----------|------------|--------|
| **ConcatMLP** | 拼接+MLP (最简单基线) | ~50M | ⭐ |
| **BilinearPooling** | 元素级乘法融合 | ~40M | ⭐⭐ |
| **CrossModalAttention** | 跨模态注意力 (ViLBERT风格) | ~80M | ⭐⭐⭐⭐ |
| **CLIPFusion** | 简单投影+相加 (CLIP风格) | ~20M | ⭐ |
| **FiLM** | 特征调制 (条件化融合) | ~60M | ⭐⭐⭐ |

---

## 实施阶段

### Phase 1: 基线实现与发现架构整理 ✅ 进行中

**1.1 传统基线实现** (已完成 ✅)
- [x] `baselines/concat_mlp.py` - 拼接+MLP
- [x] `baselines/bilinear_pooling.py` - 双线性池化
- [x] `baselines/cross_modal_attention.py` - 跨模态注意力
- [x] `baselines/clip_fusion.py` - CLIP风格
- [x] `baselines/film.py` - FiLM调制

**1.2 Phase 3 Top 10 架构整理**
- [ ] 从 `phase3_discovery/results_local/` 复制Top 10架构代码
- [ ] 统一接口格式 (vision_dim=768, language_dim=768, output_dim=768)
- [ ] 创建架构加载器

**1.3 评估框架搭建**
- [ ] 统一评估pipeline
- [ ] 支持任意融合模块的评估
- [ ] 指标收集: Accuracy, FLOPs, Params, Latency

### Phase 2: 完整评估 (核心实验)

**2.1 单数据集深度评估 (AI2D)**

```yaml
实验配置:
  dataset: ai2d
  train_epochs: 100
  batch_size: 32
  learning_rate: 1e-4
  num_runs: 3  # 3个随机种子取平均

评估指标:
  - final_accuracy: 最终测试准确率
  - best_accuracy: 最佳验证准确率
  - convergence_epoch: 收敛轮数
  - training_time: 训练时间
  - flops: 计算量
  - params: 参数量
  - latency: 推理延迟
```

**2.2 跨数据集泛化验证**

在4个数据集上测试Top 5架构:
- AI2D (主数据集)
- MMMU (多学科推理)
- VSR (空间推理)
- MathVista (数学推理)

目标: 验证架构的泛化能力

**2.3 效率-性能帕累托分析**

绘制帕累托前沿:
- X轴: FLOPs / Latency (效率)
- Y轴: Accuracy (性能)
- 对比NAS架构 vs 人工设计的位置

### Phase 3: 对比分析与报告

**3.1 统计显著性检验**

```python
# 对比方法
- 配对t检验: NAS vs Baseline
- 效应量计算 (Cohen's d)
- 置信区间估计
```

**3.2 设计效率分析**

| 指标 | AutoFusion | 人工设计 |
|------|------------|----------|
| 设计时间 | 31.5分钟 | 估算: 8小时/架构 |
| 架构产出 | 26个 | 1-5个 |
| 单位时间产出 | 0.82架构/分钟 | ~0.01架构/分钟 |
| 人力成本 | 自动化 | 专家时间 |

**3.3 架构设计模式分析**

- NAS偏好的设计模式是什么？
- 与人工设计有何不同？
- 哪些设计选择对性能最关键？

---

## 文件结构

```
expv2/
├── PLAN.md                           # 本文件
├── README.md                         # 快速开始指南
│
├── baselines/                        # 人工设计基线 ✅ 已完成
│   ├── __init__.py
│   ├── concat_mlp.py
│   ├── bilinear_pooling.py
│   ├── cross_modal_attention.py
│   ├── clip_fusion.py
│   └── film.py
│
├── discovered/                       # Phase 3 发现的架构
│   ├── __init__.py
│   ├── arch_024.py                   # Best (0.952)
│   ├── arch_019.py                   # 0.933
│   ├── arch_021.py                   # 0.933
│   ├── arch_012.py                   # 0.906
│   ├── arch_025.py                   # 0.899
│   ├── arch_004.py                   # 0.873
│   ├── arch_022.py                   # 0.873
│   ├── arch_015.py                   # 0.850
│   ├── arch_008.py                   # 0.825
│   └── arch_017.py                   # 0.819
│
├── evaluation/                       # 评估框架
│   ├── __init__.py
│   ├── base_evaluator.py             # 评估基类
│   ├── full_trainer.py               # 100 epochs训练器
│   ├── metrics.py                    # 指标计算
│   └── cross_validator.py            # 跨数据集验证
│
├── analysis/                         # 分析脚本
│   ├── compare.py                    # 对比分析
│   ├── statistical_test.py           # 统计检验
│   ├── plot_results.py               # 可视化
│   └── generate_report.py            # 报告生成
│
├── configs/                          # 配置文件
│   ├── baseline_config.yaml
│   ├── discovered_config.yaml
│   └── eval_config.yaml
│
├── scripts/                          # 运行脚本
│   ├── run_baseline_eval.py          # 评估所有基线
│   ├── run_discovered_eval.py        # 评估发现的架构
│   ├── run_all.py                    # 运行全部评估
│   └── run_on_server.sh              # 服务器脚本
│
├── results/                          # 实验结果
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 处理后数据
│   ├── figures/                      # 图表
│   └── report.md                     # 最终报告
│
└── tests/                            # 测试
    ├── test_baselines.py
    └── test_discovered.py
```

---

## 关键研究问题

### Q1: NAS架构真的比人工设计更好吗？
**假设**: AutoFusion Top 10 的平均性能 > 5个传统基线的平均性能

**验证方法**:
- 配对t检验 (p < 0.05)
- 效应量计算 (Cohen's d > 0.5)

### Q2: NAS的设计效率有多高？
**假设**: NAS在单位时间内产生的优质架构数量远超人工设计

**验证方法**:
- 对比设计时间 (31.5分钟 vs 估算专家时间)
- 对比架构产出 (26个 vs 1-5个)
- 计算设计效率比

### Q3: 最佳NAS架构的设计模式是什么？
**发现**: 从Phase 3结果看，NAS偏爱:
1. **Hybrid设计** (40%) - 组合多种融合算子
2. **中等hidden_dim** (200-900) - 不追求最大维度
3. **使用残差/密集连接** (70%) - 重视梯度流动
4. **Silu/Swish激活** (40%) - 现代激活函数

### Q4: 3 epochs few-shot评估能预测100 epochs性能吗？
**假设**: Phase 3的3ep排名与100ep排名高度相关 (Kendall's τ > 0.7)

**验证方法**:
- 计算Spearman/Kendall相关系数
- 验证NAS评估器的有效性

---

## 时间表

| 阶段 | 内容 | 时间 | 状态 |
|------|------|------|------|
| 1.1 | 基线实现 | 1天 | ✅ 完成 |
| 1.2 | 发现架构整理 | 0.5天 | ⏳ 当前 |
| 1.3 | 评估框架搭建 | 1天 | 📋 待开始 |
| 2.1 | AI2D完整评估 | 3-4天 | 📋 待开始 |
| 2.2 | 跨数据集验证 | 2-3天 | 📋 待开始 |
| 2.3 | 帕累托分析 | 0.5天 | 📋 待开始 |
| 3.1 | 统计分析 | 0.5天 | 📋 待开始 |
| 3.2 | 报告生成 | 0.5天 | 📋 待开始 |

**总计**: 9-11天

---

## 下一步行动

### 立即执行
1. **整理Phase 3 Top 10架构** → `discovered/` 目录
2. **搭建评估框架** - 统一接口支持任意融合模块
3. **实现全训练pipeline** - 100 epochs训练

### 需要决策
- 是否先在本地用少量epoch测试pipeline？
- 是否直接上服务器跑完整100 epochs？
- 是否需要消融实验 (如arch_024的组件分析)？

---

## 预期成果

1. **论文级对比实验**: NAS vs 人工设计
2. **发现最佳融合架构**: 验证arch_024的SOTA潜力
3. **设计模式洞察**: NAS vs 人类的设计偏好差异
4. **开源贡献**: 26个高质量多模态融合架构

---

*Created: 2026-02-13*
*Status: Phase 1.1 Complete ✅*
*Next: Phase 1.2 - 整理发现架构*
