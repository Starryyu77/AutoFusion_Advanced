# Phase 2.5: Evaluator Verification - 实验进度记录

## 实验目标
验证 RealDataFewShotEvaluator 的设计，找到最佳的 (数据集, 训练深度, 架构) 组合。

---

## Phase 2.5.1: Dataset Selection (数据集选择) ✅ 已完成

**执行时间**: 2026-02-11
**执行位置**: ntu-gpu43 (CUDA 3)
**完整报告**: [PHASE_2_5_1_REPORT.md](../../docs/experiments/PHASE_2_5_1_REPORT.md)

### 测试配置
- 数据集: MMMU, VSR, MathVista, AI2D
- 训练深度: 5 epochs
- 样本数: 16 shots
- 测试架构: 4个 (attention_simple, conv_fusion, transformer_fusion, mlp_simple)

### 总体结果

| 数据集 | Mean Accuracy | Std | 排名 | 备注 |
|--------|---------------|-----|------|------|
| **AI2D** | **0.2500** | 0.1369 | 1 🏆 | **选定数据集** |
| MathVista | 0.1625 | 0.0820 | 2 | 表现中等 |
| MMMU | 0.1125 | 0.0960 | 3 | 综合性过强 |
| VSR | 0.0000 | 0.0000 | 4 | 标签问题已修复 |

### 详细结果 (各架构表现)

| 架构 | AI2D | MathVista | MMMU | VSR |
|------|------|-----------|------|-----|
| attention_simple | 0.05 | 0.15 | 0.05 | 0.00 |
| conv_fusion | 0.35 | 0.30 | 0.25 | 0.00 |
| transformer_fusion | 0.40 | 0.10 | 0.00 | 0.00 |
| mlp_simple | 0.20 | 0.10 | 0.15 | 0.00 |

### 结论
**AI2D** 被选为后续实验的标准数据集：
- 最高平均准确率 (0.25)
- 良好的架构区分度 (std=0.137)
- transformer_fusion 在该数据集上表现最佳 (0.40)

### 关键修复
1. Python 3.8 兼容性 (Tuple vs tuple)
2. 自定义 collate function 处理 None 值
3. VSR 标签转换 (True/False → int)
4. MMMU 配置（加载5个学科）

---

## Phase 2.5.2: Training Depth Calibration (训练深度校准) ✅ 已完成

**执行时间**: 2026-02-11
**执行位置**: ntu-gpu43 (GPU 2)
**完整报告**: [PHASE_2_5_2_REPORT.md](../../docs/experiments/PHASE_2_5_2_REPORT.md)

### 测试配置
- 数据集: AI2D (由 2.5.1 选定)
- 训练深度: [1, 3, 5, 10] epochs
- 样本数: 16 shots
- 测试架构: 3个 (attention_simple, conv_fusion, transformer_fusion)

### 总体结果

| Epochs | Mean Accuracy | Std | Time (s) | 排名 |
|--------|---------------|-----|----------|------|
| **3** | **0.2500** | 0.0816 | 2.7 | 🥇 **推荐** |
| 10 | 0.2167 | 0.0471 | 3.2 | 2 |
| 1 | 0.1500 | 0.0816 | 4.2 | 3 |
| 5 | 0.1333 | 0.1247 | 2.9 | 4 |

### 各架构详细表现

| 架构 | 1 epoch | 3 epochs | 5 epochs | 10 epochs |
|------|---------|----------|----------|-----------|
| attention_simple | 0.05 | 0.15 | 0.30 | 0.25 |
| conv_fusion | 0.15 | 0.35 | 0.10 | 0.25 |
| transformer_fusion | 0.25 | 0.25 | 0.00 | 0.15 |

### 结论
**推荐 3 epochs** 作为标准训练深度：
- 最高平均准确率 (0.25)
- 最短时间 (2.7s / eval)
- 良好的稳定性 (std=0.082)

### 关键发现
- **非单调性**: 5 epochs 表现反而下降，可能是过拟合
- **3 epochs 最优**: 性价比最高（准确率 × 时间）
- **架构差异**: conv_fusion 在 3 epochs 达到峰值 (0.35)

---

## Phase 2.5.3: Architecture Fairness (架构公平性) ✅ 已完成

**执行时间**: 2026-02-11
**执行位置**: ntu-gpu43 (GPU 2)
**完整报告**: [PHASE_2_5_3_REPORT.md](../../docs/experiments/PHASE_2_5_3_REPORT.md)

### 测试配置
- 数据集: AI2D (由 2.5.1 确定)
- 训练深度: 3 epochs (由 2.5.2 确定)
- 样本数: 16 shots
- 测试架构: 5 类型 × 2 变体 = 10 个架构
- Seeds: [42, 123, 456]

### 公平性指标

| 指标 | 值 | 评级 |
|------|-----|------|
| 整体平均准确率 | 0.2483 | - |
| 整体标准差 | **0.0561** | **EXCELLENT** ✅ |
| 类型间最大差异 | 0.1333 | GOOD ✅ |
| **公平性评级** | **EXCELLENT** | ✅ |

### 各架构类型表现

| 架构类型 | 平均准确率 | 类型内标准差 | 排名 |
|----------|-----------|-------------|------|
| mlp_based | 0.3167 | 0.0333 | 🥇 |
| transformer_based | 0.3083 | 0.0083 | 🥈 |
| conv_based | 0.2417 | 0.0083 | 🥉 |
| attention_based | 0.1917 | 0.0250 | 4 |
| hybrid | 0.1833 | 0.0833 | 5 |

### 结论
**评估器公平性 EXCELLENT** (std=0.056 < 0.1)：
- 所有架构类型获得合理评分
- 无系统性偏见
- 可以公平比较不同类型架构

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| run_2_5_1_dataset_selection.py | ✅ | 数据集选择实验 |
| run_2_5_2_training_depth.py | ✅ | 训练深度校准实验（已修复） |
| run_2_5_3_architecture_fairness.py | ✅ | 架构公平性实验（已完善） |
| results/2_5_1_dataset_selection/ | ✅ | 2.5.1 结果 (已同步) |
| results/2_5_2_training_depth/ | ✅ | 2.5.2 结果 (已同步) |
| results/2_5_3_architecture_fairness/ | ✅ | 2.5.3 结果 (已同步) |
| docs/experiments/PHASE_2_5_1_REPORT.md | ✅ | 数据集选择实验报告 |
| docs/experiments/PHASE_2_5_2_REPORT.md | ✅ | 训练深度校准实验报告 |
| docs/experiments/PHASE_2_5_3_REPORT.md | ✅ | 架构公平性实验报告 |

---

## 已知问题与修复记录

### Issue 1: VSR 标签类型错误
**问题**: VSR 数据集使用布尔值 True/False 作为标签，导致训练失败
**修复**: 在 real_data_evaluator.py 中添加 convert_label() 函数处理多种标签类型

### Issue 2: DataLoader None 值错误
**问题**: PyTorch 默认 collate 函数无法处理 batch 中的 None 值
**修复**: 在 dataset_loader.py 中实现 custom_collate_fn()

### Issue 3: Phase 2.5.2 脚本不完整
**问题**: run_2_5_2_training_depth.py 中架构代码被替换为 '...'
**修复**: 已恢复完整的 FusionModule 定义

---

## Phase 2.5.4: Final Configuration (最终配置) ✅

**状态**: 已完成 - 基于 2.5.1/2.5.2/2.5.3 结果确定

### 推荐配置

```python
recommended_config = {
    'dataset': 'ai2d',        # From Phase 2.5.1 ✅
    'train_epochs': 3,        # From Phase 2.5.2 ✅
    'num_shots': 16,
    'batch_size': 4,
    'backbone': 'clip-vit-l-14',
    'fairness_rating': 'EXCELLENT',  # From Phase 2.5.3 ✅
}
```

### 配置验证总结

| 验证项 | 结果 | 状态 |
|--------|------|------|
| 数据集选择 (AI2D) | 最高准确率 0.25 | ✅ |
| 训练深度 (3 epochs) | 最优性价比 | ✅ |
| 架构公平性 | EXCELLENT (std=0.056) | ✅ |

---

## 下一步行动

- [x] Phase 2.5.1: 数据集选择 (AI2D) ✅
- [x] Phase 2.5.2: 训练深度校准 (3 epochs) ✅
- [x] Phase 2.5.3: 架构公平性验证 (EXCELLENT) ✅
- [x] Phase 2.5.4: 最终配置确定 ✅
- [ ] 应用配置到 Phase 1 (Prompt 对比)
- [ ] 应用配置到 Phase 3 (架构发现)
