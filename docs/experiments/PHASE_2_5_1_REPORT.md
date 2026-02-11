# Phase 2.5.1: Dataset Selection Experiment Report

## 实验概述

### 实验目标
验证并选择最适合用于评估器验证的数据集。通过对比 MMMU、VSR、MathVista 和 AI2D 四个数据集在相同实验条件下的表现，确定最能反映架构性能差异的标准数据集。

### 实验背景
Phase 2.5 评估器验证的核心目标是找到一种快速、可靠且经济的评估方法。数据集选择是关键的第一步——一个好的验证数据集应该：
1. 能够区分不同质量的架构
2. 在 few-shot 设置下仍有稳定的信号
3. 与最终任务性能有较高的相关性

### 实验时间
- **执行日期**: 2026-02-11
- **执行服务器**: ntu-gpu43 (CUDA 3)
- **总执行时间**: ~45 分钟

---

## 实验设计

### 测试数据集

| 数据集 | 全称 | 任务类型 | 样本数 | 特点 |
|--------|------|----------|--------|------|
| MMMU | Massive Multi-discipline Multimodal Understanding | 多学科问答 | 16 shots | 涵盖6大学科，综合性强 |
| VSR | Visual Spatial Reasoning | 空间推理 | 16 shots | 判断空间关系True/False |
| AI2D | AI2 Diagrams | 科学图表理解 | 16 shots | 图表结构解析 |
| MathVista | Mathematical Visual Reasoning | 数学视觉推理 | 16 shots | 结合数学与视觉 |

### 实验配置

```python
config = {
    'num_shots': 16,          # Few-shot 样本数
    'train_epochs': 5,        # 训练深度
    'batch_size': 4,          # 批大小
    'backbone': 'clip-vit-l-14',  # 预训练骨干网络
}
```

### 测试架构

使用4种不同类型的融合架构进行交叉验证：

| 架构名称 | 类型 | 描述 |
|----------|------|------|
| attention_simple | Attention | 多头注意力融合 |
| conv_fusion | Convolution | 1D卷积融合 |
| transformer_fusion | Transformer | Transformer编码器融合 |
| mlp_simple | MLP | 多层感知机融合 |

---

## 实验结果

### 总体表现

| 数据集 | 平均准确率 | 标准差 | 排名 | 备注 |
|--------|-----------|--------|------|------|
| **AI2D** | **0.2500** | 0.1369 | 🥇 1 | **选定数据集** |
| MathVista | 0.1625 | 0.0820 | 🥈 2 | 表现中等 |
| MMMU | 0.1125 | 0.0960 | 🥉 3 | 表现较弱 |
| VSR | 0.0000 | 0.0000 | 4 | 需要标签处理 |

### 详细结果

#### AI2D (选定数据集)
```json
{
  "scores": {
    "attention_simple": 0.05,
    "conv_fusion": 0.35,
    "transformer_fusion": 0.40,
    "mlp_simple": 0.20
  },
  "mean": 0.25,
  "std": 0.137
}
```

**分析**: AI2D 表现最佳，平均准确率达到 0.25。transformer_fusion 在该数据集上表现最好 (0.40)，显示出架构间有明显的区分度。

#### MathVista
```json
{
  "scores": {
    "attention_simple": 0.15,
    "conv_fusion": 0.30,
    "transformer_fusion": 0.10,
    "mlp_simple": 0.10
  },
  "mean": 0.1625,
  "std": 0.082
}
```

**分析**: 表现中等，conv_fusion 表现较好 (0.30)。但整体准确率偏低。

#### MMMU
```json
{
  "scores": {
    "attention_simple": 0.05,
    "conv_fusion": 0.25,
    "transformer_fusion": 0.0,
    "mlp_simple": 0.15
  },
  "mean": 0.1125,
  "std": 0.096
}
```

**分析**: 综合性数据集表现较弱，可能是因为学科多样性导致 16-shot 样本不足以覆盖所有领域。

#### VSR
```json
{
  "scores": {
    "attention_simple": 0.0,
    "conv_fusion": 0.0,
    "transformer_fusion": 0.0,
    "mlp_simple": 0.0
  },
  "mean": 0.0,
  "std": 0.0
}
}
```

**分析**: 所有架构准确率均为0，原因是 VSR 使用布尔值 True/False 作为标签，而评估器需要整数标签。此问题已在后续修复。

### 各架构跨数据集表现

| 架构 | AI2D | MathVista | MMMU | VSR | 平均 |
|------|------|-----------|------|-----|------|
| attention_simple | 0.05 | 0.15 | 0.05 | 0.00 | 0.063 |
| conv_fusion | 0.35 | 0.30 | 0.25 | 0.00 | 0.225 |
| transformer_fusion | 0.40 | 0.10 | 0.00 | 0.00 | 0.125 |
| mlp_simple | 0.20 | 0.10 | 0.15 | 0.00 | 0.113 |

**观察**: conv_fusion 在多数数据集上表现稳定且较好。

---

## 关键修复记录

在实验过程中发现并修复了以下关键问题：

### Fix 1: Python 3.8 兼容性
**问题**: 使用了 Python 3.9+ 的语法 `tuple[str, ...]`
**修复**: 改为 `from typing import Tuple` 并使用 `Tuple[str, ...]`
**文件**: `utils/llm_client.py`

### Fix 2: DataLoader None 值处理
**问题**: PyTorch 默认 collate 函数无法处理 batch 中的 None 值
**修复**: 在 `dataset_loader.py` 中实现 `custom_collate_fn()`
```python
def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
```

### Fix 3: VSR 标签类型转换
**问题**: VSR 使用布尔值 True/False 作为标签
**修复**: 在 `real_data_evaluator.py` 中添加 `convert_label()` 函数
```python
def convert_label(l):
    if isinstance(l, bool):
        return 1 if l else 0
    elif isinstance(l, (int, float)):
        return int(l)
    # ...
```

### Fix 4: MMMU 配置加载
**问题**: MMMU 需要加载多个学科子集
**修复**: 配置加载 5 个主要学科（Art, Business, Science, Health, Tech）

---

## 结论与建议

### 主要结论

1. **AI2D 被选为后续实验的标准数据集**
   - 最高平均准确率 (0.25)
   - 良好的架构区分度 (std=0.137)
   - 科学图表理解与融合架构任务相关性高

2. **VSR 需要额外处理**
   - 布尔标签已修复，但准确率仍需验证
   - 空间推理任务可能需要更多 shots 或不同配置

3. **MMMU 综合性过强**
   - 16-shot 不足以覆盖多学科的多样性
   - 建议增加 shots 数量或使用子集

### 后续行动

- [x] Phase 2.5.1: 数据集选择 ✅
- [ ] Phase 2.5.2: 训练深度校准（使用 AI2D）
- [ ] Phase 2.5.3: 架构公平性验证
- [ ] Phase 2.5.4: 最终配置确定

---

## 附录

### 实验脚本

**主脚本**: `experiment/phase2_5/run_2_5_1_dataset_selection.py`

核心实验流程：
```python
def run_experiment(datasets=['mmmu', 'vsr', 'ai2d', 'mathvista'],
                   num_shots=16,
                   train_epochs=5):
    for dataset_name in datasets:
        config = {
            'dataset': dataset_name,
            'num_shots': num_shots,
            'train_epochs': train_epochs,
            'batch_size': 4,
            'backbone': 'clip-vit-l-14',
        }
        evaluator = RealDataFewShotEvaluator(config)

        for arch_name, arch_code in TEST_ARCHITECTURES.items():
            result = evaluator.evaluate(arch_code)
            scores[arch_name] = result.accuracy
```

### 原始数据文件

- **本地路径**: `experiment/phase2_5/results/2_5_1_dataset_selection/results.json`
- **服务器路径**: `ntu-gpu43:/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase2_5/results/2_5_1_dataset_selection/`

### 相关文档

- [EVALUATOR_V2_DESIGN.md](../design/EVALUATOR_V2_DESIGN.md) - RealDataFewShotEvaluator 架构设计
- [EVALUATOR_VERIFICATION_DETAILS.md](../design/EVALUATOR_VERIFICATION_DETAILS.md) - 验证标准详解
- [EXPERIMENT_PLAN_V4.md](EXPERIMENT_PLAN_V4.md) - 完整实验计划

---

*报告生成时间: 2026-02-11*
*实验执行: ntu-gpu43*
*作者: AutoFusion Team*
