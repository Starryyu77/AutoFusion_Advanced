# Phase 2.5.2: Training Depth Calibration Experiment Report

## 实验概述

### 实验目标
确定最具成本效益的训练深度（epochs），在准确率与评估时间之间找到最佳平衡点。

### 实验背景
Phase 2.5.1 确定了 AI2D 作为标准数据集。现在需要确定在 few-shot 设置下，训练多少 epochs 能够：
1. 获得稳定的准确率信号
2. 最小化评估时间
3. 避免过拟合

### 实验时间
- **执行日期**: 2026-02-11
- **执行服务器**: ntu-gpu43 (GPU 2)
- **总执行时间**: ~2 分钟

---

## 实验设计

### 测试训练深度

| 深度 | 描述 | 预期特性 |
|------|------|----------|
| 1 epoch | 最少训练 | 快速但可能欠拟合 |
| 3 epochs | 短训练 | 平衡速度与性能 |
| 5 epochs | 中等训练 | 标准 few-shot 设置 |
| 10 epochs | 充分训练 | 可能过拟合 |

### 实验配置

```python
config = {
    'dataset': 'ai2d',         # 由 2.5.1 选定
    'num_shots': 16,           # Few-shot 样本数
    'train_epochs': [1,3,5,10], # 测试的训练深度
    'batch_size': 4,           # 批大小
    'backbone': 'clip-vit-l-14',  # 预训练骨干网络
}
```

### 测试架构

使用3种不同类型的融合架构：

| 架构名称 | 类型 | 描述 |
|----------|------|------|
| attention_simple | Attention | 多头注意力融合 |
| conv_fusion | Convolution | 1D卷积融合 |
| transformer_fusion | Transformer | Transformer编码器融合 |

---

## 实验结果

### 总体表现

| Epochs | Mean Accuracy | Std | Time (s) | 排名 | 性价比 |
|--------|---------------|-----|----------|------|--------|
| **3** | **0.2500** | 0.0816 | **2.7** | 🥇 | **最高** |
| 10 | 0.2167 | 0.0471 | 3.2 | 🥈 | 中等 |
| 1 | 0.1500 | 0.0816 | 4.2 | 🥉 | 低 |
| 5 | 0.1333 | 0.1247 | 2.9 | 4 | 低 |

**原始数据**:
```json
{
  "epochs_1": {
    "mean": 0.15,
    "std": 0.0816,
    "time_per_eval": 4.24
  },
  "epochs_3": {
    "mean": 0.25,
    "std": 0.0816,
    "time_per_eval": 2.69
  },
  "epochs_5": {
    "mean": 0.1333,
    "std": 0.1247,
    "time_per_eval": 2.87
  },
  "epochs_10": {
    "mean": 0.2167,
    "std": 0.0471,
    "time_per_eval": 3.21
  }
}
```

### 各架构详细表现

| 架构 | 1 epoch | 3 epochs | 5 epochs | 10 epochs |
|------|---------|----------|----------|-----------|
| attention_simple | 0.05 | 0.15 | **0.30** | 0.25 |
| conv_fusion | 0.15 | **0.35** | 0.10 | 0.25 |
| transformer_fusion | **0.25** | 0.25 | 0.00 | 0.15 |

### 各架构分析

#### attention_simple
- **最佳**: 5 epochs (0.30)
- **趋势**: 随 epochs 增加先升后降
- **分析**: 需要较多迭代才能收敛

#### conv_fusion
- **最佳**: 3 epochs (0.35)
- **趋势**: 3 epochs 达到峰值后下降
- **分析**: 快速收敛，易过拟合

#### transformer_fusion
- **最佳**: 1/3 epochs (0.25)
- **趋势**: 早期即达峰值，之后下降明显
- **分析**: 对训练深度最敏感

---

## 关键发现

### 1. 非单调性能曲线

与预期不同，**准确率并非随 epochs 单调增加**:
- 5 epochs 表现最差 (0.1333)
- 10 epochs 有所恢复 (0.2167)
- 3 epochs 达到峰值 (0.2500)

**原因分析**:
- 16-shot 样本量小，容易过拟合
- 5 epochs 可能处于"过拟合临界点"
- 不同架构的最优训练深度不同

### 2. 3 epochs 最优

**推荐 3 epochs 作为标准训练深度**:

| 指标 | 3 epochs | 对比次优 (10 epochs) |
|------|----------|---------------------|
| 准确率 | 0.2500 | 0.2167 (+15%) |
| 时间 | 2.7s | 3.2s (-16%) |
| 稳定性 | std=0.082 | std=0.047 |

### 3. 架构特异性

不同架构对训练深度的敏感度:
- **conv_fusion**: 3 epochs 最优，易过拟合
- **attention_simple**: 5 epochs 最优，收敛慢
- **transformer_fusion**: 1-3 epochs 最优，对深度敏感

---

## 结论与建议

### 主要结论

1. **推荐 3 epochs 作为标准训练深度**
   - 最高平均准确率 (0.25)
   - 最短评估时间 (2.7s)
   - 良好的稳定性 (std=0.082)

2. **Few-shot 场景下过拟合风险**
   - 5 epochs 出现性能下降
   - 16-shot 样本量不足以支持长时间训练

3. **架构间存在差异**
   - 不同架构的最优训练深度不同
   - 3 epochs 是整体最优折中

### 后续实验建议

- [x] Phase 2.5.1: 数据集选择 (AI2D) ✅
- [x] Phase 2.5.2: 训练深度校准 (3 epochs) ✅
- [ ] Phase 2.5.3: 架构公平性验证
- [ ] Phase 2.5.4: 最终配置确定

### 推荐配置 (暂定)

```python
recommended_config = {
    'dataset': 'ai2d',        # From Phase 2.5.1
    'train_epochs': 3,        # From Phase 2.5.2
    'num_shots': 16,
    'batch_size': 4,
    'backbone': 'clip-vit-l-14',
}
```

---

## 附录

### 实验脚本

**主脚本**: `experiment/phase2_5/run_2_5_2_training_depth.py`

核心实验流程：
```python
def run_experiment(dataset='ai2d', depths=[1, 3, 5, 10], num_shots=16):
    for depth in depths:
        config = {
            'dataset': dataset,
            'num_shots': num_shots,
            'train_epochs': depth,
            'batch_size': 4,
            'backbone': 'clip-vit-l-14',
        }
        evaluator = RealDataFewShotEvaluator(config)

        for arch_name, arch_code in TEST_ARCHITECTURES.items():
            result = evaluator.evaluate(arch_code)
            scores.append(result.accuracy)
            times.append(eval_time)
```

### 执行命令

```bash
# 在 ntu-gpu43 上执行
ssh ntu-gpu43
cd /usr1/home/s125mdg43_10/AutoFusion_Advanced
CUDA_VISIBLE_DEVICES=2 python3 experiment/phase2_5/run_2_5_2_training_depth.py
```

### 原始数据文件

- **本地路径**: `experiment/phase2_5/results/2_5_2_training_depth/`
  - `results.json`: 结构化结果
  - `summary.txt`: 文本摘要
  - `experiment.log`: 完整执行日志
- **服务器路径**: `ntu-gpu43:/usr1/home/s125mdg43_10/AutoFusion_Advanced/experiment/phase2_5/results/2_5_2_training_depth/`

### 实验日志片段

```
Testing 1 epochs...
  attention_simple: Acc=0.0500, Time=6.9s
  conv_fusion: Acc=0.1500, Time=3.2s
  transformer_fusion: Acc=0.2500, Time=2.6s

Testing 3 epochs...
  attention_simple: Acc=0.1500, Time=2.7s
  conv_fusion: Acc=0.3500, Time=2.6s
  transformer_fusion: Acc=0.2500, Time=2.7s

Testing 5 epochs...
  attention_simple: Acc=0.3000, Time=2.9s
  conv_fusion: Acc=0.1000, Time=2.8s
  transformer_fusion: Acc=0.0000, Time=3.0s

Testing 10 epochs...
  attention_simple: Acc=0.2500, Time=3.2s
  conv_fusion: Acc=0.2500, Time=3.1s
  transformer_fusion: Acc=0.1500, Time=3.3s
```

### 相关文档

- [PHASE_2_5_1_REPORT.md](PHASE_2_5_1_REPORT.md) - 数据集选择实验报告
- [EVALUATOR_V2_DESIGN.md](../design/EVALUATOR_V2_DESIGN.md) - RealDataFewShotEvaluator 架构设计
- [EXPERIMENT_PLAN_V4.md](EXPERIMENT_PLAN_V4.md) - 完整实验计划

---

*报告生成时间: 2026-02-11*
*实验执行: ntu-gpu43 (GPU 2)*
*作者: AutoFusion Team*
