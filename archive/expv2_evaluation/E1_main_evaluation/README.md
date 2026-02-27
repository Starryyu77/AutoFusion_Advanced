# E1: AI2D主实验 - NAS vs Human Design

## 实验目标
验证NAS发现的架构在主要数据集上是否优于传统人工设计基线。

## 实验配置
- **数据集**: AI2D (4类科学图表理解)
- **训练轮数**: 100 epochs
- **运行次数**: 3 runs (不同随机种子)
- **架构**: 10 NAS + 5 Baseline = 15个架构

## 评估架构

### 基线 (人工设计)
| 架构 | 类型 | 设计原理 |
|------|------|----------|
| ConcatMLP | MLP | 拼接+MLP (最常用基线) |
| BilinearPooling | Bilinear | 双线性池化 |
| CrossModalAttention | Attention | 跨模态注意力 (ViLBERT) |
| CLIPFusion | Linear | CLIP风格投影 |
| FiLM | Modulation | 特征调制 |

### 发现架构 (NAS生成)
| 架构 | 3ep Reward | 设计类型 |
|------|------------|----------|
| arch_024 | 0.952 | Hybrid (Bilinear+Transformer) |
| arch_019 | 0.933 | Attention+MLP |
| arch_021 | 0.933 | Pure MLP |
| arch_012 | 0.906 | Transformer |
| arch_025 | 0.899 | Hybrid Attention |
| arch_004 | 0.873 | MLP+Attention |
| arch_022 | 0.873 | Pure MLP |
| arch_015 | 0.850 | Gated Attention |
| arch_008 | 0.825 | Bilinear |
| arch_017 | 0.819 | Attention+MLP |

## 使用方法

### 本地快速测试 (10 epochs)
```bash
# 快速测试所有架构
cd expv2
python E1_main_evaluation/scripts/run_E1.py --mode quick --gpu 0

# 只测试基线
python E1_main_evaluation/scripts/run_E1.py --mode quick --arch-type baseline --gpu 0

# 只测试发现架构
python E1_main_evaluation/scripts/run_E1.py --mode quick --arch-type discovered --gpu 0
```

### 服务器完整评估 (100 epochs, 3 runs)
```bash
# 在服务器上运行
bash E1_main_evaluation/scripts/run_on_server.sh 2  # GPU 2
```

## 评估指标
- **性能**: Accuracy, Best Val Accuracy
- **效率**: FLOPs, Parameters, Latency
- **训练**: Convergence Speed, Training Time

## 预期结果
- NAS Top-3平均准确率 > 人工基线平均准确率 (p<0.05)
- arch_024达到SOTA或接近SOTA水平

## 结果目录
```
E1_main_evaluation/results/
├── quick_test/           # 快速测试结果
│   ├── summary.json
│   └── *_result.json
└── full_3runs/           # 完整评估结果
    ├── summary.json
    └── {arch_name}/
        ├── evaluation_results.json
        └── best_model.pt
```

## 状态
- [ ] 快速测试
- [ ] 基线完整评估
- [ ] 发现架构完整评估
- [ ] 结果分析
