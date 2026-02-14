# E2: 跨数据集泛化实验

## 实验目标
验证发现架构在不同数据集上的通用性和迁移能力。

## 实验配置
- **数据集**: AI2D, MMMU, VSR, MathVista (4个)
- **架构**: Top 5 NAS架构
- **训练轮数**: 100 epochs
- **运行次数**: 3 runs

## 数据集

| 数据集 | 任务类型 | 样本数 | 难度 |
|--------|----------|--------|------|
| AI2D | 科学图表理解 | 3,000+ | ⭐⭐⭐ |
| MMMU | 多学科推理 | 11,500+ | ⭐⭐⭐⭐⭐ |
| VSR | 空间关系推理 | 10,000+ | ⭐⭐⭐⭐ |
| MathVista | 视觉数学推理 | 6,000+ | ⭐⭐⭐⭐⭐ |

## 评估架构 (Top 5)
1. arch_024 (0.952)
2. arch_019 (0.933)
3. arch_021 (0.933)
4. arch_012 (0.906)
5. arch_025 (0.899)

## 使用方法

```bash
# 评估所有数据集
cd expv2
python E2_cross_dataset/scripts/run_E2.py --dataset all --gpu 0

# 评估单个数据集
python E2_cross_dataset/scripts/run_E2.py --dataset mmmu --gpu 0
python E2_cross_dataset/scripts/run_E2.py --dataset vsr --gpu 0
python E2_cross_dataset/scripts/run_E2.py --dataset mathvista --gpu 0
```

## 预期结果
- NAS架构在跨数据集上保持优势
- 不同架构适合不同任务类型
- 生成任务-架构匹配指南

## 结果目录
```
E2_cross_dataset/results/
├── ai2d/
├── mmmu/
├── vsr/
├── mathvista/
└── summary.json
```
