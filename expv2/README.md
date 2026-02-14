# ExpV2: NAS vs Human Design - 全面对比实验

**目标**: 证明自动化NAS可以找到比人工设计更好的多模态融合层

**价值主张**: BETTER + FASTER + MORE

---

## 实验分类

所有实验按E1-E7分类组织：

```
expv2/
├── E1_main_evaluation/        # P0: AI2D主实验
├── E2_cross_dataset/          # P0: 跨数据集泛化
├── E3_pareto_analysis/        # P0: 帕累托前沿分析
├── E4_correlation/            # P1: 3ep vs 100ep相关性
├── E5_ablation/               # P1: 消融实验
├── E6_design_patterns/        # P1: 设计模式分析
├── E7_statistical_test/       # P0: 统计显著性检验
└── shared/                    # 共享组件
    ├── baselines/             # 5个人工设计基线
    ├── discovered/            # 10个NAS发现架构
    └── evaluation/            # 评估框架
```

---

## 核心实验 (P0)

### E1: AI2D主实验 ⭐ 最优先
**目标**: 验证NAS vs 人工设计的性能差距

```bash
# 快速测试 (本地)
python E1_main_evaluation/scripts/run_E1.py --mode quick --gpu 0

# 完整评估 (服务器)
bash E1_main_evaluation/scripts/run_on_server.sh 2
```

**架构**: 10 NAS + 5 Baseline
**配置**: 100 epochs, 3 runs
**结果**: `E1_main_evaluation/results/`

### E2: 跨数据集泛化
**目标**: 验证架构通用性

```bash
python E2_cross_dataset/scripts/run_E2.py --dataset all --gpu 0
```

**数据集**: AI2D, MMMU, VSR, MathVista
**架构**: Top 5 NAS

### E3: 帕累托分析
**目标**: 展示NAS多样性优势

```bash
python E3_pareto_analysis/scripts/run_E3.py \
    --input-dir E1_main_evaluation/results/full_3runs
```

### E7: 统计检验
**目标**: 确保结果可信度

---

## 共享组件

### 基线架构 (人工设计)
```python
from shared.baselines import ConcatMLP, BilinearPooling, CrossModalAttention, CLIPFusion, FiLM
```

### 发现架构 (NAS生成)
```python
from shared.discovered import DISCOVERED_ARCHITECTURES
arch = DISCOVERED_ARCHITECTURES['arch_024']()
```

### 评估接口
```python
from shared.evaluation import FullEvaluator

evaluator = FullEvaluator(dataset='ai2d')
results = evaluator.evaluate(arch, 'arch_024')
```

---

## 论文故事线

> **"Can automated NAS discover multimodal fusion layers that are BETTER, FASTER, and MORE diverse than human-designed architectures?"**

### 核心贡献
| 贡献 | 内容 | 实验 |
|------|------|------|
| **C1 - BETTER** | NAS架构性能超越人工设计 | E1, E2 |
| **C2 - FASTER** | 31.5分钟 vs 数小时 | E1分析 |
| **C3 - MORE** | 一次搜索26个架构 | E3 |
| **C4 - INSIGHTS** | 设计偏好差异 | E6 |

---

## 快速开始

### 1. 本地测试
```bash
cd expv2
python E1_main_evaluation/scripts/run_E1.py --mode quick --gpu 0
```

### 2. 服务器完整实验
```bash
bash E1_main_evaluation/scripts/run_on_server.sh 2
```

### 3. 分析结果
```bash
python E3_pareto_analysis/scripts/run_E3.py
```

---

## 实验状态

| 实验 | 状态 | 优先级 |
|------|------|--------|
| E1 | ⏳ 待运行 | P0 |
| E2 | 📋 待开始 | P0 |
| E3 | 📋 待开始 | P0 |
| E4 | 📋 待开始 | P1 |
| E5 | 📋 待开始 | P1 |
| E6 | 📋 待开始 | P1 |
| E7 | 📋 待开始 | P0 |

---

**状态**: 文件结构重组完成 ✅
**下一步**: 运行E1快速测试
