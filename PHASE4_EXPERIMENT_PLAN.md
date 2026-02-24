# Phase 4: Optimized Architecture Discovery - 实验执行方案

**版本**: 1.0
**创建日期**: 2026-02-21
**目标**: 通过改进评估器、Reward函数和搜索空间，使NAS发现的架构在性能和效率上超越人工设计(FiLM)

---

## 一、实验目标

### 主要目标
1. **性能超越**: NAS架构在MMMU上准确率 > 46% (FiLM目前~46%)
2. **效率超越**: NAS架构FLOPs < 6.29M (FiLM水平)
3. **帕累托优势**: 在准确率-效率权衡上占据前沿

### 当前基线
| 类型 | 代表 | MMMU准确率 | FLOPs |
|------|------|-----------|-------|
| NAS (旧) | arch_017 | ~33% | 13.20M |
| 人工设计 | FiLM | ~46% | 6.29M |

---

## 二、核心改进点

### 改进1: 评估器升级
```
AI2D + 3epochs    →    MMMU + 10epochs
     ↓                        ↓
 太简单，无法区分          难数据集，能区分优劣
```

**新增功能**:
- 早停: 3epoch验证集不提升即停止
- 时间限制: 5分钟上限
- 数据集可配置

### 改进2: Reward函数优化
```python
# 旧配置
weights = {'accuracy': 1.0, 'efficiency': 0.5}
flops_constraint = None

# 新配置
weights = {'accuracy': 1.0, 'efficiency': 1.5}  # 提升效率权重
flops_constraint = 10_000_000  # 10M硬约束
flops_penalty = exponential    # 指数惩罚
```

### 改进3: 搜索空间约束
```python
# 旧配置
num_layers: max 6
hidden_dim: max 1024
use_residual: optional

# 新配置
num_layers: max 4              # 减少深度
hidden_dim: max 512            # 减少宽度
use_residual: required         # 强制残差
fusion_types: +FiLM, +lightweight_attention  # 增加高效选项
```

---

## 三、文件结构

```
phase4_optimization/
├── README.md                   # 本实验说明
├── IMPLEMENTATION_LOG.md       # 实施记录（逐步更新）
├── src/                        # 源代码修改
│   ├── evaluator_v2.py         # 改进版评估器
│   ├── reward_v2.py            # 约束奖励函数
│   └── search_space_v2.py      # 高效搜索空间
├── configs/
│   └── phase4.yaml             # 实验配置
├── results/                    # 实验结果
│   ├── discovery/              # 发现的架构
│   ├── evaluation/             # 评估结果
│   └── analysis/               # 分析报告
├── baselines/                  # SOTA基线实现
│   ├── mcan.py
│   └── ban.py
└── tests/                      # 测试代码
    ├── test_evaluator.py
    ├── test_reward.py
    └── test_integration.py
```

---

## 四、执行步骤清单

### Step 1: 创建实验目录结构
```bash
mkdir -p phase4_optimization/{src,configs,results/{discovery,evaluation,analysis},baselines,tests}
touch phase4_optimization/README.md
```

**验收标准**: 目录结构完整

---

### Step 2: 复制并修改Evaluator
**源文件**: `experiment/evaluators/real_data_evaluator.py`
**目标文件**: `phase4_optimization/src/evaluator_v2.py`

**修改内容**:
1. 添加早停逻辑
2. 添加时间限制
3. 添加配置支持

**验收标准**:
```python
# 测试代码
config = {
    'dataset': 'mmmu',
    'train_epochs': 10,
    'early_stopping': {'patience': 3, 'min_delta': 0.01},
    'max_training_time': 300
}
evaluator = RealDataFewShotEvaluatorV2(config)
# 运行一次评估，验证时间<5分钟且早停生效
```

---

### Step 3: 实现约束奖励函数
**源文件**: `experiment/base/reward.py`
**目标文件**: `phase4_optimization/src/reward_v2.py`

**实现内容**:
```python
class ConstrainedReward(MultiObjectiveReward):
    def __init__(self, config):
        self.max_flops = config['flops_constraint']['max_flops']
        self.efficiency_weight = config['weights']['efficiency']  # 1.5

    def calculate(self, eval_result):
        flops = eval_result['flops']
        # 硬约束检查
        if flops > self.max_flops:
            return RewardComponents(0, 0, 0, 0, rejected=True)
        # 指数惩罚
        penalty = math.exp(-flops / 20e6)
        # 计算奖励
        ...
```

**验收标准**:
```python
# FLOPs=15M应被reject
# FLOPs=5M应有较高奖励
# FLOPs=10M应刚好通过
```

---

### Step 4: 实现高效搜索空间
**源文件**: `experiment/phase3_discovery/run_phase3.py`
**目标文件**: `phase4_optimization/src/search_space_v2.py`

**修改内容**:
1. 添加`get_efficient_space()`函数
2. 限制层数和维度
3. 添加高效fusion类型

**验收标准**:
```python
space = get_efficient_space()
assert space['num_fusion_layers']['high'] == 4
assert space['hidden_dim']['high'] == 512
assert 'film' in space['fusion_type']
```

---

### Step 5: 创建实验配置
**文件**: `phase4_optimization/configs/phase4.yaml`

**配置内容**:
```yaml
experiment_name: "phase4_optimized_discovery"
iterations: 200
population_size: 50

evaluator:
  type: "real_data_v2"
  dataset: "mmmu"
  train_epochs: 10
  early_stopping:
    enabled: true
    patience: 3
    min_delta: 0.01
  max_training_time: 300

reward:
  type: "constrained"
  weights:
    accuracy: 1.0
    efficiency: 1.5
  flops_constraint:
    enabled: true
    max_flops: 10000000
    reject_if_exceed: true
  flops_penalty:
    type: "exponential"
    scale: 20000000

search_space:
  type: "efficient"
  # 详细配置见yaml文件
```

---

### Step 6: 小规模测试 (Checkpoint 1)
**测试配置**:
```yaml
iterations: 10
train_epochs: 5
```

**验收标准**:
- [ ] 单次评估时间 < 5分钟
- [ ] 平均FLOPs < 15M
- [ ] 无运行时错误

**如果失败**:
- 时间过长 → 改为VSR或MMMU+5ep
- FLOPs过高 → 加强约束

---

### Step 7: 完整架构搜索 (Checkpoint 2)
**配置**: `configs/phase4.yaml` (完整版)

**运行**:
```bash
python run.py --experiment phase3 --config phase4_optimization/configs/phase4.yaml --gpu 0
```

**验收标准**:
- [ ] 完成200 iterations
- [ ] 至少20个架构reward > 0.75
- [ ] 平均FLOPs < 10M

---

### Step 8: SOTA基线实现 (可并行)
**文件**: `phase4_optimization/baselines/mcan.py`, `ban.py`

**实现要求**:
1. 统一接口: `FusionModule(vision_dim, language_dim, hidden_dim)`
2. 计算FLOPs和参数量
3. 在AI2D和MMMU上可运行

**验收标准**:
```python
mcan = MCAN(768, 768, 512)
# 能forward通过
# 能计算FLOPs
```

---

### Step 9: 全面评估 (Checkpoint 3)
**评估对象**:
- Top 5 新NAS架构
- 5个旧Baseline
- 2-3个SOTA基线

**运行**:
```bash
python run.py --experiment E1 --mode full --gpu 0
python run.py --experiment E2 --dataset mmmu --gpu 0
```

**验收标准**:
- [ ] 至少1个新NAS > 46% on MMMU
- [ ] 相同架构 < 6.29M FLOPs
- [ ] 统计显著性 p < 0.05

---

### Step 10: 分析报告
**文件**: `phase4_optimization/results/analysis/report.md`

**内容**:
1. 对比表格 (准确率、FLOPs、参数量)
2. 帕累托前沿图
3. 统计显著性检验
4. 消融实验结果

---

## 五、团队协作分工

| 角色 | 负责内容 | 产出 |
|------|---------|------|
| **你** | Step 1-7, 9-10 | 核心代码、实验运行、最终报告 |
| **同学A** | Step 8: MCAN实现 | `baselines/mcan.py` |
| **同学B** | Step 8: BAN实现 | `baselines/ban.py` |
| **同学C** | SOTA调研 | `docs/sota_survey.md` |
| **同学D** | 可视化 | 图表、分析报告初稿 |

**并行策略**:
- 你执行Step 1-5时，同学C开始SOTA调研
- 你执行Step 6-7时，同学A/B开始基线实现
- 你执行Step 9时，同学D开始准备可视化

---

## 六、风险应对

| 风险 | 检查点 | 应对策略 |
|------|--------|----------|
| MMMU太慢 | CP1 | 改用VSR或MMMU+5ep |
| FLOPs约束太严 | CP2 | 放宽到15M |
| 无法超越FiLM | CP3 | 增加iterations到300，或调整reward权重 |
| SOTA实现困难 | CP3 | 简化实现，保留核心机制 |

---

## 七、验收标准

### 必须达成 (P0)
- [ ] 至少1个NAS架构 MMMU准确率 > 46%
- [ ] 同一架构 FLOPs < 6.29M

### 期望达成 (P1)
- [ ] 3个以上NAS架构超越FiLM
- [ ] 在VSR/MathVista上同样有效
- [ ] 发现可解释的设计模式

### 加分项 (P2)
- [ ] 统计显著性 p < 0.01
- [ ] FLOPs < 5M (超越CLIPFusion)
- [ ] 发现新的高效融合机制

---

## 八、参考资源

### 关键论文 (SOTA调研)
- MCAN: "Deep Modular Co-Attention Networks" (CVPR 2019)
- BAN: "Bilinear Attention Networks" (NeurIPS 2018)
- FiLM: "FiLM: Visual Reasoning with a General Conditioning Layer"

### 相关文件
- `EXPERIMENTS_SUMMARY.md` - 前期实验结果
- `OPTIMIZATION_PLAN_V2.md` - 优化方案概述
- `UNIFIED_FRAMEWORK.md` - 统一框架说明

---

## 九、实施记录

### Step 1: 创建目录结构
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 2: 修改Evaluator
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 3: 实现约束Reward
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 4: 实现高效搜索空间
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 5: 创建配置
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 6: 小规模测试
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 7: 完整搜索
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 8: SOTA实现
**状态**: ⬜ 待开始
**负责人**: 同学A/B
**日期**: ___
**备注**: ___

### Step 9: 全面评估
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

### Step 10: 分析报告
**状态**: ⬜ 待开始
**日期**: ___
**备注**: ___

---

*创建: 2026-02-21*
*更新: (待填写)*
