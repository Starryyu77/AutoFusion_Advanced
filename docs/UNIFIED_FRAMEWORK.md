# AutoFusion 统一实验框架

**版本**: 2.0
**目标**: 提供统一的实验入口，简化实验流程

---

## 快速开始

### 1. 查看实验状态

```bash
python run.py --status
```

### 2. 运行实验

#### Phase 1: Prompt策略对比
```bash
python run.py --experiment phase1 --config configs/phase1.yaml --gpu 0
```

#### Phase 3: 架构发现 (原始版本)
```bash
python run.py --experiment phase3 --config configs/phase3.yaml --gpu 0
```

#### Phase 4: 优化版架构发现 ⭐
```bash
python run.py --experiment phase3 --config configs/phase4_optimized.yaml --gpu 0
```

#### E1: 完整评估
```bash
# 快速测试 (10 epochs)
python run.py --experiment E1 --mode quick --gpu 0

# 完整评估 (100 epochs, 3 runs)
python run.py --experiment E1 --mode full --gpu 0
```

#### E2: 跨数据集评估
```bash
# 所有数据集
python run.py --experiment E2 --dataset all --gpu 0

# 单个数据集
python run.py --experiment E2 --dataset mmmu --gpu 0
```

### 3. 查看结果

```bash
# 查看E1结果
python run.py --results --experiment E1

# 查看Phase 1结果
python run.py --results --experiment phase1
```

---

## 配置文件说明

所有配置文件位于 `configs/` 目录：

| 配置文件 | 用途 | 说明 |
|---------|------|------|
| `phase1.yaml` | Prompt策略对比 | 对比CoT/FewShot/Critic/Shape/RolePlay |
| `phase3.yaml` | 架构发现 (原始) | 使用AI2D+3ep评估 |
| `phase4_optimized.yaml` | 架构发现 (优化) | 使用MMMU+10ep，严格效率约束 |

---

## 统一输出格式

所有实验结果保存到 `results/` 目录，格式统一：

```
results/
├── phase1/
│   ├── results.json          # 结构化结果
│   ├── prompts/              # 保存的prompts
│   └── generated_code/       # 生成的代码
├── phase3/
│   └── discovery_YYYYMMDD_HHMMSS/
│       ├── top_architectures/
│       │   ├── arch_001/
│       │   │   ├── code.py
│       │   │   ├── config.json
│       │   │   └── results.json
│       │   └── ...
│       └── logs/
└── phase4/                   # 优化版结果
```

### 结果文件格式 (JSON)

```json
{
  "experiment_name": "phase3_architecture_discovery",
  "timestamp": "2026-02-21T15:30:00",
  "config": { ... },
  "top_architectures": [
    {
      "id": "arch_024",
      "reward": 0.952,
      "accuracy": 0.98,
      "flops": 40770000,
      "params": 12345678,
      "iteration": 82
    }
  ],
  "search_history": [ ... ]
}
```

---

## 团队协作分工

### 你可以分配的任务

#### 任务A: SOTA调研 (分配给同学/助教)
```bash
# 任务说明
1. 搜索关键词: "AI2D benchmark", "diagram understanding"
2. 查找CVPR/ICCV/EMNLP 2022-2024论文
3. 记录: 标题、方法、FLOPs、开源链接

# 产出
docs/sota_survey.md  (使用提供的模板)
```

#### 任务B: SOTA复现 (分配给有代码能力的同学)
```bash
# 任务说明
1. 从任务A列表中选择2-3个有开源代码的方法
2. 复现到 expv2/shared/baselines/sota_*.py
3. 确保接口统一，能跑通测试

# 产出
expv2/shared/baselines/sota_mcan.py
expv2/shared/baselines/sota_ban.py
```

#### 任务C: 数据分析 (分配给同学/助教)
```bash
# 任务说明
1. 整理所有实验结果到统一表格
2. 生成对比图表 (准确率柱状图、帕累托前沿图)
3. 维护GitHub文档

# 产出
results/analysis/comparison_table.csv
results/analysis/pareto_front.png
```

---

## 核心工作流程

### 优化实验流程 (推荐)

```
Step 1: 你改进评估器 (MMMU+10ep)
        ↓
Step 2: 你改进Reward函数 (效率权重提升)
        ↓
Step 3: 并行执行
        ├── 你: 运行Phase 4搜索
        └── 他人: SOTA调研 + 复现
        ↓
Step 4: 你运行全面评估 (新NAS vs 所有Baseline)
        ↓
Step 5: 他人整理数据 + 可视化
```

---

## 配置对比

### Phase 3 (原始) vs Phase 4 (优化)

| 配置项 | Phase 3 | Phase 4 | 改进说明 |
|--------|---------|---------|----------|
| **数据集** | AI2D | MMMU | 使用更难数据集 |
| **训练轮数** | 3 epochs | 10 epochs | 更充分训练 |
| **样本数** | 16-shot | 32-shot | 更多数据 |
| **效率权重** | 0.5 | 1.5 | 重视效率 |
| **FLOPs约束** | 无 | 10M硬约束 | 强制高效 |
| **网络深度** | max 6 | max 4 | 限制复杂度 |
| **隐藏维度** | max 1024 | max 512 | 限制参数量 |
| **早停** | 无 | 3epoch | 节省时间 |
| **时间限制** | 无 | 5分钟 | 防止卡住 |

---

## 成功标准

### 优化实验成功标准

| 指标 | 当前 | 目标 | 判断标准 |
|------|------|------|----------|
| MMMU准确率 | ~33% (NAS) | **>46%** | 超越FiLM |
| FLOPs | ~50M (NAS) | **<6.29M** | 比FiLM高效 |
| 帕累托前沿 | - | **左上区域** | 准确率↑效率↑ |

---

## 故障排除

### 问题1: MMMU评估太慢
```yaml
# 解决方案1: 降低配置
evaluator:
  train_epochs: 5      # 改为5
  max_training_time: 180  # 3分钟

# 解决方案2: 换用VSR
evaluator:
  dataset: "vsr"       # VSR比MMMU简单
```

### 问题2: 所有架构FLOPs仍然太高
```yaml
# 解决方案: 加强约束
reward:
  flops_constraint:
    max_flops: 5000000   # 降低到5M
    reject_if_exceed: true
  weights:
    efficiency: 2.0      # 进一步提升效率权重
```

### 问题3: 搜索结果不佳
```yaml
# 解决方案: 增加迭代数
iterations: 300          # 增加到300
search_strategy:
  stage1:
    iterations: 200      # 更多探索
  stage2:
    iterations: 100      # 更多优化
```

---

## 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-02-13 | 原始分散脚本 |
| 2.0 | 2026-02-21 | 统一框架，统一入口 |

---

## 相关文档

- `PROJECT_STATUS.md` - 项目整体进展
- `OPTIMIZATION_PLAN_V2.md` - 优化实验详细方案
- `EXPERIMENTS_SUMMARY.md` - 实验结果汇总
- `README.md` - 项目主页

---

*Framework Version: 2.0*
*Last Updated: 2026-02-21*
